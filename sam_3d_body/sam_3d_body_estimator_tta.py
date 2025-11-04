import copy
from typing import Optional, Union

import numpy as np
import torch
import cv2
import roma

from sam_3d_body.data.utils.io import load_image

from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from sam_3d_body.utils import recursive_to
from torch.utils.data import default_collate
from torchvision.transforms import ToTensor



class NoCollate:
    def __init__(self, data):
        self.data = data

def rotation_angle_difference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the angle difference (magnitude) between two batches of SO(3) rotation matrices.
    Args:
        A: Tensor of shape (*, 3, 3), batch of rotation matrices.
        B: Tensor of shape (*, 3, 3), batch of rotation matrices.
    Returns:
        Tensor of shape (*,), angle differences in radians.
    """
    # Compute relative rotation matrix
    R_rel = torch.matmul(A, B.transpose(-2, -1))  # (B, 3, 3)
    # Compute trace of relative rotation
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]  # (B,)
    # Compute angle using the trace formula
    cos_theta = (trace - 1) / 2
    # Clamp for numerical stability
    cos_theta_clamped = torch.clamp(cos_theta, -1.0, 1.0)
    # Compute angle difference
    angle = torch.acos(cos_theta_clamped)
    return angle

class SAM3DBodyEstimatorTTA:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector = None,
        human_segmentor = None,
        fov_estimator = None,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.thresh_wrist_angle = 1.1
        self._blank_hand_embeddings = None

        self.faces = self.model.head_pose.faces.cpu().numpy()
        self.model.eval()

        if self.detector is None:
            print("No human detector is used...")
        if self.sam is None:
            print("Mask-condition inference is not supported...")
        if self.fov_estimator is None:
            print("No FOV estimator... Using the default FOV!")
        
        self.transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam_3d_body.build_models import load_sam_3d_body_hf

        model, model_cfg = load_sam_3d_body_hf(model_id, **kwargs)
        return cls(model, model_cfg, **kwargs)

    def _prepare_batch(
        self,
        img,
        transform,
        boxes,
        masks=None,
        masks_score=None,
        cam_int=None,
    ):
        height, width = img.shape[:2]

        # construct batch data samples
        data_list = []
        for idx in range(boxes.shape[0]):
            data_info = dict(img=img)
            data_info["bbox"] = boxes[idx]  # shape (4,)
            data_info["bbox_format"] = "xyxy"

            if masks is not None:
                data_info["mask"] = masks[idx].copy()
                if masks_score is not None:
                    data_info["mask_score"] = masks_score[idx]
                else:
                    data_info["mask_score"] = np.array(1.0, dtype=np.float32)
            else:
                data_info["mask"] = np.zeros((height, width, 1), dtype=np.uint8)
                data_info["mask_score"] = np.array(0.0, dtype=np.float32)

            data_list.append(transform(data_info))

        batch = default_collate(data_list)

        max_num_person = batch["img"].shape[0]
        for key in [
            "img",
            "img_size",
            "ori_img_size",
            "bbox_center",
            "bbox_scale",
            "bbox",
            "affine_trans",
            "mask",
            "mask_score",
        ]:
            if key in batch:
                batch[key] = batch[key].unsqueeze(0).float()
        if "mask" in batch:
            batch["mask"] = batch["mask"].unsqueeze(2)
        batch["person_valid"] = torch.ones((1, max_num_person))

        batch["cam_int"] = torch.tensor(
            [[[(height ** 2 + width ** 2) ** 0.5, 0, width / 2.],
            [0, (height ** 2 + width ** 2) ** 0.5, height / 2.],
            [0, 0, 1]]],
        ).to(batch["img"])
        
        batch['img_ori'] = [NoCollate(img)]
        return batch

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.
        
        Args:
            img: Input image (path or numpy array)
            bboxes: Optional pre-computed bounding boxes
            masks: Optional pre-computed masks (numpy array). If provided, SAM2 will be skipped.
            det_cat_id: Detection category ID
            bbox_thr: Bounding box threshold
            nms_thr: NMS threshold
        """

        # clear all cached results
        self.batch = None
        self.image_embeddings = None
        self.output = None
        self.prev_prompt = []
        torch.cuda.empty_cache()

        if type(img) == str:
            img = load_image(img, backend="cv2", image_format="bgr")
            image_format = "bgr"
        else:
            print ("####### Please make sure the input image is in RGB format")
            image_format = "rgb"
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.detector is not None:
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes = self.detector.run_human_detection(
                img,
                det_cat_id=det_cat_id,
                bbox_thr=bbox_thr,
                nms_thr=nms_thr,
                default_to_full_image=False,
            )
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
            self.is_crop = False

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Handle masks - either provided externally or generated via SAM2
        masks_score = None
        if masks is not None:
            # Use provided masks - ensure they match the number of detected boxes
            print(f"Using provided masks: {masks.shape}")
            if len(masks.shape) == 2:
                # Single mask - expand to match number of boxes
                masks = np.expand_dims(masks, axis=0)
                masks = np.repeat(masks, len(boxes), axis=0)
            masks_score = np.ones(len(masks), dtype=np.float32)  # Set high confidence for provided masks
        elif use_mask and self.sam is not None:
            # Generate masks using SAM2
            masks, masks_score = self.sam.run_sam(img, boxes)
            # TODO: clean-up needed, move to notebook
            # Stress test demo --> use the same bbox for all instances
            # boxes = np.concatenate([boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)], axis=0)[None, :].repeat(boxes.shape[0], axis=0)
        else:
            masks, masks_score = None, None


        #################### Construct batch data samples ####################
        batch = self._prepare_batch(img, self.transform, boxes, masks, masks_score)

        #################### Run model inference on an image ####################

        if self.cfg.MODEL.NAME == "promptable_threepo_triplet":
            batch['lhand_img'] = torch.zeros_like(batch['img'][:, :, :, :self.cfg.MODEL.HAND_IMAGE_SIZE[0], :self.cfg.MODEL.HAND_IMAGE_SIZE[1]])
            batch['rhand_img'] = torch.zeros_like(batch['img'][:, :, :, :self.cfg.MODEL.HAND_IMAGE_SIZE[0], :self.cfg.MODEL.HAND_IMAGE_SIZE[1]])

        batch = recursive_to(batch, "cuda")
        assert self.cfg.MODEL.NAME == "promptable_threepo_triplet"
        self.model._initialize_batch(batch)

        if self.fov_estimator is not None:
            input_image = batch['img_ori'][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()

        ## Stage 1: Body + blank hands
        hand_embeddings = (
            None
            if self._blank_hand_embeddings is None
            else self._blank_hand_embeddings.repeat(batch["img"].shape[0], 1, 1, 1)
        )
        pose_output_ab = self.model.inference_pose_branch(
            batch,
            image_embeddings=None,
            hand_embeddings=hand_embeddings,
        )
        lhand_blank_embeddings, rhand_blank_embeddings = torch.chunk(pose_output_ab["hand_embeddings"], 2, dim=2)

        ## Stage 2: Body + hand crops
        updated_batch = self.model.get_hand_box(pose_output_ab, batch, scale_factor=None)
        pose_output = self.model.inference_pose_branch(
            updated_batch,
            image_embeddings=pose_output_ab["image_embeddings"],
            hand_embeddings=None,
        )
        lhand_embeddings, rhand_embeddings = torch.chunk(pose_output["hand_embeddings"], 2, dim=2)
        ori_local_wrist_rotmat = roma.euler_to_rotmat(
            "XZY",
            pose_output['atlas']['body_pose'][:, [41, 43, 42, 31, 33, 32]].unflatten(1, (2, 3))
        )

        # Stage 3: Re-run with each hand
        ## Left...
        left_xyxy = np.concatenate([
            (updated_batch['left_center'][:, 0] - updated_batch['left_scale'][:, 0] / 2).reshape(-1, 1),
            (updated_batch['left_center'][:, 1] - updated_batch['left_scale'][:, 1] / 2).reshape(-1, 1),
            (updated_batch['left_center'][:, 0] + updated_batch['left_scale'][:, 0] / 2).reshape(-1, 1),
            (updated_batch['left_center'][:, 1] + updated_batch['left_scale'][:, 1] / 2).reshape(-1, 1),
        ], axis=1)
        print("Body, going into left...", left_xyxy)

        batch_lhand = self._prepare_batch(img, self.transform_hand, left_xyxy, cam_int=cam_int.clone())
        batch_lhand = recursive_to(batch_lhand, "cuda")
        lhand_output = self.model.inference_pose_branch(
            batch_lhand,
            image_embeddings=None,
            hand_embeddings=torch.cat([lhand_embeddings, rhand_blank_embeddings], dim=2),
        )
    
        ## Right...
        right_xyxy = np.concatenate([
            (updated_batch['right_center'][:, 0] - updated_batch['right_scale'][:, 0] / 2).reshape(-1, 1),
            (updated_batch['right_center'][:, 1] - updated_batch['right_scale'][:, 1] / 2).reshape(-1, 1),
            (updated_batch['right_center'][:, 0] + updated_batch['right_scale'][:, 0] / 2).reshape(-1, 1),
            (updated_batch['right_center'][:, 1] + updated_batch['right_scale'][:, 1] / 2).reshape(-1, 1),
        ], axis=1)
        print("Body, going into right...", right_xyxy)

        batch_rhand = self._prepare_batch(img, self.transform_hand, right_xyxy, cam_int=cam_int.clone())
        batch_rhand = recursive_to(batch_rhand, "cuda")
        rhand_output = self.model.inference_pose_branch(
            batch_rhand,
            image_embeddings=None,
            hand_embeddings=torch.cat([lhand_blank_embeddings, rhand_embeddings], dim=2),
        )

        # Drop in hand pose
        left_hand_pose_params = lhand_output['atlas']['hand'][:, :54]
        right_hand_pose_params = rhand_output['atlas']['hand'][:, 54:]
        updated_hand_pose = torch.cat([left_hand_pose_params, right_hand_pose_params], dim=1)
            
        # Drop in hand scales
        updated_scale = pose_output['atlas']['scale'].clone()
        updated_scale[:, 9] = lhand_output['atlas']['scale'][:, 9]
        updated_scale[:, 8] = rhand_output['atlas']['scale'][:, 8]
            
        print("Doing IK...")
        # First, forward just FK
        joint_rotations = self.model.head_pose.mohr_forward(
            global_trans=pose_output['atlas']['global_rot'] * 0,
            global_rot=pose_output['atlas']['global_rot'],
            body_pose_params=pose_output['atlas']['body_pose'],
            hand_pose_params=updated_hand_pose,
            scale_params=updated_scale,
            shape_params=pose_output['atlas']['shape'],
            expr_params=pose_output['atlas']['face'],
            return_joint_rotations=True,
        )[1]

        # Get lowarm
        lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda() # left, right
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs] # B x 2 x 3 x 3
        
        # Get zero-wrist pose
        wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda() # left, right
        wrist_zero_rot_pose = lowarm_joint_rotations @ self.model.head_pose.joint_rotation[wrist_twist_joint_idxs]
        
        # Get globals from left & right
        left_joint_global_rots = lhand_output['atlas']['joint_global_rots']
        right_joint_global_rots = rhand_output['atlas']['joint_global_rots']
        pred_global_wrist_rotmat = torch.stack([
            left_joint_global_rots[:, 78],
            right_joint_global_rots[:, 42],
        ], dim=1)

        # Now we want to get the local poses that lead to the wrist being pred_global_wrist_rotmat
        fused_local_wrist_rotmat = torch.einsum('kabc,kabd->kadc', pred_global_wrist_rotmat, wrist_zero_rot_pose)
        wrist_xzy = roma.rotmat_to_euler("XZY", fused_local_wrist_rotmat)
        
        # Put it in.
        angle_difference = rotation_angle_difference(ori_local_wrist_rotmat, fused_local_wrist_rotmat) # B x 2 x 3 x3
        valid_angle = angle_difference < self.thresh_wrist_angle
        valid_angle = valid_angle.unsqueeze(-1)

        body_pose = pose_output['atlas']['body_pose'][:, [41, 43, 42, 31, 33, 32]].unflatten(1, (2, 3))
        updated_body_pose = torch.where(valid_angle, wrist_xzy, body_pose)
        pose_output['atlas']['body_pose'][:, [41, 43, 42, 31, 33, 32]] = updated_body_pose.flatten(1, 2)

        hand_pose = pose_output['atlas']['hand'].unflatten(1, (2, 54))
        pose_output['atlas']['hand'] = torch.where(valid_angle, updated_hand_pose.unflatten(1, (2, 54)), hand_pose).flatten(1, 2)

        hand_scale = torch.stack([pose_output['atlas']['scale'][:, 9], pose_output['atlas']['scale'][:, 8]], dim=1)
        updated_hand_scale = torch.stack([updated_scale[:, 9], updated_scale[:, 8]], dim=1)
        masked_hand_scale = torch.where(valid_angle.squeeze(-1), updated_hand_scale, hand_scale)
        pose_output['atlas']['scale'][:, 9] = masked_hand_scale[:, 0]
        pose_output['atlas']['scale'][:, 8] = masked_hand_scale[:, 1]
        
        print("Done with IK...")
            
        # Re-run forward
        with torch.no_grad():
            verts, j3d, jcoords, joint_global_rots, joint_params = self.model.head_pose.mohr_forward(
                global_trans=pose_output['atlas']['global_rot'] * 0,
                global_rot=pose_output['atlas']['global_rot'],
                body_pose_params=pose_output['atlas']['body_pose'],
                hand_pose_params=pose_output['atlas']['hand'],
                scale_params=pose_output['atlas']['scale'],
                shape_params=pose_output['atlas']['shape'],
                expr_params=pose_output['atlas']['face'],
                return_keypoints=True,
                return_joint_coords=True,
                return_joint_rotations=True,
                return_joint_params=True,
            )
            j3d = j3d[:, :70]  # 308 --> 70 keypoints
            verts[..., [1, 2]] *= -1  # Camera system difference
            j3d[..., [1, 2]] *= -1  # Camera system difference
            jcoords[..., [1, 2]] *= -1
            pose_output['atlas']['pred_keypoints_3d'] = j3d
            pose_output['atlas']['pred_vertices'] = verts
            pose_output['atlas']['pred_joint_coords'] = jcoords
            pose_output['atlas']['pred_pose_raw'][...] = 0

        out = pose_output["atlas"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(batch["img"].shape[1]):
            all_out.append(
                {
                    "bbox": batch["bbox"][0, idx].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_vertices": out["pred_vertices"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_pose_raw": out["pred_pose_raw"][idx],
                    "global_rot": out["global_rot"][idx],
                    "body_pose_params": out["body_pose"][idx],
                    "hand_pose_params": out["hand"][idx],
                    "scale_params": out["scale"][idx],
                    "shape_params": out["shape"][idx],
                    "expr_params": out["face"][idx],
                    "mask": masks[idx] if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][idx],
                }
            )

            pred_keypoints_3d_proj = (
                all_out[-1]["pred_keypoints_3d"] + all_out[-1]["pred_cam_t"]
            )
            # pred_keypoints_3d_proj[:, [1, 2]] *= -1
            pred_keypoints_3d_proj[:, [0, 1]] *= all_out[-1]["focal_length"]
            pred_keypoints_3d_proj[:, [0, 1]] = (
                pred_keypoints_3d_proj[:, [0, 1]]
                + np.array([width / 2, height / 2]) * pred_keypoints_3d_proj[:, [2]]
            )
            pred_keypoints_3d_proj[:, :2] = (
                pred_keypoints_3d_proj[:, :2] / pred_keypoints_3d_proj[:, [2]]
            )
            all_out[-1]["pred_keypoints_2d"] = pred_keypoints_3d_proj[:, :2]

            if self.cfg.MODEL.NAME == "promptable_threepo_triplet":
                all_out[-1]["lhand_bbox"] = np.array([
                    (updated_batch['left_center'][idx][0] - updated_batch['left_scale'][idx][0] / 2).item(),
                    (updated_batch['left_center'][idx][1] - updated_batch['left_scale'][idx][1] / 2).item(),
                    (updated_batch['left_center'][idx][0] + updated_batch['left_scale'][idx][0] / 2).item(),
                    (updated_batch['left_center'][idx][1] + updated_batch['left_scale'][idx][1] / 2).item(),
                ])
                all_out[-1]["rhand_bbox"] = np.array([
                    (updated_batch['right_center'][idx][0] - updated_batch['right_scale'][idx][0] / 2).item(),
                    (updated_batch['right_center'][idx][1] - updated_batch['right_scale'][idx][1] / 2).item(),
                    (updated_batch['right_center'][idx][0] + updated_batch['right_scale'][idx][0] / 2).item(),
                    (updated_batch['right_center'][idx][1] + updated_batch['right_scale'][idx][1] / 2).item(),
                ])

        return all_out
