import copy
import os
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import roma

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
import detectron2.data.transforms as T

from core.data.utils.io import load_image, resize_image
from core.models.meta_arch import load_sam3d_body

from core.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from core.utils import recursive_to
from torch.utils.data import default_collate
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_tensor


try:
    has_moge_env = True
    from moge.model.v2 import MoGeModel
except:
    has_moge_env = False
    print("Conda env. does not have MoGe installed!")

# Load the model from local.
def get_moge_model(ckpt_path):
    device = torch.device("cuda")
    moge_model = MoGeModel.from_pretrained(ckpt_path).to(device)
    return moge_model

class NoCollate:
    def __init__(self, data):
        self.data = data

def get_cam_intrinsics(model, batch):
    # We expect the image to be RGB already
    input_image = batch['img_ori'][0].data
    H, W, _ = input_image.shape

    input_image = torch.tensor(input_image / 255, dtype=torch.float32,
                                device=batch["img"].device).permute(2, 0, 1)
    # Infer w/ MoGe2
    model.eval()
    moge_data = model.infer(input_image)
    # get intrinsics
    intrinsics = denormalize_f(moge_data['intrinsics'].cpu().numpy(), H, W)
    v_focal = intrinsics[1, 1]
    # override hfov with v_focal
    intrinsics[0, 0] = v_focal
    intrinsics = (intrinsics).to(batch["img"])
    # add batch dim
    cam_intrinsics = intrinsics[None]
    return cam_intrinsics

def denormalize_f(norm_K, height, width):
    # Extract cx and cy from the normalized K matrix
    cx_norm = norm_K[0][2]  # c_x is at K[0][2]
    cy_norm = norm_K[1][2]  # c_y is at K[1][2]

    fx_norm = norm_K[0][0]  # Normalized fx
    fy_norm = norm_K[1][1]  # Normalized fy
    # s_norm = norm_K[0][1]   # Skew (usually 0)

    # Scale to absolute values
    fx_abs = fx_norm * width
    fy_abs = fy_norm * height
    cx_abs = cx_norm * width
    cy_abs = cy_norm * height
    # s_abs = s_norm * width
    s_abs = 0

    # Construct absolute K matrix
    abs_K = torch.tensor([
        [fx_abs, s_abs, cx_abs],
        [0.0, fy_abs, cy_abs],
        [0.0, 0.0, 1.0]
    ])
    return abs_K

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


class SAM3DBodyEstimatorHand:
    def __init__(
        self,
        checkpoint_path: str = "",
        proto_path: str = "",
        bbox_threshold: float = 0.5,
        detector_path: str = "",
        sam_path: str = "",
        use_mask: bool = False,
        use_triplet: bool = False,
        scale_factor = None,
        just_left_hand = False,
        use_face = False,
        moge_path: str = "",
        hand_crop_factor: float = 0.9,
        thresh_wrist_angle: float = 1.1
    ):
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.use_mask = use_mask
        self.scale_factor = scale_factor
        self.just_left_hand = just_left_hand
        self.hand_crop_factor = hand_crop_factor
        self.thresh_wrist_angle = thresh_wrist_angle

        # Initialize human detector
        self.use_detector = False
        if len(detector_path):
            self.detector = self.init_detector(detector_path, bbox_threshold)
            self.detector = self.detector.to(self.device)
            self.detector.eval()
            self.use_detector = True

        # Initialize SAM2 if needed
        if self.use_mask:
            self.sam_predictor = self.init_sam(sam_path)

        # Build SAM3D-Body model
        self.model, self.cfg = load_sam3d_body(checkpoint_path, proto_path,use_triplet=True, 
                                               use_twostage_for_hands=True, use_face=use_face)
        self.faces = self.model.head_pose.atlas.faces.numpy()
        self.model = self.model.to(self.device)
        self.model.eval()

        # is not using moge env. default to camerahmr
        if moge_path and has_moge_env:
            self.fov_estimator = get_moge_model(moge_path)
        else:
            self.fov_estimator = None
            print("No FOV estimator... Using the default FOV!")
        self._blank_hand_embeddings = None

    def init_detector(self, detector_path, threshold):
        DETECTRON_CFG = os.path.join(detector_path, "cascade_mask_rcnn_vitdet_h_75ep.py")
        DETECTRON_CKPT = os.path.join(detector_path, "model_final_f05665.pkl")

        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = threshold
        detector = instantiate(detectron2_cfg.model)
        checkpointer = DetectionCheckpointer(detector)
        checkpointer.load(DETECTRON_CKPT)

        return detector
    
    def init_sam(self, sam_path):
        checkpoint = f"{sam_path}/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        import sys
        sys.path.append(sam_path)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=self.device))
        return predictor
    
    def run_human_detection(
        self,
        img,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        default_to_full_image: bool = True,
    ):
        height, width = img.shape[:2]

        IMAGE_SIZE = 1024
        transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
        img_transformed = transforms(T.AugInput(img)).apply_image(img)
        img_transformed = torch.as_tensor(
            img_transformed.astype("float32").transpose(2, 0, 1)
        )
        inputs = {"image": img_transformed, "height": height, "width": width}

        with torch.no_grad():
            det_out = self.detector([inputs])

        det_instances = det_out[0]["instances"]
        valid_idx = (det_instances.pred_classes == det_cat_id) & (
            det_instances.scores > bbox_thr
        )
        if valid_idx.sum() == 0 and default_to_full_image:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)
        else:
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Sort boxes to keep a consistent output order
        sorted_indices = np.lexsort(
            (boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0])
        )  # shape: [len(boxes),]
        boxes = boxes[sorted_indices]
        return boxes
    
    def run_sam(self, img, boxes):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_predictor.set_image(img)
            all_masks, all_scores = [], []
            for i in range(boxes.shape[0]):
                # First prediction: bbox only
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes[[i]],
                    multimask_output=True,
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]

                mask_1 = masks[0]
                score_1 = scores[0]
                all_masks.append(mask_1)
                all_scores.append(score_1)

                # cv2.imwrite(os.path.join(save_dir, f"{os.path.basename(image_path)[:-4]}_mask_{i}.jpg"), (mask_1 * 255).astype(np.uint8))
            all_masks = np.stack(all_masks)
            all_scores = np.stack(all_scores)
        
        return all_masks, all_scores
    
    def _prepare_batch(
        self,
        img,
        transform,
        boxes,
        masks=None,
        masks_score=None,
        cam_int=None,
        get_full=False,
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

        if get_full:
            ############# Convert to full-image format #################
            full_size = 256
            _, img_full, center_full, scale_full = resize_image(
                img,
                full_size,
                batch["bbox_center"],
                batch["bbox_scale"],
            )
            bbox_min = center_full - scale_full * 0.5  # pyre-ignore
            bbox_max = center_full + scale_full * 0.5  # pyre-ignore
            bbox_full = np.concatenate([bbox_min, bbox_max], axis=1)
            bbox_full = np.clip(bbox_full, 0, full_size)

            batch_full = dict(
                img_full=to_tensor(img_full),
                full_bbox_center=center_full,
                full_bbox_scale=scale_full,
                full_bbox=bbox_full,
            )
            batch_full = default_collate([batch_full])
            batch.update(batch_full)

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

        if cam_int is not None:
            batch["cam_int"] = cam_int.to(batch["img"])
        else:
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
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
    ):
        """
        Perform model prediction in top-down format: assuming input is a full image.
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
        elif self.use_detector:
            # boxes = np.array([0, 0, width, height]).reshape(1, 4)
            if image_format == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                image_format = "bgr"
            print("Running object detector...")
            boxes = self.run_human_detection(
                img,
                det_cat_id,
                bbox_thr,
                nms_thr,
                default_to_full_image=False,
            )
            self.is_crop = True
        else:
            boxes = np.array([0, 0, width, height]).reshape(1, 4)

        # If there are no detected humans, don't run prediction
        if len(boxes) == 0:
            return []

        # The following models expect RGB images instead of BGR
        if image_format == "bgr":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get SAM2 mask if needed
        masks, masks_score = None, None
        if self.use_mask:
            masks, score = self.run_sam(img, boxes)
            # Stress test demo --> use the same bbox for all instances
            # boxes = np.concatenate([boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)], axis=0)[None, :].repeat(boxes.shape[0], axis=0)
    
        #################### Run model inference on an image ####################

        transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

        transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=self.hand_crop_factor),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

        batch = self._prepare_batch(img, transform, boxes, masks, masks_score)
        if self.cfg.MODEL.NAME == "promptable_threepo_triplet":
            batch['lhand_img'] = torch.zeros_like(batch['img'][:, :, :, :self.cfg.MODEL.HAND_IMAGE_SIZE[0], :self.cfg.MODEL.HAND_IMAGE_SIZE[1]])
            batch['rhand_img'] = torch.zeros_like(batch['img'][:, :, :, :self.cfg.MODEL.HAND_IMAGE_SIZE[0], :self.cfg.MODEL.HAND_IMAGE_SIZE[1]])

        batch = recursive_to(batch, "cuda")
        # triplet-triplet inference
        assert self.model.use_twostage_for_hands
        self.model._initialize_batch(batch)


        cam_int = get_cam_intrinsics(self.fov_estimator, batch).to(
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
        updated_batch = self.model.get_hand_box(pose_output_ab, batch, scale_factor=self.scale_factor)
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

        batch_lhand = self._prepare_batch(img, transform_hand, left_xyxy, cam_int=cam_int.clone())
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

        batch_rhand = self._prepare_batch(img, transform_hand, right_xyxy, cam_int=cam_int.clone())
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
        joint_rotations = self.model.head_pose.atlas(
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
        wrist_zero_rot_pose = lowarm_joint_rotations @ self.model.head_pose.atlas.lbs_fn_infos.joint_rotation[wrist_twist_joint_idxs]
        
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
            verts, j3d, jcoords, joint_global_rots, joint_params = self.model.head_pose.atlas(
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
                    "pred_global_rots": out['joint_global_rots'][idx],
                    "angle_diff": angle_difference[idx].cpu().numpy(),
                }
            )

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
