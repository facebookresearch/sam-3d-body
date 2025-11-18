import copy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
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
    
def fix_wrist_euler(wrist_xzy, limits_x=(-2.2, 1.0), limits_z=(-2.2, 1.5), limits_y=(-1.2, 1.5)):
    """
    wrist_xzy: B x 2 x 3 (X, Z, Y angles)
    Returns: Fixed angles within joint limits
    """
    x, z, y = wrist_xzy[..., 0], wrist_xzy[..., 1], wrist_xzy[..., 2]
    
    x_alt = torch.atan2(torch.sin(x + torch.pi), torch.cos(x + torch.pi))
    z_alt = torch.atan2(torch.sin(-(z + torch.pi)), torch.cos(-(z + torch.pi)))
    y_alt = torch.atan2(torch.sin(y + torch.pi), torch.cos(y + torch.pi))
    
    # Calculate L2 violation distance
    def calc_violation(val, limits):
        below = torch.clamp(limits[0] - val, min=0.0)
        above = torch.clamp(val - limits[1], min=0.0)
        return below**2 + above**2
    
    violation_orig = (
        calc_violation(x, limits_x) +
        calc_violation(z, limits_z) +
        calc_violation(y, limits_y)
    )
    
    violation_alt = (
        calc_violation(x_alt, limits_x) +
        calc_violation(z_alt, limits_z) +
        calc_violation(y_alt, limits_y)
    )
    
    # Use alternative where it has lower L2 violation
    use_alt = violation_alt < violation_orig
    
    # Stack alternative and apply mask
    wrist_xzy_alt = torch.stack([x_alt, z_alt, y_alt], dim=-1)
    result = torch.where(use_alt.unsqueeze(-1), wrist_xzy_alt, wrist_xzy)
    
    return result

class SAM3DBodyEstimatorUnified:
    def __init__(
        self,
        sam_3d_body_model,
        model_cfg,
        human_detector = None,
        human_segmentor = None,
        fov_estimator = None,
        prompt_wrists = True,
        use_hand_box = True,
    ):
        self.device = sam_3d_body_model.device
        self.model, self.cfg = sam_3d_body_model, model_cfg
        self.detector = human_detector
        self.sam = human_segmentor
        self.fov_estimator = fov_estimator
        self.prompt_wrists = prompt_wrists
        self.use_hand_box = use_hand_box
        self.thresh_wrist_angle = 1.4   # we used 1.1 before

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

    def get_hand_box(self, pose_output, batch, scale_factor=None):
        # get the left and right hand boxes and images from the predictions
        left_hand_joint_idx = torch.arange(42, 63)  # 21 joints
        right_hand_joint_idx = torch.arange(21, 42)  # 21 joints
        left_wrist_idx = 62
        right_wrist_idx = 41

        # Get hand crops
        ## First, decode images
        batch_size, num_person = batch["img"].shape[:2]
        full_imgs = [
            img.data for img in batch["img_ori"] for _ in range(num_person)
        ]  # cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        full_imgs_hw = torch.LongTensor(
            [img.shape[:2] for img in full_imgs]
        ).numpy()

        if scale_factor is None:
            scale_factor = 2.0
            scale_factor = 1.6
        hand_body_ratio = 0.10
        square_box = True
        
        ## Get hand 2d KPS
        left_hand_kps_2d = pose_output["atlas"]["pred_keypoints_2d"][
            :, left_hand_joint_idx
        ]
        right_hand_kps_2d = pose_output["atlas"]["pred_keypoints_2d"][
            :, right_hand_joint_idx
        ]

        ## Get minmaxes (xy)
        left_min = left_hand_kps_2d.amin(dim=1).cpu().long().numpy()
        left_max = left_hand_kps_2d.amax(dim=1).cpu().long().numpy()
        right_min = right_hand_kps_2d.amin(dim=1).cpu().long().numpy()
        right_max = right_hand_kps_2d.amax(dim=1).cpu().long().numpy()
        left_min[:, 0] = np.clip(left_min[:, 0], a_min=0, a_max=full_imgs_hw[:, 1])
        left_min[:, 1] = np.clip(left_min[:, 1], a_min=0, a_max=full_imgs_hw[:, 0])
        left_max[:, 0] = np.clip(left_max[:, 0], a_min=0, a_max=full_imgs_hw[:, 1])
        left_max[:, 1] = np.clip(left_max[:, 1], a_min=0, a_max=full_imgs_hw[:, 0])
        right_min[:, 0] = np.clip(
            right_min[:, 0], a_min=0, a_max=full_imgs_hw[:, 1]
        )
        right_min[:, 1] = np.clip(
            right_min[:, 1], a_min=0, a_max=full_imgs_hw[:, 0]
        )
        right_max[:, 0] = np.clip(
            right_max[:, 0], a_min=0, a_max=full_imgs_hw[:, 1]
        )
        right_max[:, 1] = np.clip(
            right_max[:, 1], a_min=0, a_max=full_imgs_hw[:, 0]
        )
        ## Get center & scale
        left_center = (left_max + left_min) / 2
        right_center = (right_max + right_min) / 2
        left_scale = (left_max - left_min) * scale_factor
        right_scale = (right_max - right_min) * scale_factor
        if square_box:
            left_scale[...] = left_scale.max(axis=1)[:, None]
            right_scale[...] = right_scale.max(axis=1)[:, None]
        ## Stack

        batch['left_scale'] = left_scale
        batch['left_center'] = left_center
        batch['right_scale'] = right_scale
        batch['right_center'] = right_center
        
        return batch

    @torch.no_grad()
    def process_one_image(
        self,
        img: Union[str, np.ndarray],
        bboxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        cam_int: Optional[np.ndarray] = None,
        det_cat_id: int = 0,
        bbox_thr: float = 0.5,
        nms_thr: float = 0.3,
        use_mask: bool = False,
        prompt_wrists_type: str = "v3",
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
            assert bboxes is not None, "Mask-conditioned inference requires bboxes input!"
            masks = masks.reshape(-1, height, width, 1).astype(np.uint8)
            masks_score = np.ones(len(masks), dtype=np.float32)  # Set high confidence for provided masks
            use_mask = True
        elif use_mask and self.sam is not None:
            print("Running SAM to get mask from bbox...")
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

        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)

        if cam_int is not None:
            print("Using provided camera intrinsics...")
            cam_int = cam_int.to(batch["img"])
            batch["cam_int"] = cam_int.clone()
        elif self.fov_estimator is not None:
            print("Running FOV estimator ...")
            input_image = batch['img_ori'][0].data
            cam_int = self.fov_estimator.get_cam_intrinsics(input_image).to(
                batch["img"]
            )
            batch["cam_int"] = cam_int.clone()
        else:
            cam_int = batch["cam_int"].clone()

        ## Stage 1: Body
        self.model.hand_batch_idx = []
        self.model.body_batch_idx = list(range(batch["img"].shape[1]))
        self.model.disable_hand = True
        self.model.disable_body = False
        pose_output, full_output = self.model.forward_step(batch)
        batch = self.get_hand_box(pose_output, batch, scale_factor=None)
        ori_local_wrist_rotmat = roma.euler_to_rotmat(
            "XZY",
            pose_output['atlas']['body_pose'][:, [41, 43, 42, 31, 33, 32]].unflatten(1, (2, 3))
        )

        if self.use_hand_box:
            # TODO: Assuming square crop into backbone
            pred_left_hand_box = pose_output["atlas"]["hand_box"][:, 0].detach().cpu().numpy() * self.cfg.MODEL.IMAGE_SIZE[0]
            pred_right_hand_box = pose_output["atlas"]["hand_box"][:, 1].detach().cpu().numpy() * self.cfg.MODEL.IMAGE_SIZE[0]
            pred_left_hand_logits = pose_output["atlas"]["hand_logits"][:, 0].detach().cpu().numpy()
            pred_right_hand_logits = pose_output["atlas"]["hand_logits"][:, 1].detach().cpu().numpy()
            
            # Change boxes into squares
            batch['left_center'] = pred_left_hand_box[:, :2]
            batch['left_scale'] = pred_left_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
            batch['right_center'] = pred_right_hand_box[:, :2]
            batch['right_scale'] = pred_right_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
            
            # Crop to full. batch["affine_trans"] is full-to-crop, right application
            batch['left_scale'] = batch['left_scale'] / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
            batch['right_scale'] = batch['right_scale'] / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
            batch['left_center'] = (batch['left_center'] - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
            batch['right_center'] = (batch['right_center'] - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]

        # Stage 3: Re-run with each hand
        self.model.hand_batch_idx = list(range(batch["img"].shape[1]))
        self.model.body_batch_idx = []
        self.model.disable_hand = False
        self.model.disable_body = True
        ## Left...
        left_xyxy = np.concatenate([
            (batch['left_center'][:, 0] - batch['left_scale'][:, 0] * 1 / 2).reshape(-1, 1),
            (batch['left_center'][:, 1] - batch['left_scale'][:, 1] * 1 / 2).reshape(-1, 1),
            (batch['left_center'][:, 0] + batch['left_scale'][:, 0] * 1 / 2).reshape(-1, 1),
            (batch['left_center'][:, 1] + batch['left_scale'][:, 1] * 1 / 2).reshape(-1, 1),
        ], axis=1)
        print("Body, going into left; flipping...", left_xyxy)
        
        # Flip image & box
        flipped_img = img[:, ::-1]
        tmp = left_xyxy.copy()
        left_xyxy[:, 0] = width - tmp[:, 2] - 1
        left_xyxy[:, 2] = width - tmp[:, 0] - 1

        batch_lhand = self._prepare_batch(flipped_img, self.transform_hand, left_xyxy, cam_int=cam_int.clone())
        batch_lhand = recursive_to(batch_lhand, "cuda")
        lhand_output = self.model.forward_step(batch_lhand)[0]
        lhand_output['atlas'] = lhand_output['atlas_hand']
        
        # Unflip output
        ## Flip scale
        ### Get MHR values
        scale_r_hands_mean = -0.1798856556415558
        scale_l_hands_mean = -0.18402963876724243
        scale_r_hands_std = 0.04739458113908768
        scale_l_hands_std = 0.04183576628565788
        ### Apply
        lhand_output['atlas']['scale'][:, 9] = ((scale_r_hands_mean + scale_r_hands_std * lhand_output['atlas']['scale'][:, 8]) - scale_l_hands_mean) / scale_l_hands_std
        ## Get the right hand global rotation, flip it, put it in as left.
        lhand_output['atlas']['joint_global_rots'][:, 78] = lhand_output['atlas']['joint_global_rots'][:, 42].clone()
        lhand_output['atlas']['joint_global_rots'][:, 78, [1, 2], :] *= -1
        ### Flip hand pose
        lhand_output['atlas']['hand'][:, :54] = lhand_output['atlas']['hand'][:, 54:]
        ### Unflip box
        batch_lhand['bbox_center'][:, :, 0] = width - batch_lhand['bbox_center'][:, :, 0] - 1
    
        ## Right...
        right_xyxy = np.concatenate([
            (batch['right_center'][:, 0] - batch['right_scale'][:, 0] * 1 / 2).reshape(-1, 1),
            (batch['right_center'][:, 1] - batch['right_scale'][:, 1] * 1 / 2).reshape(-1, 1),
            (batch['right_center'][:, 0] + batch['right_scale'][:, 0] * 1 / 2).reshape(-1, 1),
            (batch['right_center'][:, 1] + batch['right_scale'][:, 1] * 1 / 2).reshape(-1, 1),
        ], axis=1)
        print("Body, going into right...", right_xyxy)

        batch_rhand = self._prepare_batch(img, self.transform_hand, right_xyxy, cam_int=cam_int.clone())
        batch_rhand = recursive_to(batch_rhand, "cuda")
        rhand_output = self.model.forward_step(batch_rhand)[0]
        rhand_output['atlas'] = rhand_output['atlas_hand']

        # Now, get some criteria for whehter to replace.
        ## CRITERIA 1: LOCAL WRIST POSE DIFFERENCE
        joint_rotations = pose_output['atlas']['joint_global_rots']
        ### Get lowarm
        lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda() # left, right
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs] # B x 2 x 3 x 3
        ### Get zero-wrist pose
        wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda() # left, right
        wrist_zero_rot_pose = lowarm_joint_rotations @ self.model.head_pose.joint_rotation[wrist_twist_joint_idxs]
        ### Get globals from left & right
        left_joint_global_rots = lhand_output['atlas']['joint_global_rots']
        right_joint_global_rots = rhand_output['atlas']['joint_global_rots']
        pred_global_wrist_rotmat = torch.stack([
            left_joint_global_rots[:, 78],
            right_joint_global_rots[:, 42],
        ], dim=1)
        ### Now we want to get the local poses that lead to the wrist being pred_global_wrist_rotmat
        fused_local_wrist_rotmat = torch.einsum('kabc,kabd->kadc', pred_global_wrist_rotmat, wrist_zero_rot_pose)
        ### What's the angle difference?
        angle_difference = rotation_angle_difference(ori_local_wrist_rotmat, fused_local_wrist_rotmat) # B x 2 x 3 x3
        angle_difference_valid_mask = angle_difference < self.thresh_wrist_angle
        ## CRITERIA 2: hand box size
        hand_box_size_thresh = 64
        hand_box_size_valid_mask = torch.stack([
            (batch_lhand['bbox_scale'].flatten(0, 1) > hand_box_size_thresh).all(dim=1),
            (batch_rhand['bbox_scale'].flatten(0, 1) > hand_box_size_thresh).all(dim=1),
        ], dim=1)
        ## CRITERIA 3: all hand 2D KPS (including wrist) inside of box.
        hand_kps2d_thresh = 0.5
        # hand_kps2d_thresh = 99
        hand_kps2d_valid_mask = torch.stack([
            lhand_output['atlas']['pred_keypoints_2d_cropped'].abs().amax(dim=(1, 2)) < hand_kps2d_thresh,
            rhand_output['atlas']['pred_keypoints_2d_cropped'].abs().amax(dim=(1, 2)) < hand_kps2d_thresh,
        ], dim=1)
        ## CRITERIA 4: 2D wrist distance.
        hand_wrist_kps2d_thresh = 0.25
        # hand_wrist_kps2d_thresh = 99
        kps_right_wrist_idx = 41
        kps_left_wrist_idx = 62
        right_kps_full = rhand_output['atlas']['pred_keypoints_2d'][:, [kps_right_wrist_idx]].clone()
        left_kps_full = lhand_output['atlas']['pred_keypoints_2d'][:, [kps_right_wrist_idx]].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1 # Flip left hand
        body_right_kps_full = pose_output['atlas']['pred_keypoints_2d'][:, [kps_right_wrist_idx]].clone()
        body_left_kps_full = pose_output['atlas']['pred_keypoints_2d'][:, [kps_left_wrist_idx]].clone()
        right_kps_dist = (right_kps_full - body_right_kps_full).flatten(0, 1).norm(dim=-1) / batch_lhand['bbox_scale'].flatten(0, 1)[:, 0]
        left_kps_dist = (left_kps_full - body_left_kps_full).flatten(0, 1).norm(dim=-1) / batch_rhand['bbox_scale'].flatten(0, 1)[:, 0]
        hand_wrist_kps2d_valid_mask = torch.stack([
            left_kps_dist < hand_wrist_kps2d_thresh,
            right_kps_dist < hand_wrist_kps2d_thresh,
        ], dim=1)
        ## Left-right
        hand_valid_mask = (
            angle_difference_valid_mask
            & hand_box_size_valid_mask
            & hand_kps2d_valid_mask
            & hand_wrist_kps2d_valid_mask
        )

        if self.prompt_wrists:
            # TODO: check rotation_angle_difference beforehand.
            # TODO: Left first or right first? Does it matter?
            # TODO: Remember, we have keypoint confidences. For hand crops as well.
            self.model.hand_batch_idx = []
            self.model.body_batch_idx = list(range(batch["img"].shape[1]))
            self.model.disable_hand = True
            self.model.disable_body = False
            
            # Get right & left keypoints from crops; full image. Each are B x 1 x 2
            kps_right_wrist_idx = 41
            kps_left_wrist_idx = 62
            right_kps_full = rhand_output['atlas']['pred_keypoints_2d'][:, [kps_right_wrist_idx]].clone()
            left_kps_full = lhand_output['atlas']['pred_keypoints_2d'][:, [kps_right_wrist_idx]].clone()
            left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1 # Flip left hand
            
            # Next, get them to crop-normalized space.
            right_kps_crop = self.model._full_to_crop(batch, right_kps_full)
            left_kps_crop = self.model._full_to_crop(batch, left_kps_full)

            # Get right & left keypoints from crops; full image. Each are B x 1 x 2
            kps_right_elbow_idx = 8
            kps_left_elbow_idx = 7
            right_kps_elbow_full = pose_output['atlas']['pred_keypoints_2d'][:, [kps_right_elbow_idx]].clone()
            left_kps_elbow_full = pose_output['atlas']['pred_keypoints_2d'][:, [kps_left_elbow_idx]].clone()
            
            # Next, get them to crop-normalized space.
            right_kps_elbow_crop = self.model._full_to_crop(batch, right_kps_elbow_full)
            left_kps_elbow_crop = self.model._full_to_crop(batch, left_kps_elbow_full)
    
            # Assemble them into keypoint prompts
            keypoint_prompt = torch.cat(
                [right_kps_crop, left_kps_crop, right_kps_elbow_crop, left_kps_elbow_crop], dim=1
            )
            keypoint_prompt = torch.cat([keypoint_prompt, keypoint_prompt[..., [-1]]], dim=-1)
            keypoint_prompt[:, 0, -1] = kps_right_wrist_idx
            keypoint_prompt[:, 1, -1] = kps_left_wrist_idx
            keypoint_prompt[:, 2, -1] = kps_right_elbow_idx
            keypoint_prompt[:, 3, -1] = kps_left_elbow_idx
            
            if keypoint_prompt.shape[0] > 1:
                # Replace invalid keypoints to dummy prompts
                invalid_prompt = (
                    (keypoint_prompt[..., 0] < -0.5) |
                    (keypoint_prompt[..., 0] > 0.5) |
                    (keypoint_prompt[..., 1] < -0.5) |
                    (keypoint_prompt[..., 1] > 0.5) |
                    (~hand_valid_mask[..., [1, 0, 1, 0]])
                ).unsqueeze(-1)
                dummy_prompt = torch.zeros((1, 1, 3)).to(keypoint_prompt)
                dummy_prompt[:, :, -1] = -2
                keypoint_prompt[:, :, :2] = torch.clamp(
                    keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
                )  # [-0.5, 0.5] --> [0, 1]
                keypoint_prompt = torch.where(invalid_prompt, dummy_prompt, keypoint_prompt)
            else:
                # Only keep valid keypoints
                valid_keypoint = (
                    torch.all((keypoint_prompt[:, :, :2] > -0.5) & (keypoint_prompt[:, :, :2] < 0.5), dim=2)
                    & hand_valid_mask[..., [1, 0, 1, 0]]
                ).squeeze()
                keypoint_prompt = keypoint_prompt[:, valid_keypoint]
                keypoint_prompt[:, :, :2] = torch.clamp(
                    keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
                )  # [-0.5, 0.5] --> [0, 1]
            
            if len(keypoint_prompt):
                pose_output, _ = self._prompt_wrists(batch, pose_output, keypoint_prompt)

        # Drop in hand pose
        left_hand_pose_params = lhand_output['atlas']['hand'][:, :54]
        right_hand_pose_params = rhand_output['atlas']['hand'][:, 54:]
        updated_hand_pose = torch.cat([left_hand_pose_params, right_hand_pose_params], dim=1)
            
        # Drop in hand scales
        updated_scale = pose_output['atlas']['scale'].clone()
        updated_scale[:, 9] = lhand_output['atlas']['scale'][:, 9]
        updated_scale[:, 8] = rhand_output['atlas']['scale'][:, 8]
        updated_scale[:, 18:] = (lhand_output['atlas']['scale'][:, 18:] + rhand_output['atlas']['scale'][:, 18:]) / 2
        
        # Update hand shape
        updated_shape = pose_output['atlas']['shape'].clone()
        updated_shape[:, 40:] = (lhand_output['atlas']['shape'][:, 40:] + rhand_output['atlas']['shape'][:, 40:]) / 2
            
        print("Doing IK...")
        # First, forward just FK
        joint_rotations = self.model.head_pose.mohr_forward(
            global_trans=pose_output['atlas']['global_rot'] * 0,
            global_rot=pose_output['atlas']['global_rot'],
            body_pose_params=pose_output['atlas']['body_pose'],
            hand_pose_params=updated_hand_pose,
            scale_params=updated_scale,
            shape_params=updated_shape,
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
        wrist_xzy = fix_wrist_euler(roma.rotmat_to_euler("XZY", fused_local_wrist_rotmat))
        
        # Put it in.
        angle_difference = rotation_angle_difference(ori_local_wrist_rotmat, fused_local_wrist_rotmat) # B x 2 x 3 x3
        valid_angle = angle_difference < self.thresh_wrist_angle
        valid_angle = valid_angle & hand_valid_mask
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
        
        # Replace shared shape and scale
        pose_output['atlas']['scale'][:, 18:] = torch.where(valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0, (
            lhand_output['atlas']['scale'][:, 18:] * valid_angle.squeeze(-1)[:, [0]] + rhand_output['atlas']['scale'][:, 18:] * valid_angle.squeeze(-1)[:, [1]]
        ) / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8), pose_output['atlas']['scale'][:, 18:])
        pose_output['atlas']['shape'][:, 40:] = torch.where(valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0, (
            lhand_output['atlas']['shape'][:, 40:] * valid_angle.squeeze(-1)[:, [0]] + rhand_output['atlas']['shape'][:, 40:] * valid_angle.squeeze(-1)[:, [1]]
        ) / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8), pose_output['atlas']['shape'][:, 40:])
        
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
                    "pred_global_rots": out['joint_global_rots'][idx],
                    "angle_diff": angle_difference[idx].cpu().numpy(),
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

            all_out[-1]["lhand_bbox"] = np.array([
                (batch_lhand['bbox_center'].flatten(0, 1)[idx][0] - batch_lhand['bbox_scale'].flatten(0, 1)[idx][0] / 2).item(),
                (batch_lhand['bbox_center'].flatten(0, 1)[idx][1] - batch_lhand['bbox_scale'].flatten(0, 1)[idx][1] / 2).item(),
                (batch_lhand['bbox_center'].flatten(0, 1)[idx][0] + batch_lhand['bbox_scale'].flatten(0, 1)[idx][0] / 2).item(),
                (batch_lhand['bbox_center'].flatten(0, 1)[idx][1] + batch_lhand['bbox_scale'].flatten(0, 1)[idx][1] / 2).item(),
            ])
            all_out[-1]["rhand_bbox"] = np.array([
                (batch_rhand['bbox_center'].flatten(0, 1)[idx][0] - batch_rhand['bbox_scale'].flatten(0, 1)[idx][0] / 2).item(),
                (batch_rhand['bbox_center'].flatten(0, 1)[idx][1] - batch_rhand['bbox_scale'].flatten(0, 1)[idx][1] / 2).item(),
                (batch_rhand['bbox_center'].flatten(0, 1)[idx][0] + batch_rhand['bbox_scale'].flatten(0, 1)[idx][0] / 2).item(),
                (batch_rhand['bbox_center'].flatten(0, 1)[idx][1] + batch_rhand['bbox_scale'].flatten(0, 1)[idx][1] / 2).item(),
            ])

        return all_out

    def _prompt_wrists(self, batch, output, keypoint_prompt):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]
        pose_output = output["atlas"]  # body-only output
        # Use previous estimate as initialization
        prev_estimate = torch.cat(
            [
                pose_output["pred_pose_raw"].detach(),  # (B, 6)
                pose_output["shape"].detach(),
                pose_output["scale"].detach(),
                pose_output["hand"].detach(),
                pose_output["face"].detach(),
            ],
            dim=1,
        ).unsqueeze(dim=1)
        if hasattr(self.model, "init_camera"):
            prev_estimate = torch.cat(
                [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                dim=-1,
            )
        
        tokens_output, pose_output = self.model.forward_decoder(
            image_embeddings,
            init_estimate=None,  # not recurring previous estimate
            keypoints=keypoint_prompt,
            prev_estimate=prev_estimate,
            condition_info=condition_info,
            batch=batch,
            full_output=None,
        )
        pose_output = pose_output[-1]

        output.update({"atlas": pose_output})
        return output, keypoint_prompt
