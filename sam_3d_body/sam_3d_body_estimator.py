import copy
from typing import Optional, Union

import numpy as np
import torch
import cv2

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


class SAM3DBodyEstimator:
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

            data_list.append(self.transform(data_info))

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
            batch[key] = batch[key].unsqueeze(0).float()
        batch["mask"] = batch["mask"].unsqueeze(2)
        batch["person_valid"] = torch.ones((1, max_num_person))

        # Set default camera intrinsics
        batch["cam_int"] = torch.tensor(
            [[[(height ** 2 + width ** 2) ** 0.5, 0, width / 2.],
            [0, (height ** 2 + width ** 2) ** 0.5, height / 2.],
            [0, 0, 1]]],
        ).to(batch["img"])
        batch['img_ori'] = [NoCollate(img)]

        #################### Run model inference on an image ####################

        if self.cfg.MODEL.NAME == "promptable_threepo_triplet":
            batch['lhand_img'] = torch.zeros_like(batch['img'][:, :, :, :self.cfg.MODEL.HAND_IMAGE_SIZE[0], :self.cfg.MODEL.HAND_IMAGE_SIZE[1]])
            batch['rhand_img'] = torch.zeros_like(batch['img'][:, :, :, :self.cfg.MODEL.HAND_IMAGE_SIZE[0], :self.cfg.MODEL.HAND_IMAGE_SIZE[1]])

        batch = recursive_to(batch, "cuda")
        if self.cfg.MODEL.NAME == "promptable_threepo_triplet":
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

            pose_output_ab, _ = self.model.forward_step(batch)
            updated_batch = self.model.get_hand_box(pose_output_ab, batch)
            pose_output, _ = self.model.forward_step(updated_batch, return_feature_only=False)


        # Cache information for future prompting
        self.batch = copy.deepcopy(batch)
        self.pose_output = copy.deepcopy(pose_output)

        out = pose_output["atlas"]
        out = recursive_to(out, "cpu")
        out = recursive_to(out, "numpy")
        all_out = []
        for idx in range(max_num_person):
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
