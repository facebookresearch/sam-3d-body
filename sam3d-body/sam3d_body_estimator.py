import copy
import os
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate, LazyConfig
import detectron2.data.transforms as T

from core.data.utils.io import load_image, resize_image
from core.models.meta_arch.sam3d_body import load_sam3d_body

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


class SAM3DBodyEstimator:
    def __init__(
        self,
        checkpoint_path: str = "",
        proto_path: str = "",
        bbox_threshold: int = 0.8,
        detector_path: str = "",
        use_mask: bool = False,
    ):
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.use_mask = use_mask

        # Initialize human detector
        self.use_detector = False
        if len(detector_path):
            self.detector = self.init_detector(detector_path, bbox_threshold)
            self.detector = self.detector.to(self.device)
            self.detector.eval()
            self.use_detector = True

        # Initialize SAM2 if needed
        if self.use_mask:
            self.sam = self.init_sam()
        
        # Build SAM3D-Body model
        self.model, self.cfg = load_sam3d_body(checkpoint_path, proto_path)
        self.faces = self.model.head_pose.atlas.faces.numpy()
        self.model = self.model.to(self.device)
        self.model.eval()
    
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
        height, width = img.shape[:2]

        if bboxes is not None:
            boxes = bboxes.reshape(-1, 4)
            self.is_crop = True
        elif self.use_detector:
            # boxes = np.array([0, 0, width, height]).reshape(1, 4)
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

        transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )
        # construct batch data samples
        data_list = []
        for idx in range(boxes.shape[0]):
            data_info = dict(img=img)
            data_info["bbox"] = boxes[idx]  # shape (4,)
            data_info["bbox_format"] = "xyxy"

            data_list.append(transform(data_info))

        batch = default_collate(data_list)

        ############# Convert to full-image format #################
        # full_size = 256
        # _, img_full, center_full, scale_full = resize_image(
        #     img,
        #     full_size,
        #     batch["bbox_center"],
        #     batch["bbox_scale"],
        # )
        # bbox_min = center_full - scale_full * 0.5  # pyre-ignore
        # bbox_max = center_full + scale_full * 0.5  # pyre-ignore
        # bbox_full = np.concatenate([bbox_min, bbox_max], axis=1)
        # bbox_full = np.clip(bbox_full, 0, full_size)

        # batch_full = dict(
        #     img_full=to_tensor(img_full),
        #     full_bbox_center=center_full,
        #     full_bbox_scale=scale_full,
        #     full_bbox=bbox_full,
        # )
        # batch_full = default_collate([batch_full])
        # batch.update(batch_full)

        max_num_person = batch["img"].shape[0]
        for key in [
            "img",
            "img_size",
            "ori_img_size",
            "bbox_center",
            "bbox_scale",
            "bbox",
            "affine_trans",
        ]:
            batch[key] = batch[key].unsqueeze(0).float()
        batch["person_valid"] = torch.ones((1, max_num_person))

        # FIXME: Use default camera intrinsics for now
        batch["cam_int"] = torch.tensor(
            [[[(height ** 2 + width ** 2) ** 0.5, 0, width / 2.],
            [0, (height ** 2 + width ** 2) ** 0.5, height / 2.],
            [0, 0, 1]]],
        ).to(batch["img"])

        #############################################################

        batch = recursive_to(batch, "cuda")
        self.model._initialize_batch(batch)
        with torch.no_grad():
            pose_output, full_output = self.model.forward_step(batch)

        # Cache information for future prompting
        self.batch = copy.deepcopy(batch)
        self.pose_output = copy.deepcopy(pose_output)
        self.full_output = copy.deepcopy(full_output)

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
                }
            )

        return all_out
