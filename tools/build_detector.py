# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import torch
import numpy as np


class HumanDetector:
    def __init__(self, name="vitdet", device="cuda", **kwargs):
        self.device = device

        if name == "vitdet":
            print("########### Using human detector: ViTDet...")
            self.detector = load_detectron2_vitdet(**kwargs)
            self.detector_func = run_detectron2_vitdet

            self.detector = self.detector.to(self.device)
            self.detector.eval()
        else:
            raise NotImplementedError
    
    def run_human_detection(self, img, **kwargs):
        return self.detector_func(self.detector, img, **kwargs)
        

def load_detectron2_vitdet(path):
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate, LazyConfig

    DETECTRON_CFG = os.path.join(path, "cascade_mask_rcnn_vitdet_h_75ep.py")
    DETECTRON_CKPT = os.path.join(path, "model_final_f05665.pkl")

    detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = instantiate(detectron2_cfg.model)
    checkpointer = DetectionCheckpointer(detector)
    checkpointer.load(DETECTRON_CKPT)

    detector.eval()
    return detector


def run_detectron2_vitdet(
    detector,
    img,
    det_cat_id: int = 0,
    bbox_thr: float = 0.5,
    nms_thr: float = 0.3,
    default_to_full_image: bool = True,
):
    import detectron2.data.transforms as T

    height, width = img.shape[:2]

    IMAGE_SIZE = 1024
    transforms = T.ResizeShortestEdge(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE)
    img_transformed = transforms(T.AugInput(img)).apply_image(img)
    img_transformed = torch.as_tensor(
        img_transformed.astype("float32").transpose(2, 0, 1)
    )
    inputs = {"image": img_transformed, "height": height, "width": width}

    with torch.no_grad():
        det_out = detector([inputs])

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
        