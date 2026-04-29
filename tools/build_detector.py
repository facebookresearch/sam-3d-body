# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class HumanDetector:
    def __init__(self, name="vitdet", device="cuda", offload_mode="none", **kwargs):
        self.device = torch.device(device)
        self.offload_mode = offload_mode

        if name == "vitdet":
            print(f"########### Using human detector: ViTDet... (offload_mode={offload_mode!r})")
            # DetectionCheckpointer always loads to CPU internally;
            # load_detectron2_vitdet moves to CPU explicitly after.
            self.detector = load_detectron2_vitdet(**kwargs)
            self.detector_func = run_detectron2_vitdet

            if offload_mode == "none":
                # Original behaviour: keep on target device persistently
                self.detector = self.detector.to(self.device)
            # offload_mode="cpu": stays on CPU, moved per-call in run_human_detection
            self.detector.eval()
        elif name == "sam3":
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self.detector = build_sam3_image_model(device=str(self.device))
            self.processor = Sam3Processor(self.detector)
            self.detector_func = lambda detector, img, **kwargs: self.sam3_run(
                img, **kwargs
            )
        else:
            raise NotImplementedError
        
    def sam3_run(self, img, det_cat_id: int = 0, bbox_thr: float = 0.5, **kwargs):
        # switch bgr to rgb 
        img = img[:, :, ::-1].copy()
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        inference_state = self.processor.set_image(img)
        # Prompt the model with text
        output = self.processor.set_text_prompt(state=inference_state, prompt="person")

        # Get the masks, bounding boxes, and scores
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        
        confident_idx = scores > bbox_thr
        boxes = boxes[confident_idx].cpu().numpy()
        
        # resize the box with a scale factor 1.2
        scale = 1.2
        enlarged_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = (x2 - x1) * scale
            h = (y2 - y1) * scale
            new_x1 = max(cx - w / 2, 0)
            new_y1 = max(cy - h / 2, 0)
            new_x2 = cx + w / 2
            new_y2 = cy + h / 2
            enlarged_boxes.append([new_x1, new_y1, new_x2, new_y2])
        return np.array(enlarged_boxes)

    def run_human_detection(self, img, **kwargs):
        if self.offload_mode == "cpu" and self.device.type != "cpu":
            self.detector.to(self.device)
            try:
                result = self.detector_func(self.detector, img, **kwargs)
            finally:
                self.detector.cpu()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            return result
        else:
            return self.detector_func(self.detector, img, **kwargs)


def load_detectron2_vitdet(path=""):
    """
    Load vitdet detector similar to 4D-Humans demo.py approach.
    Checkpoint is automatically downloaded from the hardcoded URL.
    """
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import instantiate, LazyConfig

    # Get config file from tools directory (same folder as this file)
    cfg_path = Path(__file__).parent / "cascade_mask_rcnn_vitdet_h_75ep.py"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. "
            "Make sure cascade_mask_rcnn_vitdet_h_75ep.py exists in the tools directory."
        )

    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        if path == ""
        else os.path.join(path, "model_final_f05665.pkl")
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = instantiate(detectron2_cfg.model)
    checkpointer = DetectionCheckpointer(detector)
    checkpointer.load(detectron2_cfg.train.init_checkpoint)
    # DetectionCheckpointer loads to whichever device the model skeleton is on.
    # Explicitly move to CPU so the caller controls final placement.
    detector = detector.cpu()
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
    # Follow the detector's actual device so this works with CPU offloading
    _device = next(detector.parameters()).device
    img_transformed = torch.as_tensor(
        img_transformed.astype("float32").transpose(2, 0, 1)
    ).to(_device)
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
