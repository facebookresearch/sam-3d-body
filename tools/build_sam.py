# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import numpy as np
from PIL import Image


class HumanSegmentor:
    def __init__(self, name="sam2", device="cuda", offload_mode="none", **kwargs):
        self.device = torch.device(device)
        self.offload_mode = offload_mode

        if name == "sam2":
            print(f"########### Using human segmentor: SAM2... (offload_mode={offload_mode!r})")
            if offload_mode == "cpu":
                # Load to CPU; move to device only during run_sam
                self.sam = load_sam2("cpu", **kwargs)
            else:
                self.sam = load_sam2(device, **kwargs)
            self.sam_func = run_sam2
        elif name == "sam3":
            print(f"########### Using human segmentor: SAM3... (offload_mode={offload_mode!r})")
            self.sam = load_sam3(device, **kwargs)
            self.sam_func = run_sam3
        else:
            raise NotImplementedError

    def run_sam(self, img, boxes, **kwargs):
        if self.offload_mode == "cpu" and self.device.type != "cpu":
            self.sam.model.to(self.device)
            try:
                result = self.sam_func(self.sam, img, boxes)
            finally:
                self.sam.model.cpu()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            return result
        else:
            return self.sam_func(self.sam, img, boxes)
        

def load_sam2(device, path):
    checkpoint = f"{path}/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    import sys
    sys.path.append(path)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
    predictor.model.eval()

    return predictor


def load_sam3(device, path):
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(device=device)
    predictor = Sam3Processor(model)
    return predictor


def run_sam2(sam_predictor, img, boxes):
    _device = next(sam_predictor.model.parameters()).device
    _dtype = torch.bfloat16 if _device.type == "cuda" else torch.float32
    with torch.autocast(_device.type, dtype=_dtype):
        sam_predictor.set_image(img)
        all_masks, all_scores = [], []
        for i in range(boxes.shape[0]):
            # First prediction: bbox only
            masks, scores, logits = sam_predictor.predict(
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


def run_sam3(sam_predictor, img, boxes):
    # switch bgr to rgb 
    img = img[:, :, ::-1].copy()
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    inference_state = sam_predictor.set_image(img)
    # Prompt the model with text
    output = sam_predictor.set_text_prompt(state=inference_state, prompt="person")

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    score_threshold = 0.5
    confident_idx = scores > score_threshold
    masks = masks[confident_idx].float().squeeze(1).cpu().numpy()
    scores = scores[confident_idx].cpu().numpy()

    return masks, scores