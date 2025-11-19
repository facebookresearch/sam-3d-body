# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import numpy as np


class HumanSegmentor:
    def __init__(self, name="sam2", device="cuda", **kwargs):
        self.device = device

        if name == "sam2":
            print("########### Using human segmentor: SAM2...")
            self.sam = load_sam2(device, **kwargs)
            self.sam_func = run_sam2

        else:
            raise NotImplementedError
    
    def run_sam(self, img, boxes, **kwargs):
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


def run_sam2(sam_predictor, img, boxes):
    with torch.autocast("cuda", dtype=torch.bfloat16):
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
        