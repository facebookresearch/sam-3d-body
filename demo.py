import argparse
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from tqdm import tqdm
import cv2
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator, SAM3DBodyEstimatorTTA, SAM3DBodyEstimatorUnified
from tools.vis_utils import visualize_sample


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join(
            "./output", os.path.basename(args.image_folder)
        )
    else:
        output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mohr_path = args.mohr_path or os.environ.get("SAM3D_MOHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model, model_cfg = load_sam_3d_body(args.checkpoint_path, mohr_path)
    model = model.to(device)

    human_detector, human_segmentor, fov_estimator = None, None, None
    if len(detector_path):
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    if len(segmentor_path):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if len(fov_path):
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(
            name=args.fov_name, device=device, path=fov_path
        )

    estimator = SAM3DBodyEstimatorUnified(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
        prompt_wrists=args.prompt_wrists,
    )

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    images_list = sorted([image for ext in image_extensions for image in glob(os.path.join(args.image_folder, ext))])

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(image_path, use_mask=args.use_mask)

        img = cv2.imread(image_path)
        rend_img = visualize_sample(img, outputs, estimator.faces)
        for i, img in enumerate(rend_img):
            cv2.imwrite(f"{output_folder}/{os.path.basename(image_path)[:-4]}_{i}.jpg", img.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SAM 3D Body Demo - Single Image Human Mesh Recovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

Environment Variables:
  SAM3D_MOHR_PATH: Path to MoHR/assets folder
  SAM3D_DETECTOR_PATH: Path to human detection model folder
  SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
  SAM3D_FOV_PATH: Path to fov estimation model folder
        """
    )
    parser.add_argument("--image_folder", required=True, type=str,
                        help="Path to folder containing input images")
    parser.add_argument("--output_folder", default="", type=str,
                        help="Path to output folder (default: ./output/<image_folder_name>)")
    parser.add_argument("--checkpoint_path", required=True, type=str,
                        help="Path to SAM 3D Body model checkpoint")
    parser.add_argument("--detector_name", default="vitdet", type=str,
                        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).")
    parser.add_argument("--segmentor_name", default="sam2", type=str,
                        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).")
    parser.add_argument("--fov_name", default="moge2", type=str,
                        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).")
    parser.add_argument("--detector_path", default="", type=str,
                        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)")
    parser.add_argument("--segmentor_path", default="", type=str,
                        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)")
    parser.add_argument("--fov_path", default="", type=str,
                        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)")
    parser.add_argument("--mohr_path", default="", type=str,
                        help="Path to MoHR/assets folder (or set SAM3D_MOHR_PATH)")
    parser.add_argument("--bbox_thresh", default=0.8, type=float,
                        help="Bounding box detection threshold")
    parser.add_argument("--use_mask", action="store_true", default=False,
                        help="Use mask-conditioned prediction")
    parser.add_argument("--use_tta", action="store_true", default=False)
    parser.add_argument("--prompt_wrists", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
