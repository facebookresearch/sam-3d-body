import argparse
import os
import random
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from sam3d_body_estimator import SAM3DBodyEstimator

from tqdm import tqdm


import numpy as np
import cv2
from core.visualization.renderer import Renderer
from core.visualization.skeleton_visualizer import SkeletonVisualizer
from core.metadata.atlas70 import pose_info as atlas70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(atlas70_pose_info)

def visualize_sample(img_cv2, outputs, faces):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_mesh.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img3 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img


DETECTOR_FOLDER = "/large_experiments/3po/model/cascade_mask_rcnn_vitdet"
# PROTO_PATH = "/large_experiments/3po/model/atlas_250825"
PROTO_PATH = "/large_experiments/3po/model/atlas_250926_dev2/assets"
MOGE_PATH = "/private/home/jiawliu/cores/Human-Object/postprocess/MoGe/checkpoints/model.pt"

def main(args):
    if args.output_folder == "":
        output_folder = os.path.join(
            "./output", os.path.basename(args.image_folder)
        )
    else:
        output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    model = SAM3DBodyEstimator(
        checkpoint_path=args.checkpoint_path,
        proto_path=PROTO_PATH,
        detector_path=DETECTOR_FOLDER,
        moge_path=MOGE_PATH,
    )

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    images_list = [image for ext in image_extensions for image in glob(os.path.join(args.image_folder, ext))]

    for image_path in tqdm(images_list):
        outputs = model.process_one_image(image_path)

        # TODO [Devansh?]: change visualization stuffs to the demo-version Devansh/Soyong are using
        img = cv2.imread(image_path)
        rend_img = visualize_sample(img, outputs, model.faces)
        for i, img in enumerate(rend_img):
            cv2.imwrite(f"{output_folder}/{os.path.basename(image_path)[:-4]}_{i}.jpg", img.astype(np.uint8))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ThreePO Mesh Regressor')
    parser.add_argument("--image_folder", default="", type=str)
    parser.add_argument("--output_folder", default="", type=str)
    parser.add_argument("--checkpoint_path", default="", type=str)
    parser.add_argument("--bbox_thresh", default=0.8, type=float)
    parser.add_argument("--use_mask", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
