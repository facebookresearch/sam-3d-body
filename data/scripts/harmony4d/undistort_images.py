# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
from pathlib import Path

import _init_paths

import cv2
import numpy as np

import os

import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from config import cfg
from lib.datasets.ego_exo_scene import EgoExoSceneVis
import warnings


import argparse


def save_frames(cam_name: str, src_dir: str, dst_dir: str, scale: float):
    scene = EgoExoSceneVis(cfg=cfg, root_dir=src_dir)
    intrinsics = scene.exo_cameras[cam_name].intrinsics.calibration_matrix()
    distortion = scene.exo_cameras[cam_name].intrinsics.params[4:]
    src_cam_dir = os.path.join(src_dir, "exo", cam_name)
    dst_cam_dir = os.path.join(dst_dir, cam_name)
    src_cam_img_dir = os.path.join(src_cam_dir, "images")
    dst_cam_img_dir = dst_cam_dir

    if not os.path.exists(dst_cam_img_dir):
        try:
            os.makedirs(dst_cam_img_dir)
        except:
            pass

    img_files = sorted(
        [
            img_file
            for img_file in os.listdir(src_cam_img_dir)
            if img_file.endswith(".jpg") and img_file.startswith("0")
        ]
    )
    for _, frame_file in enumerate(tqdm(img_files), start=1):
        frame_idx = int(frame_file[:5])
        src_img_file = os.path.join(src_cam_img_dir, frame_file)
        src_img = cv2.imread(src_img_file)
        if frame_idx == 1:
            map1, map2, new_K = undistort_exocam_info(
                src_img, intrinsics, distortion, scale
            )
        dst_image = cv2.remap(
            src_img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        cv2.imwrite(os.path.join(dst_cam_img_dir, frame_file), dst_image)
    return


# Loads dataframe at target path to csv
def load_csv_to_df(filepath: str) -> pd.DataFrame:
    with open(filepath, "r") as csv_file:
        return pd.read_csv(csv_file)


def undistort_exocam_info(image, intrinsics, distortion_coeffs, scale=1):
    dim2 = None
    dim3 = None
    balance = 0.8
    # Load the distortion parameters
    distortion_coeffs = distortion_coeffs
    # Load the camera intrinsic parameters
    intrinsics = intrinsics

    DIM = image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    dim1 = (int(DIM[0] * scale), int(DIM[1] * scale))

    assert (
        dim1[0] / dim1[1] == DIM[0] / DIM[1]
    ), "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = intrinsics  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intrinsics, distortion_coeffs, DIM, np.eye(3), new_size=dim2, balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, distortion_coeffs, np.eye(3), new_K, dim3, cv2.CV_16SC2
    )

    return map1, map2, new_K


def undistort_exocam(image, intrinsics, distortion_coeffs, scale=1):
    dim2 = None
    dim3 = None
    balance = 0.8
    # Load the distortion parameters
    distortion_coeffs = distortion_coeffs
    # Load the camera intrinsic parameters
    intrinsics = intrinsics

    DIM = image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    dim1 = (int(DIM[0] * scale), int(DIM[1] * scale))

    assert (
        dim1[0] / dim1[1] == DIM[0] / DIM[1]
    ), "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = intrinsics  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intrinsics, distortion_coeffs, DIM, np.eye(3), new_size=dim2, balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, distortion_coeffs, np.eye(3), new_K, dim3, cv2.CV_16SC2
    )
    undistorted_image = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return undistorted_image, new_K


def undistort_image(image, intrinsics, distortion_coeffs):
    image = np.array(image)
    return undistort_exocam(
        image, intrinsics, distortion_coeffs, dimension=(image.shape[1], image.shape[0])
    )


def extract_images(seq: str, input_dir: str, output_dir: str, scale: float):
    print(f"Extract images for {seq}.")

    src_dir = os.path.join(input_dir, seq)
    img_dir = os.path.join(src_dir, "exo")
    if not os.path.exists(img_dir):
        warnings.warn(f"{img_dir} does not exist.", category=UserWarning)
        return

    dst_dir = os.path.join(output_dir, seq)
    exo_cams = sorted(os.listdir(img_dir))

    if not os.path.exists(dst_dir):
        try:
            os.makedirs(dst_dir)
        except:
            pass

    with ProcessPoolExecutor(max_workers=10) as ex:
        futures = [
            ex.submit(save_frames, exo_cam, src_dir, dst_dir, scale)
            for exo_cam in exo_cams
        ]

        for fut in as_completed(futures):
            # We don’t store results — just catch any exceptions for logging.
            try:
                fut.result()
            except Exception as e:
                print(f"✗ Exception during image undistortion: {e}")


def main():
    parser = argparse.ArgumentParser(description="Undistorted Harmony4D Images")
    parser.add_argument("--seqs", type=str, help="sequences", default="")
    parser.add_argument(
        "--src_dir", type=str, help="harmony4d dataset path", required=True
    )
    parser.add_argument(
        "--dst_dir", type=str, help="undistorted image path", required=True
    )
    parser.add_argument("--scale", type=float, help="scale", default=1)
    args = parser.parse_args()

    seqs = args.seqs
    src_dir: str = args.src_dir
    dst_dir: str = args.dst_dir
    scale: float = args.scale

    if seqs == "":
        seqs: List[str] = []
        for split in ["train", "test"]:
            for seq in sorted(os.listdir(os.path.join(src_dir, split))):
                for subseq in sorted(os.listdir(os.path.join(src_dir, split, seq))):
                    seqs.append(os.path.join(split, seq, subseq))
    else:
        seqs: List[str] = seqs.split(",")

    for seq in seqs:
        extract_images(seq=seq, input_dir=src_dir, output_dir=dst_dir, scale=scale)


if __name__ == "__main__":
    main()
