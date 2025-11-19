# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import webdataset as wds
from typing import List, Any, Dict
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import warnings
import argparse


def get_anno_key(row: Dict[str, Any]):
    dataset = row["dataset"]
    img_name = row["image"]
    if dataset == "egoexo4d" or dataset == "harmony4d":
        return "-".join([dataset, img_name[:-4].replace("/", "-")])
    elif dataset == "egohumans":
        seq, subseq, cam, img_idx = img_name[:-4].split("/")
        person_idx = int(row["person_id"])
        person = f"aira{person_idx+1:02}"
        return "-".join([dataset, seq, subseq, person, cam, img_idx])
    elif dataset == "3dpw":
        seq, img_idx = img_name[:-4].split("/")
        return f"{dataset}-{seq}-{img_idx}"
    elif (
        dataset == "aic" or dataset == "coco" or dataset == "sa1b" or dataset == "mpii"
    ):
        return f"{dataset}-{img_name[:-4]}"
    else:
        raise NotImplementedError


def get_img_name(row: Dict[str, Any]):
    dataset = row["dataset"]
    img_name = row["image"]
    if dataset == "coco":
        _, split, _ = img_name.split("_")
        return os.path.join(split, img_name)
    elif dataset == "mpii":
        return os.path.join("images", img_name)
    elif dataset == "aic":
        return os.path.join("train", "images", img_name)
    else:
        return img_name


def create_webdatset_shard(
    img_dir: str,
    ann_dir: str,
    wds_dir: str,
    filename: str,
):
    shard = filename.replace("parquet", "tar")
    ann_file: str = os.path.join(ann_dir, filename)
    res_file: str = os.path.join(wds_dir, shard)
    if os.path.exists(ann_file):
        hf_data = pd.read_parquet(ann_file)
    else:
        warnings.warn(f"Annotation file not found: {ann_file}", UserWarning)
        return

    img_annos = []
    for idx, row in tqdm(hf_data.iterrows()):
        subject_idx = row["subject_idx"]
        subject_cnt = row["subject_cnt"]
        if subject_idx == 0:
            img_anno: Dict[str, Any] = {}
            img_anno["__key__"] = get_anno_key(row)
            img_name = get_img_name(row)
            img = cv2.imread(os.path.join(img_dir, img_name))
            img_anno["jpg"] = cv2.imencode(".jpg", img)[1].tobytes()
            img_anno["metadata.json"] = {"width": img.shape[1], "height": img.shape[0]}
            img_anno["annotation.pyd"] = []

        anno = {}
        anno["person_id"] = row["person_id"]
        anno["keypoints_2d"] = np.stack(row["keypoints_2d"])
        anno["keypoints_3d"] = np.stack(row["keypoints_3d"])

        proto_params: Dict[str, np.ndarray] = {}
        proto_params["global_rot"] = row["global_rot"]
        proto_params["body_pose_params"] = row["body_pose_params"]
        proto_params["hand_pose_params"] = row["hand_pose_params"]
        proto_params["scale_params"] = row["scale_params"]
        proto_params["shape_params"] = row["shape_params"]
        proto_params["expr_params"] = row["expr_params"]
        anno["proto_params"] = proto_params

        anno["proto_valid"] = row["proto_valid"]
        anno["proto_version"] = row["date"]

        anno["bbox"] = row["bbox"]
        anno["bbox_format"] = row["bbox_format"]
        anno["bbox_score"] = row.get("bbox_score", 1.0)
        anno["center"] = row["bbox_center"]
        anno["scale"] = np.array(row["bbox_scale"])

        metadata: Dict[str, Any] = {}
        metadata["cam_trans"] = row["global_trans"]
        metadata["cam_int"] = np.stack(row["cam_int"])
        metadata["loss"] = row.get("loss", 0.0)
        anno["metadata"] = metadata

        mask_size = row["mask_size"]
        mask_cnts = row["mask_cnts"]
        if mask_size.tolist() != [-1, -1] and mask_cnts != b"":
            assert mask_size[0] == img.shape[0] and mask_size[1] == img.shape[1]
            anno["mask"] = {"size": mask_size, "counts": mask_cnts}
        img_anno["annotation.pyd"].append(anno)

        if subject_idx == subject_cnt - 1:
            img_annos.append(img_anno)

    with wds.TarWriter(res_file) as sink:
        for img_anno in tqdm(img_annos):
            sink.write(img_anno)


def create_webdataset_shard_multiprocess(
    img_dir: str,
    ann_dir: str,
    wds_dir: str,
    files: List[str],
    max_workers: int,
) -> None:
    """Parallel download without returning results."""
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                create_webdatset_shard,
                img_dir,
                ann_dir,
                wds_dir,
                file,
            )
            for file in files
        ]

        for fut in as_completed(futures):
            # We don’t store results — just catch any exceptions for logging.
            try:
                fut.result()
            except Exception as e:
                print(f"✗ Exception during webdataset: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a webdataset for SAM-3D-Body-Data split."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Folder path for the dataset images.",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        help="Folder path for the dataset annotations.",
    )
    parser.add_argument(
        "--webdataset_dir",
        type=str,
        help="Folder path save to the webdatset.",
    )
    parser.add_argument(
        "--max_workers",
        default=16,
        type=int,
        help="Maximum number of processes.",
    )
    parser.add_argument(
        "--shard_idxs",
        default="",
        type=str,
        help="Indices of the shards to be created.",
    )

    args = parser.parse_args()
    img_dir: str = args.image_dir
    ann_dir: str = args.annotation_dir
    wds_dir: str = args.webdataset_dir
    max_workers: int = args.max_workers
    shard_idxs: str = args.shard_idxs

    if shard_idxs != "":
        shard_idxs = sorted([int(shard_idx) for shard_idx in shard_idxs.split(",")])
        files: List[str] = [f"{shard_idx:06}.parquet" for shard_idx in shard_idxs]
    else:
        pattern = re.compile(r"^\d{6}\.parquet$")
        files = sorted([file for file in os.listdir(ann_dir) if pattern.match(file)])

    if not os.path.exists(wds_dir):
        os.makedirs(wds_dir)
    create_webdataset_shard_multiprocess(img_dir, ann_dir, wds_dir, files, max_workers)


if __name__ == "__main__":
    main()
