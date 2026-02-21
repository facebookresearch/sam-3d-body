#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import hashlib
import json
import os
import shutil
import time
from glob import glob

import cv2
import numpy as np
import pyrootutils
import torch

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tiff")


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob(os.path.join(input_path, ext)))
    return sorted(images)


def resize_keep_aspect(img_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    h, w = img_bgr.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return img_bgr, 1.0, 1.0
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, new_w / float(w), new_h / float(h)


def pick_single_person(outputs: list[dict]) -> list[dict]:
    if len(outputs) <= 1:
        return outputs
    areas = []
    for out in outputs:
        x1, y1, x2, y2 = out["bbox"]
        areas.append(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
    best_idx = int(np.argmax(np.asarray(areas)))
    return [outputs[best_idx]]


def to_npz_payload(
    outputs: list[dict],
    image_path: str,
    orig_w: int,
    orig_h: int,
    proc_w: int,
    proc_h: int,
    scale_x: float,
    scale_y: float,
    map_2d_to_original: bool,
    infer_ms: float,
) -> dict:
    payload = {
        "schema_version": np.asarray("sam3d.fast_npz.v1"),
        "image_path": np.asarray(image_path),
        "num_people": np.asarray(len(outputs), dtype=np.int32),
        "orig_w": np.asarray(orig_w, dtype=np.int32),
        "orig_h": np.asarray(orig_h, dtype=np.int32),
        "proc_w": np.asarray(proc_w, dtype=np.int32),
        "proc_h": np.asarray(proc_h, dtype=np.int32),
        "scale_x": np.asarray(scale_x, dtype=np.float32),
        "scale_y": np.asarray(scale_y, dtype=np.float32),
        "infer_ms": np.asarray(infer_ms, dtype=np.float32),
    }

    for pid, out in enumerate(outputs):
        prefix = f"person_{pid:03d}_"

        bbox = out["bbox"].astype(np.float32)
        k2d = out["pred_keypoints_2d"].astype(np.float32)
        if map_2d_to_original and scale_x > 0 and scale_y > 0:
            bbox = bbox.copy()
            bbox[[0, 2]] /= scale_x
            bbox[[1, 3]] /= scale_y
            k2d = k2d.copy()
            k2d[:, 0] /= scale_x
            k2d[:, 1] /= scale_y

        payload[prefix + "bbox"] = bbox
        payload[prefix + "focal_length"] = np.asarray(out["focal_length"], dtype=np.float32)
        payload[prefix + "pred_cam_t"] = out["pred_cam_t"].astype(np.float32)
        payload[prefix + "pred_keypoints_2d"] = k2d
        payload[prefix + "pred_keypoints_3d"] = out["pred_keypoints_3d"].astype(np.float32)
        payload[prefix + "pred_vertices"] = out["pred_vertices"].astype(np.float32)

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast JPEG -> NPZ inference for SAM 3D Body")
    parser.add_argument("--input", required=True, type=str, help="Input image path or folder")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for NPZ files")
    parser.add_argument(
        "--checkpoint_path",
        default="./checkpoints/sam-3d-body-dinov3/model.ckpt",
        type=str,
        help="Path to SAM 3D Body checkpoint",
    )
    parser.add_argument(
        "--mhr_path",
        default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        type=str,
        help="Path to MHR model asset",
    )
    parser.add_argument(
        "--max_side",
        default=1280,
        type=int,
        help="Downsize so long side <= max_side. Set 0 to disable resize.",
    )
    parser.add_argument(
        "--single_person",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only best single person (default: true)",
    )
    parser.add_argument(
        "--map_2d_to_original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map bbox and 2D keypoints back to original image scale",
    )
    parser.add_argument(
        "--save_compressed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use savez_compressed (smaller files, slower writes)",
    )
    parser.add_argument("--bbox_thr", default=0.8, type=float, help="BBox threshold")
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional cache directory for NPZ outputs keyed by image+settings+model",
    )
    return parser.parse_args()


def _file_sig(path: str) -> dict:
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _cache_key(image_path: str, context: dict) -> str:
    h = hashlib.sha256()
    h.update(_sha256_file(image_path).encode("utf-8"))
    h.update(json.dumps(context, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    images = list_images(args.input)
    if len(images) == 0:
        raise RuntimeError(f"No images found for input: {args.input}")

    cache_enabled = len(args.cache_dir) > 0
    if cache_enabled:
        os.makedirs(args.cache_dir, exist_ok=True)

    cache_context = {
        "schema": "sam3d.fast_npz.v1",
        "checkpoint": _file_sig(args.checkpoint_path),
        "mhr": _file_sig(args.mhr_path),
        "max_side": int(args.max_side),
        "single_person": bool(args.single_person),
        "map_2d_to_original": bool(args.map_2d_to_original),
        "bbox_thr": float(args.bbox_thr),
        "save_compressed": bool(args.save_compressed),
    }

    misses: list[tuple[str, str, str]] = []
    for image_path in images:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_npz = os.path.join(args.output_dir, f"{stem}.npz")
        cache_npz = ""

        if cache_enabled:
            t0 = time.perf_counter()
            key = _cache_key(image_path, cache_context)
            cache_npz = os.path.join(args.cache_dir, f"{key}.npz")
            if os.path.exists(cache_npz):
                shutil.copy2(cache_npz, out_npz)
                cache_ms = (time.perf_counter() - t0) * 1000.0
                print(f"[cache-hit] {os.path.basename(image_path)} -> {out_npz} | cache_ms={cache_ms:.1f}")
                continue

        misses.append((image_path, out_npz, cache_npz))

    if len(misses) == 0:
        print("All inputs served from cache. Model load skipped.")
        return

    device = detect_device()
    print(f"Using device: {device}")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=args.checkpoint_path,
        device=device,
        mhr_path=args.mhr_path,
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    for image_path, out_npz, cache_npz in misses:

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"[skip] unreadable: {image_path}")
            continue

        orig_h, orig_w = img_bgr.shape[:2]
        proc_bgr, scale_x, scale_y = resize_keep_aspect(img_bgr, args.max_side)
        proc_h, proc_w = proc_bgr.shape[:2]

        infer_t0 = time.perf_counter()
        if args.max_side > 0 and (proc_w != orig_w or proc_h != orig_h):
            proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
            outputs = estimator.process_one_image(
                proc_rgb,
                bbox_thr=args.bbox_thr,
                use_mask=False,
            )
        else:
            outputs = estimator.process_one_image(
                image_path,
                bbox_thr=args.bbox_thr,
                use_mask=False,
            )
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0

        if args.single_person:
            outputs = pick_single_person(outputs)

        save_t0 = time.perf_counter()
        payload = to_npz_payload(
            outputs=outputs,
            image_path=image_path,
            orig_w=orig_w,
            orig_h=orig_h,
            proc_w=proc_w,
            proc_h=proc_h,
            scale_x=scale_x,
            scale_y=scale_y,
            map_2d_to_original=args.map_2d_to_original,
            infer_ms=infer_ms,
        )

        if args.save_compressed:
            np.savez_compressed(out_npz, **payload)
        else:
            np.savez(out_npz, **payload)
        save_ms = (time.perf_counter() - save_t0) * 1000.0

        if cache_enabled and len(cache_npz) > 0:
            shutil.copy2(out_npz, cache_npz)

        print(
            f"[ok] {os.path.basename(image_path)} -> {out_npz} | "
            f"people={len(outputs)} infer_ms={infer_ms:.1f} save_ms={save_ms:.1f} "
            f"size={orig_w}x{orig_h}->{proc_w}x{proc_h}"
        )


if __name__ == "__main__":
    main()
