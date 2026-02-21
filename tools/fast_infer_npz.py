#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from sam_3d_body import run_fast_infer


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
        "--inference_type",
        default="full",
        type=str,
        choices=["full", "body"],
        help="'full' = body+hand decoders (default). 'body' = body only (faster, no hand detail).",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional cache directory for NPZ outputs keyed by image+settings+model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fast_infer(
        input_path=args.input,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        mhr_path=args.mhr_path,
        max_side=args.max_side,
        single_person=args.single_person,
        map_2d_to_original=args.map_2d_to_original,
        save_compressed=args.save_compressed,
        cache_dir=args.cache_dir,
        bbox_thr=args.bbox_thr,
        inference_type=args.inference_type,
    )


if __name__ == "__main__":
    main()
