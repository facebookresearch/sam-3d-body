#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam_3d_body.reference_assets import (
    build_reference_assets,
    discover_reference_videos,
    load_reference_manifest,
)
from sam_3d_body.video_processor import VideoExtractionConfig


_DEFAULT_CHECKPOINT_DIR = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "sam-3d-body-dinov3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch preprocess reference videos into skeleton npz files and metadata.json."
        )
    )
    parser.add_argument("--input-dir", default="", help="Directory containing reference videos.")
    parser.add_argument("--manifest-json", default="", help="Manifest JSON path for per-video metadata.")
    parser.add_argument("--output-dir", required=True, help="Output directory for npz and metadata.")

    parser.add_argument("--action-type", default="", help="Default action type for all videos.")
    parser.add_argument("--athlete-name", default="", help="Default athlete name.")
    parser.add_argument("--camera-view", default="", help="Default camera view.")
    parser.add_argument("--handedness", default="", help="Default handedness.")
    parser.add_argument("--skeleton-version", default="sam3db_v1", help="Output skeleton version tag.")

    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--start-time-sec", type=float, default=0.0)
    parser.add_argument("--end-time-sec", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--bbox-thr", type=float, default=0.5)
    parser.add_argument("--use-mask", action="store_true")
    parser.add_argument("--inference-type", default="body")

    parser.add_argument(
        "--checkpoint-path",
        default=str(_DEFAULT_CHECKPOINT_DIR / "model.ckpt"),
        help="SAM-3D-Body checkpoint path.",
    )
    parser.add_argument(
        "--mhr-path",
        default=str(_DEFAULT_CHECKPOINT_DIR / "assets" / "mhr_model.pt"),
        help="MHR model path.",
    )
    parser.add_argument("--device", default="cuda", help="Inference device.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    return parser.parse_args()


def _normalize_optional_str(value: str) -> str | None:
    trimmed = value.strip()
    return trimmed or None


def _build_default_video_config(args: argparse.Namespace) -> VideoExtractionConfig:
    return VideoExtractionConfig(
        target_fps=args.target_fps,
        start_time_sec=args.start_time_sec,
        end_time_sec=args.end_time_sec,
        max_frames=args.max_frames,
        bbox_thr=args.bbox_thr,
        use_mask=args.use_mask,
        inference_type=args.inference_type,
    )


def _load_entries(args: argparse.Namespace):
    input_dir = args.input_dir.strip()
    manifest_json = args.manifest_json.strip()
    if bool(input_dir) == bool(manifest_json):
        raise ValueError("Provide exactly one of --input-dir or --manifest-json.")

    default_video_config = _build_default_video_config(args)
    default_action_type = _normalize_optional_str(args.action_type)
    default_athlete_name = _normalize_optional_str(args.athlete_name)
    default_camera_view = _normalize_optional_str(args.camera_view)
    default_handedness = _normalize_optional_str(args.handedness)

    if manifest_json:
        return load_reference_manifest(
            manifest_json,
            default_action_type=default_action_type,
            default_athlete_name=default_athlete_name,
            default_camera_view=default_camera_view,
            default_handedness=default_handedness,
            default_video_config=default_video_config,
        )

    if default_action_type is None:
        raise ValueError("--action-type is required when using --input-dir.")

    return discover_reference_videos(
        input_dir,
        action_type=default_action_type,
        athlete_name=default_athlete_name,
        camera_view=default_camera_view,
        handedness=default_handedness,
        video_config=default_video_config,
    )


def _load_estimator(args: argparse.Namespace) -> SAM3DBodyEstimator:
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        mhr_path=args.mhr_path,
    )
    return SAM3DBodyEstimator(model, model_cfg)


def main() -> None:
    args = parse_args()
    entries = _load_entries(args)
    if not entries:
        raise ValueError("No videos found for preprocessing.")

    estimator = _load_estimator(args)
    metadata = build_reference_assets(
        entries,
        estimator=estimator,
        output_dir=args.output_dir,
        skeleton_version=args.skeleton_version,
        overwrite=args.overwrite,
    )
    print(json.dumps({"assetCount": metadata["assetCount"], "outputDir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
