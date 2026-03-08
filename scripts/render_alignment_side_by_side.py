#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sam_3d_body.technique_alignment import (
    AlignmentConfig,
    build_alignment_report,
    load_skeleton_sequence_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align two skeleton asset sequences with DTW and render a side-by-side "
            "comparison video from the matched source frames."
        )
    )
    parser.add_argument("--user-video", required=True, help="Path to the user video file.")
    parser.add_argument(
        "--reference-video",
        required=True,
        help="Path to the reference video file.",
    )
    parser.add_argument(
        "--user-skeleton-npz",
        required=True,
        help="Path to the user skeleton npz file.",
    )
    parser.add_argument(
        "--reference-skeleton-npz",
        required=True,
        help="Path to the reference skeleton npz file.",
    )
    parser.add_argument(
        "--user-render-npz",
        required=True,
        help="Path to the user render npz file.",
    )
    parser.add_argument(
        "--reference-render-npz",
        required=True,
        help="Path to the reference render npz file.",
    )
    parser.add_argument(
        "--output-video",
        required=True,
        help="Path to the output side-by-side mp4 file.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the alignment report JSON.",
    )
    parser.add_argument(
        "--user-label",
        default="User",
        help="Label shown on the left video panel.",
    )
    parser.add_argument(
        "--reference-label",
        default="Reference",
        help="Label shown on the right video panel.",
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=720,
        help="Output panel height for each video.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=30.0,
        help="FPS for the output side-by-side video.",
    )
    parser.add_argument(
        "--dtw-radius",
        type=int,
        default=5,
        help="fastdtw radius (ignored when exact_dtw fallback is used).",
    )
    return parser.parse_args()


def _load_render_frame_indices(render_npz_path: str | Path) -> np.ndarray:
    with np.load(render_npz_path, allow_pickle=True) as data:
        if "frame_indices" not in data:
            raise ValueError(f"Missing frame_indices in render npz: {render_npz_path}")
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32)
        if frame_indices.ndim != 1 or frame_indices.size == 0:
            raise ValueError(f"Invalid frame_indices in render npz: {render_npz_path}")
        return frame_indices


def _resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid frame shape")
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _ensure_even_dimensions(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_bottom = height % 2
    pad_right = width % 2
    if pad_bottom == 0 and pad_right == 0:
        return frame
    return cv2.copyMakeBorder(
        frame,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def _load_resized_frames(
    video_path: str | Path,
    source_frame_indices: list[int],
    panel_height: int,
) -> dict[int, np.ndarray]:
    requested = sorted(set(int(index) for index in source_frame_indices))
    if not requested:
        raise ValueError(f"No frame indices requested for {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    frames: dict[int, np.ndarray] = {}
    requested_set = set(requested)
    current_index = 0
    last_requested = requested[-1]

    try:
        while current_index <= last_requested:
            ok, frame = capture.read()
            if not ok:
                break
            if current_index in requested_set:
                resized = _resize_to_height(frame, panel_height)
                frames[current_index] = _ensure_even_dimensions(resized)
            current_index += 1
    finally:
        capture.release()

    missing = [index for index in requested if index not in frames]
    if missing:
        raise ValueError(
            f"Video ended before requested frames were loaded: {video_path}; missing={missing[:10]}"
        )
    return frames


def _overlay_text(
    frame: np.ndarray,
    *,
    title: str,
    source_frame_index: int,
    sequence_frame_index: int,
    timestamp_sec: float,
    distance: float,
) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 86), (0, 0, 0), thickness=-1)

    title_text = title.strip() or "Video"
    meta_text = (
        f"src={source_frame_index}  seq={sequence_frame_index}  "
        f"t={timestamp_sec:.3f}s  d={distance:.4f}"
    )

    cv2.putText(
        output,
        title_text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        meta_text,
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 255, 200),
        2,
        cv2.LINE_AA,
    )
    return output


def _compose_alignment_frame(
    *,
    step_index: int,
    alignment_item: dict[str, Any],
    user_frame: np.ndarray,
    reference_frame: np.ndarray,
    user_label: str,
    reference_label: str,
    user_source_frame_index: int,
    reference_source_frame_index: int,
) -> np.ndarray:
    distance = float(alignment_item["distance"])
    user_annotated = _overlay_text(
        user_frame,
        title=f"{user_label}  step={step_index}",
        source_frame_index=user_source_frame_index,
        sequence_frame_index=int(alignment_item["userFrameIndex"]),
        timestamp_sec=float(alignment_item["userTimestamp"]),
        distance=distance,
    )
    reference_annotated = _overlay_text(
        reference_frame,
        title=f"{reference_label}  step={step_index}",
        source_frame_index=reference_source_frame_index,
        sequence_frame_index=int(alignment_item["referenceFrameIndex"]),
        timestamp_sec=float(alignment_item["referenceTimestamp"]),
        distance=distance,
    )
    composed = np.concatenate([user_annotated, reference_annotated], axis=1)
    return _ensure_even_dimensions(composed)


def _build_video_summary(
    alignment_report: dict[str, Any],
    *,
    output_video: Path,
    output_json: Path | None,
) -> dict[str, Any]:
    summary = dict(alignment_report.get("summary") or {})
    alignment_path = alignment_report.get("alignmentPath") or []
    user_frame_indices = [int(item["userFrameIndex"]) for item in alignment_path]
    reference_frame_indices = [int(item["referenceFrameIndex"]) for item in alignment_path]
    summary.update(
        {
            "algorithm": alignment_report.get("algorithm"),
            "distance": alignment_report.get("distance"),
            "outputVideo": str(output_video.resolve()),
            "outputJson": str(output_json.resolve()) if output_json is not None else None,
            "numAlignedPairs": len(alignment_path),
            "numUniqueUserSequenceFrames": len(set(user_frame_indices)),
            "numUniqueReferenceSequenceFrames": len(set(reference_frame_indices)),
        }
    )
    return summary


def main() -> None:
    args = parse_args()

    user_sequence = load_skeleton_sequence_npz(args.user_skeleton_npz)
    reference_sequence = load_skeleton_sequence_npz(args.reference_skeleton_npz)
    config = AlignmentConfig(dtw_radius=args.dtw_radius)
    alignment_report = build_alignment_report(
        user_sequence=user_sequence,
        reference_sequence=reference_sequence,
        config=config,
    )

    user_render_frame_indices = _load_render_frame_indices(args.user_render_npz)
    reference_render_frame_indices = _load_render_frame_indices(args.reference_render_npz)

    alignment_path = alignment_report.get("alignmentPath") or []
    if not alignment_path:
        raise ValueError("Alignment report does not contain alignmentPath")

    user_source_frame_indices = [
        int(user_render_frame_indices[int(item["userFrameIndex"])]) for item in alignment_path
    ]
    reference_source_frame_indices = [
        int(reference_render_frame_indices[int(item["referenceFrameIndex"])])
        for item in alignment_path
    ]

    user_frames = _load_resized_frames(
        args.user_video,
        user_source_frame_indices,
        panel_height=args.panel_height,
    )
    reference_frames = _load_resized_frames(
        args.reference_video,
        reference_source_frame_indices,
        panel_height=args.panel_height,
    )

    first_frame = _compose_alignment_frame(
        step_index=0,
        alignment_item=alignment_path[0],
        user_frame=user_frames[user_source_frame_indices[0]],
        reference_frame=reference_frames[reference_source_frame_indices[0]],
        user_label=args.user_label,
        reference_label=args.reference_label,
        user_source_frame_index=user_source_frame_indices[0],
        reference_source_frame_index=reference_source_frame_indices[0],
    )

    output_video = Path(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.output_fps),
        (int(first_frame.shape[1]), int(first_frame.shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_video}")

    try:
        writer.write(first_frame)
        for step_index, alignment_item in enumerate(alignment_path[1:], start=1):
            user_source_frame_index = user_source_frame_indices[step_index]
            reference_source_frame_index = reference_source_frame_indices[step_index]
            frame = _compose_alignment_frame(
                step_index=step_index,
                alignment_item=alignment_item,
                user_frame=user_frames[user_source_frame_index],
                reference_frame=reference_frames[reference_source_frame_index],
                user_label=args.user_label,
                reference_label=args.reference_label,
                user_source_frame_index=user_source_frame_index,
                reference_source_frame_index=reference_source_frame_index,
            )
            writer.write(frame)
    finally:
        writer.release()

    output_json_path: Path | None = None
    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(alignment_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = _build_video_summary(
        alignment_report,
        output_video=output_video,
        output_json=output_json_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
