from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .technique_alignment import SkeletonSequence


@dataclass
class VideoExtractionConfig:
    target_fps: float = 12.0
    start_time_sec: float = 0.0
    end_time_sec: float | None = None
    max_frames: int = 240
    bbox_thr: float = 0.5
    use_mask: bool = False
    inference_type: str = "body"


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox.tolist()
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(bbox: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox.tolist()
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.tolist()
    bx1, by1, bx2, by2 = b.tolist()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = max(1e-6, area_a + area_b - intersection)
    return intersection / union


def _point_in_bbox(point_px: tuple[float, float], bbox: np.ndarray) -> bool:
    x, y = point_px
    x1, y1, x2, y2 = bbox.tolist()
    return x1 <= x <= x2 and y1 <= y <= y2


def _select_person_output(
    outputs: list[dict[str, Any]],
    previous_bbox: np.ndarray | None,
    selection_point_px: tuple[float, float] | None,
) -> dict[str, Any] | None:
    if not outputs:
        return None

    if previous_bbox is not None:
        return max(outputs, key=lambda item: _bbox_iou(previous_bbox, np.asarray(item["bbox"], dtype=np.float32)))

    if selection_point_px is not None:
        containing = [
            output
            for output in outputs
            if _point_in_bbox(selection_point_px, np.asarray(output["bbox"], dtype=np.float32))
        ]
        if containing:
            return max(containing, key=lambda item: _bbox_area(np.asarray(item["bbox"], dtype=np.float32)))
        return min(
            outputs,
            key=lambda item: np.linalg.norm(
                np.asarray(_bbox_center(np.asarray(item["bbox"], dtype=np.float32)))
                - np.asarray(selection_point_px, dtype=np.float32)
            ),
        )

    return max(outputs, key=lambda item: _bbox_area(np.asarray(item["bbox"], dtype=np.float32)))


def extract_skeleton_sequence_from_video(
    video_path: str | Path,
    estimator: SAM3DBodyEstimator,
    config: VideoExtractionConfig | None = None,
    selection_point_px: tuple[float, float] | None = None,
    joint_names: tuple[str, ...] | None = None,
) -> SkeletonSequence:
    config = config or VideoExtractionConfig()
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_file}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = max(1.0, config.target_fps)
    sample_every_n_frames = max(1, int(round(source_fps / max(config.target_fps, 1e-6))))

    start_frame = max(0, int(round(config.start_time_sec * source_fps)))
    end_frame = (
        int(round(config.end_time_sec * source_fps))
        if config.end_time_sec is not None
        else None
    )

    keypoints_sequence: list[np.ndarray] = []
    timestamps: list[float] = []
    frame_index = 0
    previous_bbox: np.ndarray | None = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index < start_frame:
                frame_index += 1
                continue
            if end_frame is not None and frame_index > end_frame:
                break
            if ((frame_index - start_frame) % sample_every_n_frames) != 0:
                frame_index += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            outputs = estimator.process_one_image(
                frame_rgb,
                bbox_thr=config.bbox_thr,
                use_mask=config.use_mask,
                inference_type=config.inference_type,
            )
            selected = _select_person_output(outputs, previous_bbox, selection_point_px)
            if selected is None:
                frame_index += 1
                continue

            if "pred_keypoints_3d" not in selected:
                frame_index += 1
                continue

            keypoints = np.asarray(selected["pred_keypoints_3d"], dtype=np.float32)
            if keypoints.ndim != 2 or keypoints.shape[1] != 3:
                frame_index += 1
                continue

            keypoints_sequence.append(keypoints)
            timestamps.append(frame_index / source_fps)
            previous_bbox = np.asarray(selected["bbox"], dtype=np.float32)

            if len(keypoints_sequence) >= config.max_frames:
                break
            frame_index += 1
    finally:
        cap.release()

    if not keypoints_sequence:
        raise ValueError("No valid skeleton frames extracted from video")

    return SkeletonSequence(
        keypoints_3d=np.stack(keypoints_sequence, axis=0),
        timestamps=np.asarray(timestamps, dtype=np.float32),
        joint_names=joint_names,
    )
