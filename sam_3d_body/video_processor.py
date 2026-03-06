from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import numpy as np

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .technique_alignment import SkeletonSequence

_REMOTE_VIDEO_SCHEMES = {"http", "https"}


@dataclass
class VideoExtractionConfig:
    target_fps: float = 12.0
    start_time_sec: float = 0.0
    end_time_sec: float | None = None
    max_frames: int = 240
    bbox_thr: float = 0.5
    use_mask: bool = False
    inference_type: str = "body"


def _normalize_cam_intrinsics(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.shape != (3, 3):
        return None
    return matrix


def _horizontal_fov_deg_from_intrinsics(
    cam_intrinsics: np.ndarray | None,
    image_width: float,
) -> float | None:
    if cam_intrinsics is None:
        return None
    fx = float(cam_intrinsics[0, 0])
    if not np.isfinite(fx) or fx <= 0:
        return None
    if not np.isfinite(image_width) or image_width <= 0:
        return None
    horizontal_fov_deg = float(np.degrees(2 * np.arctan(image_width / (2.0 * fx))))
    if not np.isfinite(horizontal_fov_deg):
        return None
    if horizontal_fov_deg <= 0.0 or horizontal_fov_deg >= 180.0:
        return None
    return horizontal_fov_deg


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


def _validate_selection_bbox_xyxy(selection_bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    selection_bbox = np.asarray(selection_bbox_xyxy, dtype=np.float32).reshape(-1)
    if selection_bbox.shape[0] != 4:
        raise ValueError("selectionBbox must contain exactly 4 values: [x1, y1, x2, y2].")
    x1, y1, x2, y2 = selection_bbox.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError("selectionBbox must satisfy x2 > x1 and y2 > y1.")
    return selection_bbox


def _select_person_output_from_bbox(
    outputs: list[dict[str, Any]],
    selection_bbox: np.ndarray,
) -> dict[str, Any]:
    target_center = np.asarray(_bbox_center(selection_bbox), dtype=np.float32)

    def _score(item: dict[str, Any]) -> tuple[float, float, float]:
        output_bbox = np.asarray(item["bbox"], dtype=np.float32)
        iou = _bbox_iou(selection_bbox, output_bbox)
        center_distance = float(
            np.linalg.norm(np.asarray(_bbox_center(output_bbox), dtype=np.float32) - target_center)
        )
        # Prioritize overlap first, then nearest center, then larger boxes.
        return (iou, -center_distance, _bbox_area(output_bbox))

    return max(outputs, key=_score)


def _select_person_output(
    outputs: list[dict[str, Any]],
    previous_bbox: np.ndarray | None,
    selection_bbox: np.ndarray | None,
    selection_point_px: tuple[float, float] | None,
) -> dict[str, Any] | None:
    if not outputs:
        return None

    if previous_bbox is not None:
        return max(outputs, key=lambda item: _bbox_iou(previous_bbox, np.asarray(item["bbox"], dtype=np.float32)))

    if selection_bbox is not None:
        return _select_person_output_from_bbox(outputs, selection_bbox)

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


def _is_remote_video_path(video_path: str) -> bool:
    parsed = urlparse(video_path)
    return parsed.scheme in _REMOTE_VIDEO_SCHEMES and bool(parsed.netloc)


@contextmanager
def _resolve_video_file(video_path: str | Path):
    raw_path = str(video_path)
    if _is_remote_video_path(raw_path):
        parsed = urlparse(raw_path)
        suffix = Path(parsed.path).suffix or ".mp4"
        temp_file = tempfile.NamedTemporaryFile(
            prefix="sam3db_video_",
            suffix=suffix,
            delete=False,
        )
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            with urlopen(raw_path, timeout=30) as response, temp_path.open("wb") as output:
                shutil.copyfileobj(response, output)
            yield temp_path
            return
        except Exception as exc:
            raise FileNotFoundError(f"Video not found: {raw_path}") from exc
        finally:
            temp_path.unlink(missing_ok=True)

    local_file = Path(raw_path)
    if not local_file.exists():
        raise FileNotFoundError(f"Video not found: {local_file}")
    yield local_file


def extract_skeleton_sequence_from_video(
    video_path: str | Path,
    estimator: SAM3DBodyEstimator,
    config: VideoExtractionConfig | None = None,
    selection_bbox_xyxy: tuple[float, float, float, float] | None = None,
    selection_point_px: tuple[float, float] | None = None,
    joint_names: tuple[str, ...] | None = None,
    return_camera_metadata: bool = False,
) -> SkeletonSequence | tuple[SkeletonSequence, dict[str, Any] | None]:
    config = config or VideoExtractionConfig()
    selection_bbox: np.ndarray | None = None
    if selection_bbox_xyxy is not None:
        selection_bbox = _validate_selection_bbox_xyxy(selection_bbox_xyxy)

    with _resolve_video_file(video_path) as video_file:
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
        horizontal_fov_deg: list[float] = []
        camera_timestamps: list[float] = []
        camera_source: str | None = None
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
                selected = _select_person_output(
                    outputs,
                    previous_bbox,
                    selection_bbox,
                    selection_point_px,
                )
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

                timestamp_sec = frame_index / source_fps
                keypoints_sequence.append(keypoints)
                timestamps.append(timestamp_sec)
                previous_bbox = np.asarray(selected["bbox"], dtype=np.float32)

                cam_intrinsics = _normalize_cam_intrinsics(selected.get("cam_intrinsics"))
                horizontal_fov = _horizontal_fov_deg_from_intrinsics(
                    cam_intrinsics,
                    image_width=float(frame_rgb.shape[1]),
                )
                if horizontal_fov is not None:
                    horizontal_fov_deg.append(horizontal_fov)
                    camera_timestamps.append(timestamp_sec)
                    if camera_source is None:
                        raw_camera_source = selected.get("camera_source")
                        if isinstance(raw_camera_source, str) and raw_camera_source.strip():
                            camera_source = raw_camera_source.strip()

                if len(keypoints_sequence) >= config.max_frames:
                    break
                frame_index += 1
        finally:
            cap.release()

        if not keypoints_sequence:
            raise ValueError("No valid skeleton frames extracted from video")

        sequence = SkeletonSequence(
            keypoints_3d=np.stack(keypoints_sequence, axis=0),
            timestamps=np.asarray(timestamps, dtype=np.float32),
            joint_names=joint_names,
        )
        if not return_camera_metadata:
            return sequence

        camera_metadata: dict[str, Any] | None = None
        if horizontal_fov_deg and len(horizontal_fov_deg) == len(camera_timestamps):
            camera_metadata = {
                "source": camera_source or "unknown",
                "horizontalFovDeg": [float(value) for value in horizontal_fov_deg],
                "timestamps": [float(value) for value in camera_timestamps],
            }
        return sequence, camera_metadata
