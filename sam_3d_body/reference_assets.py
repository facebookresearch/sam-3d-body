from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import cv2
import numpy as np

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .technique_alignment import SkeletonSequence, normalize_skeleton_sequence
from .video_processor import (
    VideoExtractionConfig,
    _horizontal_fov_deg_from_intrinsics,
    _is_remote_video_path,
    _normalize_cam_intrinsics,
    _resolve_video_file,
    _select_person_output,
    _validate_selection_bbox_xyxy,
)


DEFAULT_METADATA_FILENAME = "metadata.json"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
RENDER_ASSET_SCHEMA_VERSION = "technique_reference_render.v1"


@dataclass
class ReferenceExtractionResult:
    sequence: SkeletonSequence
    selected_outputs: list[dict[str, Any]]
    frame_indices: np.ndarray
    source_fps: float
    image_size_hw: tuple[int, int]


@dataclass(frozen=True)
class ReferenceAssetBundle:
    entry: "ReferenceVideoEntry"
    sequence: SkeletonSequence
    skeleton_path: Path
    render_path: Path
    asset_metadata: dict[str, Any]
    render_asset: dict[str, Any]


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = normalized.strip("_")
    return normalized or "reference"


def _normalize_video_path(video_path: str | Path) -> str | Path:
    raw_path = str(video_path).strip()
    if _is_remote_video_path(raw_path):
        return raw_path
    return Path(raw_path)


def _derive_reference_id(video_path: str | Path) -> str:
    raw_path = str(video_path).strip()
    if _is_remote_video_path(raw_path):
        return _slugify(Path(urlparse(raw_path).path).stem)
    return _slugify(Path(raw_path).stem)


@dataclass(frozen=True)
class ReferenceVideoEntry:
    video_path: str | Path
    action_type: str
    reference_id: str | None = None
    athlete_name: str | None = None
    camera_view: str | None = None
    handedness: str | None = None
    selection_bbox_xyxy: tuple[float, float, float, float] | None = None
    selection_point_px: tuple[float, float] | None = None
    video_config: VideoExtractionConfig = field(default_factory=VideoExtractionConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_path", _normalize_video_path(self.video_path))
        cleaned_action_type = self.action_type.strip()
        if not cleaned_action_type:
            raise ValueError("action_type must not be empty.")
        object.__setattr__(self, "action_type", cleaned_action_type)

        if self.reference_id is None or not self.reference_id.strip():
            object.__setattr__(
                self,
                "reference_id",
                _derive_reference_id(self.video_path),
            )
        else:
            object.__setattr__(self, "reference_id", _slugify(self.reference_id))

        if self.selection_bbox_xyxy is not None:
            bbox = _validate_selection_bbox_xyxy(self.selection_bbox_xyxy)
            object.__setattr__(
                self,
                "selection_bbox_xyxy",
                tuple(float(value) for value in bbox.tolist()),
            )

        if self.selection_bbox_xyxy is not None and self.selection_point_px is not None:
            raise ValueError("selection_bbox_xyxy and selection_point_px are mutually exclusive.")


def _parse_video_config(
    raw_config: dict[str, Any] | None,
    default: VideoExtractionConfig | None = None,
) -> VideoExtractionConfig:
    base = default or VideoExtractionConfig()
    if raw_config is None:
        return VideoExtractionConfig(
            target_fps=base.target_fps,
            start_time_sec=base.start_time_sec,
            end_time_sec=base.end_time_sec,
            max_frames=base.max_frames,
            bbox_thr=base.bbox_thr,
            use_mask=base.use_mask,
            inference_type=base.inference_type,
        )

    return VideoExtractionConfig(
        target_fps=float(raw_config.get("targetFps", base.target_fps)),
        start_time_sec=float(raw_config.get("startTimeSec", base.start_time_sec)),
        end_time_sec=(
            None
            if raw_config.get("endTimeSec", base.end_time_sec) is None
            else float(raw_config.get("endTimeSec", base.end_time_sec))
        ),
        max_frames=int(raw_config.get("maxFrames", base.max_frames)),
        bbox_thr=float(raw_config.get("bboxThr", base.bbox_thr)),
        use_mask=bool(raw_config.get("useMask", base.use_mask)),
        inference_type=str(raw_config.get("inferenceType", base.inference_type)),
    )


def discover_reference_videos(
    input_dir: str | Path,
    *,
    action_type: str,
    athlete_name: str | None = None,
    camera_view: str | None = None,
    handedness: str | None = None,
    video_config: VideoExtractionConfig | None = None,
) -> list[ReferenceVideoEntry]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Input path must be a directory: {root}")

    entries: list[ReferenceVideoEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        entries.append(
            ReferenceVideoEntry(
                video_path=path,
                action_type=action_type,
                athlete_name=athlete_name,
                camera_view=camera_view,
                handedness=handedness,
                video_config=_parse_video_config(None, default=video_config),
            )
        )
    return entries


def load_reference_manifest(
    manifest_path: str | Path,
    *,
    default_action_type: str | None = None,
    default_athlete_name: str | None = None,
    default_camera_view: str | None = None,
    default_handedness: str | None = None,
    default_video_config: VideoExtractionConfig | None = None,
) -> list[ReferenceVideoEntry]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("assets"), list):
        raw_items = payload["assets"]
    else:
        raise ValueError("Manifest must be a JSON array or an object with an 'assets' array.")

    entries: list[ReferenceVideoEntry] = []
    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest item at index {idx} must be an object.")
        video_path_raw = raw.get("videoPath")
        if not isinstance(video_path_raw, str) or not video_path_raw.strip():
            raise ValueError(f"Manifest item at index {idx} is missing valid 'videoPath'.")

        action_type = str(raw.get("actionType") or default_action_type or "").strip()
        if not action_type:
            raise ValueError(
                f"Manifest item at index {idx} is missing 'actionType', and no default_action_type was provided."
            )

        selection_point_px_raw = raw.get("selectionPointPx")
        selection_point_px: tuple[float, float] | None = None
        if selection_point_px_raw is not None:
            if (
                not isinstance(selection_point_px_raw, (list, tuple))
                or len(selection_point_px_raw) != 2
            ):
                raise ValueError(
                    f"Manifest item at index {idx} has invalid 'selectionPointPx'. Expected [x, y]."
                )
            selection_point_px = (
                float(selection_point_px_raw[0]),
                float(selection_point_px_raw[1]),
            )

        selection_bbox_xyxy_raw = raw.get("selectionBbox")
        selection_bbox_xyxy: tuple[float, float, float, float] | None = None
        if selection_bbox_xyxy_raw is not None:
            if (
                not isinstance(selection_bbox_xyxy_raw, (list, tuple))
                or len(selection_bbox_xyxy_raw) != 4
            ):
                raise ValueError(
                    f"Manifest item at index {idx} has invalid 'selectionBbox'. Expected [x1, y1, x2, y2]."
                )
            selection_bbox_xyxy = (
                float(selection_bbox_xyxy_raw[0]),
                float(selection_bbox_xyxy_raw[1]),
                float(selection_bbox_xyxy_raw[2]),
                float(selection_bbox_xyxy_raw[3]),
            )

        entries.append(
            ReferenceVideoEntry(
                video_path=video_path_raw,
                action_type=action_type,
                reference_id=raw.get("referenceId"),
                athlete_name=raw.get("athleteName", default_athlete_name),
                camera_view=raw.get("cameraView", default_camera_view),
                handedness=raw.get("handedness", default_handedness),
                selection_bbox_xyxy=selection_bbox_xyxy,
                selection_point_px=selection_point_px,
                video_config=_parse_video_config(
                    raw.get("videoConfig"),
                    default=default_video_config,
                ),
                metadata=(
                    dict(raw["metadata"])
                    if isinstance(raw.get("metadata"), dict)
                    else {}
                ),
            )
        )

    return entries


def save_skeleton_sequence_npz(
    sequence: SkeletonSequence,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "keypoints_3d": sequence.keypoints_3d,
        "timestamps": sequence.timestamps,
    }
    if sequence.joint_names is not None:
        payload["joint_names"] = np.asarray(sequence.joint_names, dtype=object)
    np.savez(path, **payload)


def _extract_reference_result(
    *,
    entry: ReferenceVideoEntry,
    estimator: SAM3DBodyEstimator,
) -> ReferenceExtractionResult:
    selection_bbox = (
        _validate_selection_bbox_xyxy(entry.selection_bbox_xyxy)
        if entry.selection_bbox_xyxy is not None
        else None
    )
    with _resolve_video_file(entry.video_path) as video_file:
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_file}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0:
            source_fps = max(1.0, entry.video_config.target_fps)
        sample_every_n_frames = max(
            1, int(round(source_fps / max(entry.video_config.target_fps, 1e-6)))
        )

        start_frame = max(0, int(round(entry.video_config.start_time_sec * source_fps)))
        end_frame = (
            int(round(entry.video_config.end_time_sec * source_fps))
            if entry.video_config.end_time_sec is not None
            else None
        )

        keypoints_sequence: list[np.ndarray] = []
        timestamps: list[float] = []
        frame_indices: list[int] = []
        selected_outputs: list[dict[str, Any]] = []

        frame_index = 0
        previous_bbox: np.ndarray | None = None
        image_size_hw: tuple[int, int] | None = None
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
                if image_size_hw is None:
                    image_size_hw = (int(frame_rgb.shape[0]), int(frame_rgb.shape[1]))
                outputs = estimator.process_one_image(
                    frame_rgb,
                    bbox_thr=entry.video_config.bbox_thr,
                    use_mask=entry.video_config.use_mask,
                    inference_type=entry.video_config.inference_type,
                )

                selected = _select_person_output(
                    outputs,
                    previous_bbox,
                    selection_bbox,
                    entry.selection_point_px,
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

                keypoints_sequence.append(keypoints)
                timestamps.append(frame_index / source_fps)
                frame_indices.append(frame_index)
                selected_outputs.append(selected)

                previous_bbox = np.asarray(selected["bbox"], dtype=np.float32)
                if len(keypoints_sequence) >= entry.video_config.max_frames:
                    break
                frame_index += 1
        finally:
            cap.release()

        if not keypoints_sequence:
            raise ValueError("No valid skeleton frames extracted from video")
        if image_size_hw is None:
            raise ValueError("Failed to capture image size from video")

        return ReferenceExtractionResult(
            sequence=SkeletonSequence(
                keypoints_3d=np.stack(keypoints_sequence, axis=0),
                timestamps=np.asarray(timestamps, dtype=np.float32),
                joint_names=None,
            ),
            selected_outputs=selected_outputs,
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            source_fps=float(source_fps),
            image_size_hw=image_size_hw,
        )


def _stack_optional_field(
    selected_outputs: list[dict[str, Any]],
    key: str,
) -> np.ndarray | None:
    values: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for output in selected_outputs:
        if key not in output or output[key] is None:
            return None
        value = np.asarray(output[key])
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise ValueError(
                f"Inconsistent shape for field '{key}': {value.shape} vs {expected_shape}"
            )
        values.append(value)
    if not values:
        return None
    return np.stack(values, axis=0)


def _cast_float_array(array: np.ndarray, float_dtype: str) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.floating):
        return array
    target_dtype = np.float16 if float_dtype == "float16" else np.float32
    if array.dtype == target_dtype:
        return array
    return array.astype(target_dtype)


def save_render_asset_npz(
    *,
    extraction: ReferenceExtractionResult,
    estimator: SAM3DBodyEstimator,
    output_path: str | Path,
    float_dtype: str = "float16",
    include_masks: bool = False,
) -> dict[str, Any]:
    if float_dtype not in {"float16", "float32"}:
        raise ValueError("float_dtype must be one of: float16, float32")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image_height, image_width = extraction.image_size_hw
    principal_point_xy = np.asarray(
        [image_width * 0.5, image_height * 0.5], dtype=np.float32
    )

    payload: dict[str, Any] = {
        "schema_version": np.asarray(RENDER_ASSET_SCHEMA_VERSION),
        "keypoints_3d": extraction.sequence.keypoints_3d,
        "keypoints_3d_normalized": normalize_skeleton_sequence(extraction.sequence),
        "timestamps": extraction.sequence.timestamps,
        "frame_indices": extraction.frame_indices,
        "source_fps": np.asarray(extraction.source_fps, dtype=np.float32),
        "image_size_hw": np.asarray(extraction.image_size_hw, dtype=np.int32),
        "principal_point_xy": principal_point_xy,
    }
    if extraction.sequence.joint_names is not None:
        payload["joint_names"] = np.asarray(extraction.sequence.joint_names, dtype=object)

    optional_field_map = {
        "bbox_xyxy": "bbox",
        "keypoints_2d": "pred_keypoints_2d",
        "vertices_3d": "pred_vertices",
        "cam_t": "pred_cam_t",
        "cam_intrinsics": "cam_intrinsics",
        "focal_length": "focal_length",
        "global_rot": "global_rot",
        "body_pose_params": "body_pose_params",
        "hand_pose_params": "hand_pose_params",
        "shape_params": "shape_params",
        "scale_params": "scale_params",
        "pred_joint_coords": "pred_joint_coords",
        "pred_global_rots": "pred_global_rots",
        "mhr_model_params": "mhr_model_params",
    }
    for payload_key, output_key in optional_field_map.items():
        stacked = _stack_optional_field(extraction.selected_outputs, output_key)
        if stacked is None:
            continue
        payload[payload_key] = stacked

    horizontal_fov_deg: np.ndarray | None = None
    camera_source: str | None = None
    if "cam_intrinsics" in payload:
        cam_intrinsics = np.asarray(payload["cam_intrinsics"], dtype=np.float32)
        if cam_intrinsics.ndim == 4 and cam_intrinsics.shape[1] == 1:
            cam_intrinsics = cam_intrinsics[:, 0]
        if cam_intrinsics.ndim == 3 and cam_intrinsics.shape[1:] == (3, 3):
            payload["cam_intrinsics"] = cam_intrinsics
            hfov_values: list[float] = []
            for frame_intrinsics in cam_intrinsics:
                normalized_intrinsics = _normalize_cam_intrinsics(frame_intrinsics)
                hfov = _horizontal_fov_deg_from_intrinsics(
                    normalized_intrinsics,
                    float(image_width),
                )
                if hfov is None:
                    hfov_values = []
                    break
                hfov_values.append(hfov)
            if len(hfov_values) == int(cam_intrinsics.shape[0]):
                horizontal_fov_deg = np.asarray(hfov_values, dtype=np.float32)
                payload["horizontal_fov_deg"] = horizontal_fov_deg
        else:
            payload.pop("cam_intrinsics", None)

    camera_source_values = _stack_optional_field(
        extraction.selected_outputs,
        "camera_source",
    )
    if camera_source_values is not None:
        raw_sources = np.asarray(camera_source_values).reshape(-1).tolist()
        normalized_sources = [str(item).strip() for item in raw_sources if str(item).strip()]
        if normalized_sources:
            camera_source = normalized_sources[0]
            payload["camera_source"] = np.asarray(camera_source)

    if include_masks:
        masks = _stack_optional_field(extraction.selected_outputs, "mask")
        if masks is not None:
            payload["masks"] = masks.astype(np.uint8)

    if hasattr(estimator, "faces") and getattr(estimator, "faces") is not None:
        payload["faces"] = np.asarray(getattr(estimator, "faces"), dtype=np.int32)

    for key, value in list(payload.items()):
        payload[key] = _cast_float_array(np.asarray(value), float_dtype)

    np.savez_compressed(path, **payload)
    return {
        "path": str(path),
        "schemaVersion": RENDER_ASSET_SCHEMA_VERSION,
        "floatDtype": float_dtype,
        "fields": sorted(payload.keys()),
        "cameraSource": camera_source,
        "horizontalFovDeg": (
            horizontal_fov_deg.astype(np.float32).tolist() if horizontal_fov_deg is not None else None
        ),
        "timestamps": (
            extraction.sequence.timestamps.astype(np.float32).tolist()
            if horizontal_fov_deg is not None
            else None
        ),
        "horizontalFovDegCount": (
            int(horizontal_fov_deg.shape[0]) if horizontal_fov_deg is not None else 0
        ),
        "horizontalFovDegRange": (
            [
                float(np.min(horizontal_fov_deg)),
                float(np.max(horizontal_fov_deg)),
            ]
            if horizontal_fov_deg is not None and horizontal_fov_deg.size > 0
            else None
        ),
    }


def _sequence_duration_seconds(sequence: SkeletonSequence) -> float:
    if sequence.num_frames <= 1:
        return 0.0
    return float(sequence.timestamps[-1] - sequence.timestamps[0])


def _validate_unique_reference_ids(entries: Iterable[ReferenceVideoEntry]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.reference_id in seen:
            raise ValueError(f"Duplicate reference_id detected: {entry.reference_id}")
        seen.add(entry.reference_id or "")


def _build_asset_metadata(
    entry: ReferenceVideoEntry,
    sequence: SkeletonSequence,
    output_npz: Path,
    extraction: ReferenceExtractionResult,
    render_asset: dict[str, Any],
) -> dict[str, Any]:
    return {
        "referenceId": entry.reference_id,
        "actionType": entry.action_type,
        "athleteName": entry.athlete_name,
        "cameraView": entry.camera_view,
        "handedness": entry.handedness,
        "sourceVideoPath": str(entry.video_path),
        "skeletonPath": str(output_npz),
        "renderAssetPath": render_asset["path"],
        "renderAssetSchemaVersion": render_asset["schemaVersion"],
        "renderAssetFloatDtype": render_asset["floatDtype"],
        "renderAssetFields": render_asset["fields"],
        "cameraSource": render_asset.get("cameraSource"),
        "horizontalFovDegCount": render_asset.get("horizontalFovDegCount"),
        "horizontalFovDegRange": render_asset.get("horizontalFovDegRange"),
        "selectionPointPx": (
            [entry.selection_point_px[0], entry.selection_point_px[1]]
            if entry.selection_point_px is not None
            else None
        ),
        "selectionBbox": (
            [
                entry.selection_bbox_xyxy[0],
                entry.selection_bbox_xyxy[1],
                entry.selection_bbox_xyxy[2],
                entry.selection_bbox_xyxy[3],
            ]
            if entry.selection_bbox_xyxy is not None
            else None
        ),
        "videoConfig": {
            "targetFps": entry.video_config.target_fps,
            "startTimeSec": entry.video_config.start_time_sec,
            "endTimeSec": entry.video_config.end_time_sec,
            "maxFrames": entry.video_config.max_frames,
            "bboxThr": entry.video_config.bbox_thr,
            "useMask": entry.video_config.use_mask,
            "inferenceType": entry.video_config.inference_type,
        },
        "numFrames": sequence.num_frames,
        "numJoints": sequence.num_joints,
        "durationSec": _sequence_duration_seconds(sequence),
        "sourceFps": extraction.source_fps,
        "imageSizeHw": [extraction.image_size_hw[0], extraction.image_size_hw[1]],
        "frameIndices": extraction.frame_indices.tolist(),
        "jointNames": list(sequence.joint_names) if sequence.joint_names is not None else None,
        "metadata": entry.metadata,
    }


def build_reference_assets_metadata(
    asset_entries: list[dict[str, Any]],
    *,
    skeleton_version: str = "sam3db_v1",
    fov_estimator_name: str | None = None,
    fov_estimator_path: str | None = None,
    render_asset_float_dtype: str = "float16",
    render_include_masks: bool = False,
) -> dict[str, Any]:
    return {
        "schemaVersion": "technique_reference_assets.v2",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "skeletonVersion": skeleton_version,
        "fovEstimator": {
            "name": fov_estimator_name,
            "path": fov_estimator_path,
        }
        if fov_estimator_name is not None
        else None,
        "jointUnit": "model_space",
        "renderAssetEnabled": True,
        "renderAssetSchemaVersion": RENDER_ASSET_SCHEMA_VERSION,
        "renderAssetFloatDtype": render_asset_float_dtype,
        "renderAssetIncludeMasks": render_include_masks,
        "assetCount": len(asset_entries),
        "assets": asset_entries,
    }


def save_reference_assets_metadata(
    metadata: dict[str, Any],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def build_reference_asset_bundle(
    entry: ReferenceVideoEntry,
    *,
    estimator: SAM3DBodyEstimator,
    output_dir: str | Path,
    render_asset_float_dtype: str = "float16",
    render_include_masks: bool = False,
    overwrite: bool = False,
) -> ReferenceAssetBundle:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if not _is_remote_video_path(str(entry.video_path)) and not Path(entry.video_path).exists():
        raise FileNotFoundError(f"Reference video not found: {entry.video_path}")

    output_npz = output_root / f"{entry.reference_id}.npz"
    if output_npz.exists() and not overwrite:
        raise FileExistsError(
            f"Reference output already exists: {output_npz}. Set overwrite=True to replace."
        )

    render_output_npz = output_root / f"{entry.reference_id}.render.npz"
    if render_output_npz.exists() and not overwrite:
        raise FileExistsError(
            f"Reference render output already exists: {render_output_npz}. Set overwrite=True to replace."
        )

    extraction = _extract_reference_result(
        entry=entry,
        estimator=estimator,
    )
    sequence = extraction.sequence
    save_skeleton_sequence_npz(sequence, output_npz)
    render_asset = save_render_asset_npz(
        extraction=extraction,
        estimator=estimator,
        output_path=render_output_npz,
        float_dtype=render_asset_float_dtype,
        include_masks=render_include_masks,
    )
    asset_metadata = _build_asset_metadata(entry, sequence, output_npz, extraction, render_asset)
    return ReferenceAssetBundle(
        entry=entry,
        sequence=sequence,
        skeleton_path=output_npz,
        render_path=render_output_npz,
        asset_metadata=asset_metadata,
        render_asset=render_asset,
    )


def build_reference_assets(
    entries: list[ReferenceVideoEntry],
    *,
    estimator: SAM3DBodyEstimator,
    output_dir: str | Path,
    skeleton_version: str = "sam3db_v1",
    fov_estimator_name: str | None = None,
    fov_estimator_path: str | None = None,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
    render_asset_float_dtype: str = "float16",
    render_include_masks: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not entries:
        raise ValueError("No reference videos provided.")

    _validate_unique_reference_ids(entries)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / metadata_filename
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"Metadata file already exists: {metadata_path}. Set overwrite=True to replace."
        )

    bundles = [
        build_reference_asset_bundle(
            entry,
            estimator=estimator,
            output_dir=output_root,
            render_asset_float_dtype=render_asset_float_dtype,
            render_include_masks=render_include_masks,
            overwrite=overwrite,
        )
        for entry in entries
    ]
    metadata = build_reference_assets_metadata(
        [bundle.asset_metadata for bundle in bundles],
        skeleton_version=skeleton_version,
        fov_estimator_name=fov_estimator_name,
        fov_estimator_path=fov_estimator_path,
        render_asset_float_dtype=render_asset_float_dtype,
        render_include_masks=render_include_masks,
    )
    save_reference_assets_metadata(metadata, metadata_path)
    return metadata
