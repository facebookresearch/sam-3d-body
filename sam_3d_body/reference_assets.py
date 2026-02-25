from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .technique_alignment import SkeletonSequence
from .video_processor import VideoExtractionConfig, extract_skeleton_sequence_from_video


DEFAULT_METADATA_FILENAME = "metadata.json"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = normalized.strip("_")
    return normalized or "reference"


def _derive_reference_id(video_path: Path) -> str:
    return _slugify(video_path.stem)


@dataclass(frozen=True)
class ReferenceVideoEntry:
    video_path: Path
    action_type: str
    reference_id: str | None = None
    athlete_name: str | None = None
    camera_view: str | None = None
    handedness: str | None = None
    selection_point_px: tuple[float, float] | None = None
    video_config: VideoExtractionConfig = field(default_factory=VideoExtractionConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_path", Path(self.video_path))
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

        entries.append(
            ReferenceVideoEntry(
                video_path=Path(video_path_raw),
                action_type=action_type,
                reference_id=raw.get("referenceId"),
                athlete_name=raw.get("athleteName", default_athlete_name),
                camera_view=raw.get("cameraView", default_camera_view),
                handedness=raw.get("handedness", default_handedness),
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
) -> dict[str, Any]:
    return {
        "referenceId": entry.reference_id,
        "actionType": entry.action_type,
        "athleteName": entry.athlete_name,
        "cameraView": entry.camera_view,
        "handedness": entry.handedness,
        "sourceVideoPath": str(entry.video_path),
        "skeletonPath": str(output_npz),
        "selectionPointPx": (
            [entry.selection_point_px[0], entry.selection_point_px[1]]
            if entry.selection_point_px is not None
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
        "jointNames": list(sequence.joint_names) if sequence.joint_names is not None else None,
        "metadata": entry.metadata,
    }


def build_reference_assets(
    entries: list[ReferenceVideoEntry],
    *,
    estimator: SAM3DBodyEstimator,
    output_dir: str | Path,
    skeleton_version: str = "sam3db_v1",
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
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

    assets: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.video_path.exists():
            raise FileNotFoundError(f"Reference video not found: {entry.video_path}")
        output_npz = output_root / f"{entry.reference_id}.npz"
        if output_npz.exists() and not overwrite:
            raise FileExistsError(
                f"Reference output already exists: {output_npz}. Set overwrite=True to replace."
            )

        sequence = extract_skeleton_sequence_from_video(
            video_path=entry.video_path,
            estimator=estimator,
            config=entry.video_config,
            selection_point_px=entry.selection_point_px,
        )
        save_skeleton_sequence_npz(sequence, output_npz)
        assets.append(_build_asset_metadata(entry, sequence, output_npz))

    metadata = {
        "schemaVersion": "technique_reference_assets.v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "skeletonVersion": skeleton_version,
        "jointUnit": "model_space",
        "assetCount": len(assets),
        "assets": assets,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata

