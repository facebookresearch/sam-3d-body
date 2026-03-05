from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from sam_3d_body.reference_assets import (
    ReferenceVideoEntry,
    build_reference_assets,
    discover_reference_videos,
    load_reference_manifest,
)
from sam_3d_body.video_processor import VideoExtractionConfig


def _write_dummy_video(path: Path, fps: float, num_frames: int) -> None:
    width, height = 120, 80
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to create test video")
    try:
        for frame_idx in range(num_frames):
            color = int((frame_idx * 13) % 255)
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class _DummyEstimator:
    def __init__(self) -> None:
        self.call_count = 0
        self.faces = np.array([[0, 1, 2], [2, 3, 4]], dtype=np.int32)

    def process_one_image(self, _frame_rgb: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        value = float(self.call_count)
        self.call_count += 1
        keypoints = np.array(
            [
                [value, 0.0, 0.0],
                [value + 1.0, 0.0, 0.0],
                [value + 2.0, 0.0, 0.0],
                [value + 3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        keypoints_2d = keypoints[:, :2] * 4.0 + 16.0
        vertices = np.array(
            [
                [value, 0.0, 0.0],
                [value + 0.2, 0.1, 0.0],
                [value + 0.4, 0.3, 0.0],
                [value + 0.6, 0.4, 0.1],
                [value + 0.8, 0.5, 0.2],
            ],
            dtype=np.float32,
        )
        return [
            {
                "bbox": np.array([8, 8, 80, 72], dtype=np.float32),
                "pred_keypoints_3d": keypoints,
                "pred_keypoints_2d": keypoints_2d,
                "pred_vertices": vertices,
                "pred_cam_t": np.array([0.1, -0.05, 3.2], dtype=np.float32),
                "focal_length": np.float32(1200.0),
                "global_rot": np.array([0.0, 0.1, 0.2], dtype=np.float32),
                "body_pose_params": np.linspace(0.0, 1.0, 12, dtype=np.float32),
                "hand_pose_params": np.linspace(0.0, 1.0, 6, dtype=np.float32),
                "shape_params": np.linspace(-1.0, 1.0, 10, dtype=np.float32),
                "scale_params": np.array([1.0], dtype=np.float32),
                "pred_joint_coords": keypoints.copy(),
                "pred_global_rots": np.tile(np.eye(3, dtype=np.float32), (4, 1, 1)),
                "mhr_model_params": np.linspace(0.0, 1.0, 16, dtype=np.float32),
                "mask": np.zeros((80, 120, 1), dtype=np.uint8),
            }
        ]


def test_discover_reference_videos_filters_and_sorts(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir(parents=True)
    (video_dir / "b_ref.mp4").touch()
    (video_dir / "a_ref.mov").touch()
    (video_dir / "ignore.txt").touch()

    entries = discover_reference_videos(
        video_dir,
        action_type="smash",
        athlete_name="lin",
        camera_view="side",
        video_config=VideoExtractionConfig(target_fps=9.0, max_frames=99),
    )

    assert [entry.reference_id for entry in entries] == ["a_ref", "b_ref"]
    assert all(entry.action_type == "smash" for entry in entries)
    assert all(entry.athlete_name == "lin" for entry in entries)
    assert all(entry.camera_view == "side" for entry in entries)
    assert all(entry.video_config.target_fps == 9.0 for entry in entries)
    assert all(entry.video_config.max_frames == 99 for entry in entries)


def test_load_reference_manifest_applies_defaults(tmp_path: Path) -> None:
    video_path = tmp_path / "ref_clip.mp4"
    video_path.touch()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "videoPath": str(video_path),
                        "referenceId": "Smash Pro 01",
                        "selectionPointPx": [123, 45],
                        "metadata": {"source": "youtube"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    entries = load_reference_manifest(
        manifest_path,
        default_action_type="smash",
        default_athlete_name="pro_player",
        default_camera_view="back",
        default_handedness="right",
        default_video_config=VideoExtractionConfig(target_fps=15.0, max_frames=200),
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry.reference_id == "smash_pro_01"
    assert entry.action_type == "smash"
    assert entry.athlete_name == "pro_player"
    assert entry.camera_view == "back"
    assert entry.handedness == "right"
    assert entry.selection_point_px == (123.0, 45.0)
    assert entry.video_config.target_fps == 15.0
    assert entry.video_config.max_frames == 200
    assert entry.metadata["source"] == "youtube"


def test_build_reference_assets_writes_npz_and_metadata(tmp_path: Path) -> None:
    video1 = tmp_path / "smash_1.mp4"
    video2 = tmp_path / "smash_2.mp4"
    _write_dummy_video(video1, fps=10.0, num_frames=8)
    _write_dummy_video(video2, fps=12.0, num_frames=12)

    entries = [
        ReferenceVideoEntry(
            video_path=video1,
            action_type="smash",
            reference_id="smash_pro_001",
            athlete_name="athlete_a",
            camera_view="side",
            handedness="right",
            selection_point_px=(20.0, 20.0),
            video_config=VideoExtractionConfig(target_fps=5.0, max_frames=3),
        ),
        ReferenceVideoEntry(
            video_path=video2,
            action_type="smash",
            reference_id="smash_pro_002",
            athlete_name="athlete_b",
            camera_view="back",
            handedness="left",
            video_config=VideoExtractionConfig(target_fps=6.0, max_frames=4),
            metadata={"source": "manual_pick"},
        ),
    ]

    output_dir = tmp_path / "out_assets"
    metadata = build_reference_assets(
        entries,
        estimator=_DummyEstimator(),  # type: ignore[arg-type]
        output_dir=output_dir,
        skeleton_version="test_v1",
    )

    assert metadata["schemaVersion"] == "technique_reference_assets.v2"
    assert metadata["skeletonVersion"] == "test_v1"
    assert metadata["renderAssetEnabled"] is True
    assert metadata["renderAssetSchemaVersion"] == "technique_reference_render.v1"
    assert metadata["assetCount"] == 2
    assert len(metadata["assets"]) == 2

    metadata_file = output_dir / "metadata.json"
    assert metadata_file.exists()
    file_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert file_metadata["assetCount"] == 2

    for asset in file_metadata["assets"]:
        npz_path = Path(asset["skeletonPath"])
        assert npz_path.exists()
        npz_data = np.load(npz_path, allow_pickle=True)
        assert "keypoints_3d" in npz_data
        assert "timestamps" in npz_data
        assert npz_data["keypoints_3d"].shape[2] == 3
        assert asset["renderAssetPath"] is not None
        render_npz_path = Path(asset["renderAssetPath"])
        assert render_npz_path.exists()
        render_data = np.load(render_npz_path, allow_pickle=True)
        assert render_data["schema_version"].item() == "technique_reference_render.v1"
        assert "vertices_3d" in render_data
        assert "keypoints_2d" in render_data
        assert "faces" in render_data
        assert "frame_indices" in render_data
        assert "image_size_hw" in render_data
        assert render_data["vertices_3d"].shape[2] == 3

def test_build_reference_assets_rejects_duplicate_reference_id(tmp_path: Path) -> None:
    video = tmp_path / "dup.mp4"
    _write_dummy_video(video, fps=10.0, num_frames=4)
    entries = [
        ReferenceVideoEntry(video_path=video, action_type="smash", reference_id="dup_id"),
        ReferenceVideoEntry(video_path=video, action_type="smash", reference_id="dup_id"),
    ]

    with pytest.raises(ValueError, match="Duplicate reference_id"):
        build_reference_assets(
            entries,
            estimator=_DummyEstimator(),  # type: ignore[arg-type]
            output_dir=tmp_path / "out",
        )
