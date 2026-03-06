from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from scripts import build_reference_assets as script


def _write_dummy_video(path: Path, fps: float = 10.0, num_frames: int = 8) -> None:
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
                "cam_intrinsics": np.array(
                    [
                        [800.0, 0.0, 60.0],
                        [0.0, 800.0, 40.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                "camera_source": "moge2",
                "mask": np.zeros((80, 120, 1), dtype=np.uint8),
            }
        ]


def test_to_sql_literal_handles_common_types() -> None:
    assert script._to_sql_literal(None) == "NULL"
    assert script._to_sql_literal(True) == "1"
    assert script._to_sql_literal(False) == "0"
    assert script._to_sql_literal(12) == "12"
    assert script._to_sql_literal(1.5) == "1.5"
    assert script._to_sql_literal("a'b") == "'a''b'"


def test_parse_wrangler_json_output_with_prefix_lines() -> None:
    raw_output = (
        "Proxy environment variables detected. We'll use your proxy for fetch requests.\n"
        "[{\"results\":[{\"name\":\"id\"}],\"success\":true}]"
    )
    parsed = script._parse_wrangler_json_output(raw_output)
    assert isinstance(parsed, list)
    assert parsed[0]["results"][0]["name"] == "id"


def test_parse_args_requires_video_path() -> None:
    with pytest.raises(SystemExit):
        script.parse_args(["--output-dir", "out", "--action-type", "smash"])


def test_parse_args_rejects_legacy_batch_flags() -> None:
    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--input-dir",
                "videos",
                "--output-dir",
                "out",
                "--action-type",
                "smash",
            ]
        )


def test_load_entry_defaults_reference_id_from_video_name(tmp_path: Path) -> None:
    video_path = tmp_path / "Smash Pro 01.mp4"
    video_path.touch()

    args = script.parse_args(
        [
            "--video-path",
            str(video_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--action-type",
            "smash",
        ]
    )
    entry = script._load_entry(args)

    assert entry.video_path == video_path
    assert entry.reference_id == "smash_pro_01"
    assert entry.selection_point_px is None


def test_load_entry_rejects_directory_video_path(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    args = script.parse_args(
        [
            "--video-path",
            str(video_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--action-type",
            "smash",
        ]
    )

    with pytest.raises(ValueError, match="must point to a file"):
        script._load_entry(args)


def test_main_builds_single_video_assets_and_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    video_path = tmp_path / "Smash Pro 01.mp4"
    _write_dummy_video(video_path)
    output_dir = tmp_path / "out_assets"

    monkeypatch.setattr(script, "_load_estimator", lambda _args: _DummyEstimator())

    script.main(
        [
            "--video-path",
            str(video_path),
            "--output-dir",
            str(output_dir),
            "--action-type",
            "smash",
            "--reference-id",
            "smash_ref_001",
            "--athlete-name",
            "athlete_a",
            "--camera-view",
            "side",
            "--handedness",
            "right",
            "--selection-point-px",
            "12.5",
            "24.0",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "assetCount": 1,
        "outputDir": str(output_dir),
    }

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["assetCount"] == 1
    assert len(metadata["assets"]) == 1
    asset = metadata["assets"][0]
    assert asset["referenceId"] == "smash_ref_001"
    assert asset["actionType"] == "smash"
    assert asset["athleteName"] == "athlete_a"
    assert asset["cameraView"] == "side"
    assert asset["handedness"] == "right"
    assert asset["selectionPointPx"] == [12.5, 24.0]
    assert Path(asset["skeletonPath"]).exists()
    assert Path(asset["renderAssetPath"]).exists()


def test_publish_assets_uploads_and_upserts_with_render_column(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skeleton_path = tmp_path / "smash_ref_001.npz"
    render_path = tmp_path / "smash_ref_001.render.npz"
    skeleton_path.write_bytes(b"skeleton")
    render_path.write_bytes(b"render")

    metadata = {
        "skeletonVersion": "sam3db_v2",
        "assets": [
            {
                "referenceId": "smash_ref_001",
                "actionType": "smash",
                "athleteName": "athlete_a",
                "cameraView": "side",
                "handedness": "right",
                "sourceVideoPath": "/tmp/source.mp4",
                "skeletonPath": str(skeleton_path),
                "renderAssetPath": str(render_path),
                "videoConfig": {"targetFps": 12.0},
                "durationSec": 1.2,
                "numFrames": 12,
                "numJoints": 33,
                "renderAssetSchemaVersion": "technique_reference_render.v1",
                "renderAssetFloatDtype": "float16",
                "renderAssetFields": ["keypoints_3d"],
            }
        ],
    }
    args = argparse.Namespace(
        cf_backend_dir=str(tmp_path),
        cf_target="remote",
        cf_env="staging",
        d1_database="",
        r2_bucket="",
        r2_prefix="refs",
        asset_base_url="https://assets.example.com",
        title_template="{action_type}-{reference_id}",
        skeleton_version="sam3db_v1",
        priority_score=100,
        is_active=1,
    )

    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path) -> str:
        commands.append(command)
        assert cwd == tmp_path
        if (
            len(command) >= 4
            and command[:4] == ["npx", "wrangler", "d1", "execute"]
            and "--json" in command
            and "PRAGMA table_info(technique_reference_assets);" in command
        ):
            return json.dumps([{"results": [{"name": "id"}, {"name": "render_asset_url"}]}])
        return ""

    monkeypatch.setattr(script, "_run_command", fake_run)

    summary = script._publish_assets(args, metadata)

    assert summary["upsertedAssetCount"] == 1
    assert summary["uploadFileCount"] == 2
    assert summary["renderAssetColumnDetected"] is True
    assert summary["cfTarget"] == "remote"
    assert summary["upsertedAssetIds"] == ["smash_smash_ref_001"]

    r2_put_commands = [
        command
        for command in commands
        if len(command) >= 5 and command[:5] == ["npx", "wrangler", "r2", "object", "put"]
    ]
    assert len(r2_put_commands) == 2
    assert r2_put_commands[0][5] == "duolian-storage-staging/refs/smash/smash_ref_001/smash_ref_001.npz"
    assert r2_put_commands[1][5] == (
        "duolian-storage-staging/refs/smash/smash_ref_001/smash_ref_001.render.npz"
    )
    assert "--remote" in r2_put_commands[0]

    upsert_command = next(
        command
        for command in commands
        if len(command) >= 4
        and command[:4] == ["npx", "wrangler", "d1", "execute"]
        and "--json" not in command
        and any("INSERT INTO technique_reference_assets" in token for token in command)
    )
    upsert_sql = next(token for token in upsert_command if "INSERT INTO technique_reference_assets" in token)
    assert "render_asset_url" in upsert_sql
    assert "https://assets.example.com/refs/smash/smash_ref_001/smash_ref_001.npz" in upsert_sql
    assert (
        "https://assets.example.com/refs/smash/smash_ref_001/smash_ref_001.render.npz"
        in upsert_sql
    )


def test_publish_assets_skips_render_column_when_db_not_migrated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skeleton_path = tmp_path / "clear_ref_001.npz"
    render_path = tmp_path / "clear_ref_001.render.npz"
    skeleton_path.write_bytes(b"skeleton")
    render_path.write_bytes(b"render")

    metadata = {
        "skeletonVersion": "sam3db_v2",
        "assets": [
            {
                "referenceId": "clear_ref_001",
                "actionType": "clear",
                "sourceVideoPath": "/tmp/source.mp4",
                "skeletonPath": str(skeleton_path),
                "renderAssetPath": str(render_path),
                "videoConfig": {"targetFps": 10.0},
                "durationSec": 1.0,
                "numFrames": 10,
                "numJoints": 33,
            }
        ],
    }
    args = argparse.Namespace(
        cf_backend_dir=str(tmp_path),
        cf_target="remote",
        cf_env="production",
        d1_database="",
        r2_bucket="",
        r2_prefix="refs",
        asset_base_url="",
        title_template="{reference_id}",
        skeleton_version="sam3db_v1",
        priority_score=100,
        is_active=1,
    )

    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path) -> str:
        commands.append(command)
        assert cwd == tmp_path
        if (
            len(command) >= 4
            and command[:4] == ["npx", "wrangler", "d1", "execute"]
            and "--json" in command
            and "PRAGMA table_info(technique_reference_assets);" in command
        ):
            return json.dumps([{"results": [{"name": "id"}, {"name": "skeleton_asset_url"}]}])
        return ""

    monkeypatch.setattr(script, "_run_command", fake_run)

    summary = script._publish_assets(args, metadata)
    assert summary["renderAssetColumnDetected"] is False

    upsert_command = next(
        command
        for command in commands
        if len(command) >= 4
        and command[:4] == ["npx", "wrangler", "d1", "execute"]
        and "--json" not in command
        and any("INSERT INTO technique_reference_assets" in token for token in command)
    )
    upsert_sql = next(token for token in upsert_command if "INSERT INTO technique_reference_assets" in token)
    assert "render_asset_url" not in upsert_sql
    assert "r2://duolian-storage/refs/clear/clear_ref_001/clear_ref_001.npz" in upsert_sql


def test_build_wrangler_scope_flags_local() -> None:
    args = argparse.Namespace(cf_target="local", cf_env="staging")
    assert script._build_wrangler_scope_flags(args) == ["--local", "--env", "staging"]
