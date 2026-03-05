from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import build_reference_assets as script


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
