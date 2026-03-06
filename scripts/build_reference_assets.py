#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
_DEFAULT_CF_BACKEND_DIR = Path(__file__).resolve().parents[2] / "cf-backend"
_DEFAULT_R2_BUCKET_BY_ENV = {
    "staging": "duolian-storage-staging",
    "production": "duolian-storage",
}
_DEFAULT_D1_DATABASE_BY_ENV = {
    "staging": "duolian-db-staging",
    "production": "duolian-db",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch preprocess reference videos into core skeleton npz and render npz assets."
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
    parser.add_argument(
        "--fov-name",
        default="moge2",
        help="FOV estimator name. Use empty string to disable FOV estimation.",
    )
    parser.add_argument(
        "--fov-path",
        default="",
        help="Optional model path for FOV estimator (for moge2 checkpoint override).",
    )
    parser.add_argument(
        "--render-float-dtype",
        default="float16",
        choices=("float16", "float32"),
        help="Float dtype for render asset arrays.",
    )
    parser.add_argument(
        "--render-include-masks",
        action="store_true",
        help="Persist per-frame masks in render assets (larger files).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload generated assets to R2 and upsert technique_reference_assets rows in D1.",
    )
    parser.add_argument(
        "--cf-backend-dir",
        default=str(_DEFAULT_CF_BACKEND_DIR),
        help="Path to cf-backend project (contains wrangler.toml).",
    )
    parser.add_argument(
        "--cf-target",
        default="remote",
        choices=("local", "remote"),
        help="Cloudflare publish target. local uses local Wrangler state, remote uses Cloudflare resources.",
    )
    parser.add_argument(
        "--cf-env",
        default="staging",
        choices=("staging", "production"),
        help="Wrangler environment for publish operations (required for remote).",
    )
    parser.add_argument(
        "--d1-database",
        default="",
        help="Target D1 database name. Defaults by --cf-env.",
    )
    parser.add_argument(
        "--r2-bucket",
        default="",
        help="Target R2 bucket name. Defaults by --cf-env.",
    )
    parser.add_argument(
        "--r2-prefix",
        default="technique/reference-assets",
        help="R2 object key prefix for uploaded assets.",
    )
    parser.add_argument(
        "--asset-base-url",
        default="",
        help=(
            "Optional base URL for DB asset URLs, for example https://assets.example.com. "
            "If omitted, script stores r2://<bucket>/<objectKey>."
        ),
    )
    parser.add_argument(
        "--title-template",
        default="{action_type}_{reference_id}",
        help="Title template used for DB rows. Supports {action_type} and {reference_id}.",
    )
    parser.add_argument(
        "--priority-score",
        type=int,
        default=100,
        help="priority_score value for inserted reference assets.",
    )
    parser.add_argument(
        "--is-active",
        type=int,
        default=1,
        choices=(0, 1),
        help="is_active value for inserted reference assets.",
    )

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
    fov_estimator = None
    fov_name = _normalize_optional_str(args.fov_name)
    if fov_name is not None:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(
            name=fov_name,
            device=args.device,
            path=_normalize_optional_str(args.fov_path) or "",
        )

    return SAM3DBodyEstimator(
        model,
        model_cfg,
        fov_estimator=fov_estimator,
    )


def _run_command(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "Command failed with non-zero exit code.\n"
            f"cwd: {cwd}\n"
            f"command: {shlex.join(command)}\n"
            f"exitCode: {completed.returncode}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )
    return completed.stdout.strip()


def _to_sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return "NULL"
        if isinstance(value, int):
            return str(value)
        return repr(numeric)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _collect_column_names(value: Any) -> set[str]:
    column_names: set[str] = set()
    if isinstance(value, dict):
        candidate = value.get("name")
        if isinstance(candidate, str):
            column_names.add(candidate)
        for child in value.values():
            column_names.update(_collect_column_names(child))
        return column_names
    if isinstance(value, list):
        for child in value:
            column_names.update(_collect_column_names(child))
    return column_names


def _parse_wrangler_json_output(raw_output: str) -> Any:
    text = raw_output.strip()
    if not text:
        raise ValueError("Wrangler JSON output is empty.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidate_indexes = sorted(
        {idx for idx, char in enumerate(text) if char in ("[", "{")},
        reverse=True,
    )
    for start_idx in candidate_indexes:
        candidate = text[start_idx:].strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("Failed to locate valid JSON content in Wrangler output.")


def _resolve_publish_targets(args: argparse.Namespace) -> tuple[str, str]:
    d1_database = args.d1_database.strip() or _DEFAULT_D1_DATABASE_BY_ENV[args.cf_env]
    r2_bucket = args.r2_bucket.strip() or _DEFAULT_R2_BUCKET_BY_ENV[args.cf_env]
    return d1_database, r2_bucket


def _build_wrangler_scope_flags(args: argparse.Namespace) -> list[str]:
    scope_flags: list[str] = []
    if args.cf_target == "remote":
        scope_flags.extend(["--remote", "--env", args.cf_env])
        return scope_flags

    scope_flags.append("--local")
    if args.cf_env:
        scope_flags.extend(["--env", args.cf_env])
    return scope_flags


def _sanitize_segment(value: str) -> str:
    normalized = value.strip().replace("\\", "/").strip("/")
    normalized = normalized.replace("/", "_").replace(" ", "_")
    return normalized or "unknown"


def _build_object_key(
    *,
    prefix: str,
    action_type: str,
    reference_id: str,
    filename: str,
) -> str:
    segments = [
        prefix.strip("/"),
        _sanitize_segment(action_type),
        _sanitize_segment(reference_id),
        filename.strip("/"),
    ]
    return "/".join(segment for segment in segments if segment)


def _build_asset_url(*, bucket: str, object_key: str, asset_base_url: str) -> str:
    base = asset_base_url.strip().rstrip("/")
    if not base:
        return f"r2://{bucket}/{object_key}"
    return f"{base}/{object_key}"


def _render_title(*, template: str, action_type: str, reference_id: str) -> str:
    try:
        rendered = template.format(action_type=action_type, reference_id=reference_id)
    except KeyError as exc:
        raise ValueError(
            "--title-template only supports placeholders: {action_type}, {reference_id}"
        ) from exc
    cleaned = rendered.strip()
    if not cleaned:
        raise ValueError("Rendered title is empty; adjust --title-template.")
    return cleaned


def _fetch_table_columns(
    *,
    args: argparse.Namespace,
    cf_backend_dir: Path,
    d1_database: str,
) -> set[str]:
    command = [
        "npx",
        "wrangler",
        "d1",
        "execute",
        d1_database,
        "--json",
        "--command",
        "PRAGMA table_info(technique_reference_assets);",
        *_build_wrangler_scope_flags(args),
    ]
    output = _run_command(command, cwd=cf_backend_dir)
    payload = _parse_wrangler_json_output(output)
    columns = _collect_column_names(payload)
    if not columns:
        raise ValueError("No columns found for technique_reference_assets; check DB connection.")
    return columns


def _upsert_reference_asset_row(
    *,
    args: argparse.Namespace,
    cf_backend_dir: Path,
    d1_database: str,
    include_render_asset_url: bool,
    row: dict[str, Any],
) -> None:
    columns = [
        "id",
        "action_type",
        "title",
        "athlete_name",
        "camera_view",
        "handedness",
        "skeleton_asset_url",
        "source_video_url",
        "skeleton_version",
        "fps",
        "duration_sec",
        "num_frames",
        "num_joints",
        "priority_score",
        "is_active",
        "metadata_json",
        "created_at",
        "updated_at",
    ]
    values = [
        row["id"],
        row["action_type"],
        row["title"],
        row["athlete_name"],
        row["camera_view"],
        row["handedness"],
        row["skeleton_asset_url"],
        row["source_video_url"],
        row["skeleton_version"],
        row["fps"],
        row["duration_sec"],
        row["num_frames"],
        row["num_joints"],
        row["priority_score"],
        row["is_active"],
        row["metadata_json"],
        row["created_at"],
        row["updated_at"],
    ]
    if include_render_asset_url:
        columns.insert(7, "render_asset_url")
        values.insert(7, row.get("render_asset_url"))

    update_columns = [column for column in columns if column not in {"id", "created_at"}]
    sql = (
        "INSERT INTO technique_reference_assets "
        f"({', '.join(columns)}) "
        f"VALUES ({', '.join(_to_sql_literal(value) for value in values)}) "
        "ON CONFLICT(id) DO UPDATE SET "
        + ", ".join(f"{column}=excluded.{column}" for column in update_columns)
        + ";"
    )

    _run_command(
        [
            "npx",
            "wrangler",
            "d1",
            "execute",
            d1_database,
            "--command",
            sql,
            *_build_wrangler_scope_flags(args),
        ],
        cwd=cf_backend_dir,
    )


def _publish_assets(args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Any]:
    cf_backend_dir = Path(args.cf_backend_dir).resolve()
    if not cf_backend_dir.exists():
        raise FileNotFoundError(f"cf-backend directory not found: {cf_backend_dir}")

    d1_database, r2_bucket = _resolve_publish_targets(args)
    table_columns = _fetch_table_columns(
        args=args,
        cf_backend_dir=cf_backend_dir,
        d1_database=d1_database,
    )
    include_render_asset_url = "render_asset_url" in table_columns

    assets = metadata.get("assets")
    if not isinstance(assets, list):
        raise ValueError("Invalid metadata payload: missing assets array.")

    uploaded_file_count = 0
    published_asset_ids: list[str] = []
    now_ms = int(time.time() * 1000)

    for raw_asset in assets:
        if not isinstance(raw_asset, dict):
            raise ValueError("Invalid asset entry in metadata: expected object.")
        action_type = str(raw_asset.get("actionType") or "").strip()
        reference_id = str(raw_asset.get("referenceId") or "").strip()
        if not action_type or not reference_id:
            raise ValueError("Asset entry is missing actionType or referenceId.")

        skeleton_path = Path(str(raw_asset.get("skeletonPath") or ""))
        render_asset_path_raw = raw_asset.get("renderAssetPath")
        render_asset_path = (
            Path(str(render_asset_path_raw))
            if isinstance(render_asset_path_raw, str) and render_asset_path_raw.strip()
            else None
        )
        if not skeleton_path.exists():
            raise FileNotFoundError(f"Skeleton asset file not found: {skeleton_path}")
        if render_asset_path is not None and not render_asset_path.exists():
            raise FileNotFoundError(f"Render asset file not found: {render_asset_path}")

        skeleton_object_key = _build_object_key(
            prefix=args.r2_prefix,
            action_type=action_type,
            reference_id=reference_id,
            filename=skeleton_path.name,
        )
        _run_command(
            [
                "npx",
                "wrangler",
                "r2",
                "object",
                "put",
                f"{r2_bucket}/{skeleton_object_key}",
                "--file",
                str(skeleton_path),
                *_build_wrangler_scope_flags(args),
            ],
            cwd=cf_backend_dir,
        )
        uploaded_file_count += 1
        skeleton_asset_url = _build_asset_url(
            bucket=r2_bucket,
            object_key=skeleton_object_key,
            asset_base_url=args.asset_base_url,
        )

        render_object_key: str | None = None
        render_asset_url: str | None = None
        if render_asset_path is not None:
            render_object_key = _build_object_key(
                prefix=args.r2_prefix,
                action_type=action_type,
                reference_id=reference_id,
                filename=render_asset_path.name,
            )
            _run_command(
                [
                    "npx",
                    "wrangler",
                    "r2",
                    "object",
                    "put",
                    f"{r2_bucket}/{render_object_key}",
                    "--file",
                    str(render_asset_path),
                    *_build_wrangler_scope_flags(args),
                ],
                cwd=cf_backend_dir,
            )
            uploaded_file_count += 1
            render_asset_url = _build_asset_url(
                bucket=r2_bucket,
                object_key=render_object_key,
                asset_base_url=args.asset_base_url,
            )

        source_video_url = str(raw_asset.get("sourceVideoPath") or "").strip() or None
        row_id = f"{_sanitize_segment(action_type)}_{_sanitize_segment(reference_id)}"
        title = _render_title(
            template=args.title_template,
            action_type=action_type,
            reference_id=reference_id,
        )
        entry_metadata = {
            "sourceVideoPath": raw_asset.get("sourceVideoPath"),
            "selectionPointPx": raw_asset.get("selectionPointPx"),
            "videoConfig": raw_asset.get("videoConfig"),
            "cameraSource": raw_asset.get("cameraSource"),
            "horizontalFovDegCount": raw_asset.get("horizontalFovDegCount"),
            "horizontalFovDegRange": raw_asset.get("horizontalFovDegRange"),
            "fovEstimator": metadata.get("fovEstimator"),
            "renderAssetSchemaVersion": raw_asset.get("renderAssetSchemaVersion"),
            "renderAssetFloatDtype": raw_asset.get("renderAssetFloatDtype"),
            "renderAssetFields": raw_asset.get("renderAssetFields"),
            "uploadedObjectKeys": {
                "skeleton": skeleton_object_key,
                "render": render_object_key,
            },
        }
        _upsert_reference_asset_row(
            args=args,
            cf_backend_dir=cf_backend_dir,
            d1_database=d1_database,
            include_render_asset_url=include_render_asset_url,
            row={
                "id": row_id,
                "action_type": action_type,
                "title": title,
                "athlete_name": raw_asset.get("athleteName"),
                "camera_view": raw_asset.get("cameraView"),
                "handedness": raw_asset.get("handedness") or "unknown",
                "skeleton_asset_url": skeleton_asset_url,
                "render_asset_url": render_asset_url,
                "source_video_url": source_video_url,
                "skeleton_version": metadata.get("skeletonVersion") or args.skeleton_version,
                "fps": (
                    raw_asset.get("videoConfig", {}).get("targetFps")
                    if isinstance(raw_asset.get("videoConfig"), dict)
                    else None
                ),
                "duration_sec": raw_asset.get("durationSec"),
                "num_frames": raw_asset.get("numFrames"),
                "num_joints": raw_asset.get("numJoints"),
                "priority_score": args.priority_score,
                "is_active": args.is_active,
                "metadata_json": json.dumps(entry_metadata, ensure_ascii=False),
                "created_at": now_ms,
                "updated_at": now_ms,
            },
        )
        published_asset_ids.append(row_id)

    return {
        "enabled": True,
        "cfTarget": args.cf_target,
        "cfEnv": args.cf_env,
        "d1Database": d1_database,
        "r2Bucket": r2_bucket,
        "assetBaseUrl": args.asset_base_url.strip() or None,
        "uploadFileCount": uploaded_file_count,
        "upsertedAssetCount": len(published_asset_ids),
        "upsertedAssetIds": published_asset_ids,
        "renderAssetColumnDetected": include_render_asset_url,
    }


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
        fov_estimator_name=_normalize_optional_str(args.fov_name),
        fov_estimator_path=_normalize_optional_str(args.fov_path),
        render_asset_float_dtype=args.render_float_dtype,
        render_include_masks=args.render_include_masks,
        overwrite=args.overwrite,
    )
    output: dict[str, Any] = {
        "assetCount": metadata["assetCount"],
        "outputDir": args.output_dir,
    }
    if args.publish:
        output["publish"] = _publish_assets(args, metadata)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
