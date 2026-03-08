from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from sam_3d_body import (
    SAM3DBodyEstimator,
    __version__,
    load_sam_3d_body,
)
from sam_3d_body.reference_assets import (
    DEFAULT_METADATA_FILENAME,
    ReferenceVideoEntry,
    build_reference_asset_bundle,
    build_reference_assets_metadata,
    save_reference_assets_metadata,
)

from .config import ApiSettings, load_api_settings
from .models import (
    HealthResponse,
    GeneratedAssetFileModel,
    GeneratedAssetFilesModel,
    VideoInferenceCameraModel,
    VideoInferenceRequest,
    VideoInferenceResponse,
    dump_alias_model,
)


@dataclass
class ServiceState:
    settings: ApiSettings
    estimator: SAM3DBodyEstimator | Any | None = None
    estimator_load_error: str | None = None
    estimator_lock: Lock = field(default_factory=Lock)


def _build_estimator(settings: ApiSettings) -> SAM3DBodyEstimator:
    from tools.build_fov_estimator import FOVEstimator

    checkpoint_path = Path(settings.checkpoint_path)
    mhr_path = Path(settings.mhr_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not mhr_path.exists():
        raise FileNotFoundError(f"MHR model not found: {mhr_path}")
    if not settings.fov_name.strip():
        raise ValueError("SAM3DBODY_FOV_NAME must be configured")

    model, model_cfg = load_sam_3d_body(
        checkpoint_path=str(checkpoint_path),
        device=settings.device,
        mhr_path=str(mhr_path),
    )
    fov_estimator = FOVEstimator(
        name=settings.fov_name.strip(),
        device=settings.device,
        path=settings.fov_path.strip(),
    )
    return SAM3DBodyEstimator(
        model,
        model_cfg,
        fov_estimator=fov_estimator,
    )


def _map_service_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Internal server error: {exc}")


def _ensure_estimator(state: ServiceState) -> SAM3DBodyEstimator | Any:
    if state.estimator is not None:
        return state.estimator

    with state.estimator_lock:
        if state.estimator is not None:
            return state.estimator
        try:
            state.estimator = _build_estimator(state.settings)
            state.estimator_load_error = None
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            state.estimator_load_error = str(exc)
            raise HTTPException(
                status_code=503,
                detail=f"Model is unavailable: {exc}",
            ) from exc
    return state.estimator


def _resolve_asset_output_dir(
    *,
    settings: ApiSettings,
    asset_id: str,
    storage_output_dir: str | None,
    storage_prefix: str,
) -> Path:
    if storage_output_dir is not None:
        return Path(storage_output_dir).expanduser().resolve()

    artifact_root = Path(settings.artifact_root).expanduser().resolve()
    prefix = storage_prefix.strip().strip("/")
    if prefix:
        return (artifact_root / prefix / asset_id).resolve()
    return (artifact_root / asset_id).resolve()


def _build_local_file_descriptor(
    *,
    settings: ApiSettings,
    path: Path,
) -> GeneratedAssetFileModel:
    resolved_path = path.resolve()
    artifact_root = Path(settings.artifact_root).expanduser().resolve()
    relative_path: str | None = None
    fetch_url: str | None = None
    try:
        relative_path = resolved_path.relative_to(artifact_root).as_posix()
        fetch_url = f"/artifacts/{quote(relative_path, safe='/')}"
    except ValueError:
        relative_path = None
        fetch_url = None

    return GeneratedAssetFileModel(
        path=str(resolved_path),
        relativePath=relative_path,
        fetchUrl=fetch_url,
        sizeBytes=int(resolved_path.stat().st_size),
    )


def _build_asset_entry(payload: VideoInferenceRequest) -> ReferenceVideoEntry:
    selection_bbox_xyxy = None
    selection_point_px = None
    if payload.selection is not None:
        selection_bbox_xyxy = payload.selection.bbox_xyxy
        selection_point_px = payload.selection.point_px

    return ReferenceVideoEntry(
        video_path=payload.video_path,
        action_type=(payload.asset_config.action_type or "unknown"),
        reference_id=payload.asset_config.asset_id,
        camera_view=payload.asset_config.camera_view,
        handedness=payload.asset_config.handedness,
        selection_bbox_xyxy=selection_bbox_xyxy,
        selection_point_px=selection_point_px,
        video_config=payload.video_config.to_domain(),
        metadata=dict(payload.asset_config.metadata),
    )


def create_app(
    *,
    settings: ApiSettings | None = None,
    estimator: SAM3DBodyEstimator | Any | None = None,
) -> FastAPI:
    resolved_settings = settings or load_api_settings()
    service_state = ServiceState(
        settings=resolved_settings,
        estimator=estimator,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if (
            service_state.settings.eager_model_load
            and service_state.estimator is None
        ):
            try:
                _ensure_estimator(service_state)
            except HTTPException:
                # Keep process up so /health can expose model load errors.
                pass
        yield

    app = FastAPI(
        title="sam-3d-body service",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.service_state = service_state

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        state: ServiceState = app.state.service_state
        response = HealthResponse(
            status="ok",
            version=__version__,
            modelLoaded=state.estimator is not None,
            modelLoadError=state.estimator_load_error,
        )
        return dump_alias_model(response)

    @app.post("/infer/video", response_model=VideoInferenceResponse)
    def infer_video(payload: VideoInferenceRequest) -> dict[str, Any]:
        state: ServiceState = app.state.service_state
        try:
            entry = _build_asset_entry(payload)
            output_dir = _resolve_asset_output_dir(
                settings=state.settings,
                asset_id=entry.reference_id or "asset",
                storage_output_dir=payload.storage.output_dir,
                storage_prefix=payload.storage.prefix,
            )
            bundle = build_reference_asset_bundle(
                entry,
                estimator=_ensure_estimator(state),
                output_dir=output_dir,
                render_asset_float_dtype=payload.asset_config.render_float_dtype,
                render_include_masks=payload.asset_config.render_include_masks,
                overwrite=True,
            )
            manifest = build_reference_assets_metadata(
                [bundle.asset_metadata],
                skeleton_version=payload.asset_config.skeleton_version,
                fov_estimator_name=state.settings.fov_name.strip() or None,
                fov_estimator_path=state.settings.fov_path.strip() or None,
                render_asset_float_dtype=payload.asset_config.render_float_dtype,
                render_include_masks=payload.asset_config.render_include_masks,
            )
            metadata_path = output_dir / DEFAULT_METADATA_FILENAME
            save_reference_assets_metadata(manifest, metadata_path)

            timestamps = bundle.sequence.timestamps.astype(np.float32).tolist()
            first_timestamp = float(timestamps[0])
            last_timestamp = float(timestamps[-1])
            response = VideoInferenceResponse(
                assetId=entry.reference_id or "asset",
                summary={
                    "numFrames": bundle.sequence.num_frames,
                    "numJoints": bundle.sequence.num_joints,
                    "firstTimestamp": first_timestamp,
                    "lastTimestamp": last_timestamp,
                    "durationSec": (
                        float(last_timestamp - first_timestamp)
                        if bundle.sequence.num_frames > 1
                        else 0.0
                    ),
                    "sourceFps": float(bundle.asset_metadata["sourceFps"]),
                    "imageSizeHw": list(bundle.asset_metadata["imageSizeHw"]),
                    "frameIndices": list(bundle.asset_metadata["frameIndices"]),
                },
                camera=(
                    VideoInferenceCameraModel.model_validate(
                        {
                            "source": bundle.render_asset["cameraSource"],
                            "horizontalFovDeg": bundle.render_asset["horizontalFovDeg"],
                            "timestamps": bundle.render_asset["timestamps"],
                        }
                    )
                    if bundle.render_asset.get("cameraSource") is not None
                    and bundle.render_asset.get("horizontalFovDeg") is not None
                    and bundle.render_asset.get("timestamps") is not None
                    else None
                ),
                files=GeneratedAssetFilesModel(
                    skeleton=_build_local_file_descriptor(
                        settings=state.settings,
                        path=bundle.skeleton_path,
                    ),
                    render=_build_local_file_descriptor(
                        settings=state.settings,
                        path=bundle.render_path,
                    ),
                    metadata=_build_local_file_descriptor(
                        settings=state.settings,
                        path=metadata_path,
                    ),
                ),
                manifest=manifest,
            )
            return dump_alias_model(response)
        except Exception as exc:
            raise _map_service_exception(exc) from exc

    @app.get("/artifacts/{artifact_path:path}")
    def get_artifact(artifact_path: str) -> FileResponse:
        root = Path(app.state.service_state.settings.artifact_root).expanduser().resolve()
        resolved_path = (root / artifact_path).resolve()
        try:
            resolved_path.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Artifact not found.") from exc
        if not resolved_path.exists() or not resolved_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        return FileResponse(resolved_path)

    return app


app = create_app()
