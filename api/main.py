from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from sam_3d_body import (
    SAM3DBodyEstimator,
    __version__,
    build_alignment_report,
    load_sam_3d_body,
    load_skeleton_sequence_npz,
)
from sam_3d_body.technique_alignment import SkeletonSequence
from sam_3d_body.video_processor import (
    VideoExtractionConfig,
    extract_skeleton_sequence_from_video,
)

from .config import ApiSettings, load_api_settings
from .models import (
    AlignmentInferenceRequest,
    HealthResponse,
    SequenceSourceModel,
    SkeletonSequenceModel,
    VideoInferenceCameraModel,
    VideoInferenceRequest,
    VideoInferenceResponse,
    build_video_summary,
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


def _save_sequence_npz(sequence: SkeletonSequence, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    npz_payload: dict[str, Any] = {
        "keypoints_3d": sequence.keypoints_3d,
        "timestamps": sequence.timestamps,
    }
    if sequence.joint_names is not None:
        npz_payload["joint_names"] = np.asarray(sequence.joint_names, dtype=object)
    np.savez(path, **npz_payload)
    return str(path)


def _sequence_from_source(
    source: SequenceSourceModel,
    state: ServiceState,
) -> SkeletonSequence:
    if source.sequence is not None:
        return source.sequence.to_domain()
    if source.npz_path is not None:
        return load_skeleton_sequence_npz(source.npz_path)
    if source.video_path is not None:
        estimator = _ensure_estimator(state)
        video_config: VideoExtractionConfig
        if source.video_config is None:
            video_config = VideoExtractionConfig()
        else:
            video_config = source.video_config.to_domain()
        return extract_skeleton_sequence_from_video(
            video_path=source.video_path,
            estimator=estimator,
            config=video_config,
            selection_bbox_xyxy=source.selection_bbox_xyxy,
        )
    raise ValueError("Invalid sequence source.")


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
            sequence, camera_metadata = extract_skeleton_sequence_from_video(
                video_path=payload.video_path,
                estimator=_ensure_estimator(state),
                config=payload.video_config.to_domain(),
                selection_bbox_xyxy=payload.selection_bbox_xyxy,
                return_camera_metadata=True,
            )
            saved_npz_path = None
            if payload.save_npz_path is not None:
                saved_npz_path = _save_sequence_npz(sequence, payload.save_npz_path)

            response = VideoInferenceResponse(
                sequence=SkeletonSequenceModel.from_domain(sequence),
                summary=build_video_summary(sequence),
                savedNpzPath=saved_npz_path,
                camera=(
                    VideoInferenceCameraModel.model_validate(camera_metadata)
                    if camera_metadata is not None
                    else None
                ),
            )
            return dump_alias_model(response)
        except Exception as exc:
            raise _map_service_exception(exc) from exc

    @app.post("/infer/alignment")
    def infer_alignment(payload: AlignmentInferenceRequest) -> dict[str, Any]:
        state: ServiceState = app.state.service_state
        try:
            user_sequence = _sequence_from_source(payload.user, state)
            reference_sequence = _sequence_from_source(payload.reference, state)
            return build_alignment_report(
                user_sequence=user_sequence,
                reference_sequence=reference_sequence,
                config=payload.alignment_config.to_domain(),
            )
        except Exception as exc:
            raise _map_service_exception(exc) from exc

    return app


app = create_app()
