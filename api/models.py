from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sam_3d_body.video_processor import VideoExtractionConfig


def _validate_bbox_xyxy(
    bbox: tuple[float, float, float, float],
    *,
    field_name: str,
) -> None:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{field_name} must satisfy x2 > x1 and y2 > y1.")


def _validate_point_xy(
    point: tuple[float, float],
    *,
    field_name: str,
) -> None:
    x, y = (float(value) for value in point)
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError(f"{field_name} must contain finite values.")


class VideoExtractionConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    target_fps: float = Field(default=30.0, alias="targetFps", gt=0.0)
    start_time_sec: float = Field(default=0.0, alias="startTimeSec", ge=0.0)
    end_time_sec: float | None = Field(default=None, alias="endTimeSec", ge=0.0)
    max_frames: int = Field(default=240, alias="maxFrames", ge=1)
    bbox_thr: float = Field(default=0.5, alias="bboxThr", ge=0.0, le=1.0)
    use_mask: bool = Field(default=False, alias="useMask")
    inference_type: str = Field(default="body", alias="inferenceType")

    def to_domain(self) -> VideoExtractionConfig:
        return VideoExtractionConfig(
            target_fps=self.target_fps,
            start_time_sec=self.start_time_sec,
            end_time_sec=self.end_time_sec,
            max_frames=self.max_frames,
            bbox_thr=self.bbox_thr,
            use_mask=self.use_mask,
            inference_type=self.inference_type,
        )


class VideoSelectionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    bbox_xyxy: tuple[float, float, float, float] | None = Field(default=None, alias="bbox")
    point_px: tuple[float, float] | None = Field(default=None, alias="pointPx")

    @model_validator(mode="after")
    def validate_exactly_one_selector(self) -> "VideoSelectionModel":
        num_selectors = int(self.bbox_xyxy is not None) + int(self.point_px is not None)
        if num_selectors != 1:
            raise ValueError("Exactly one of bbox or pointPx must be provided.")
        if self.bbox_xyxy is not None:
            _validate_bbox_xyxy(self.bbox_xyxy, field_name="selection.bbox")
        if self.point_px is not None:
            _validate_point_xy(self.point_px, field_name="selection.pointPx")
        return self


class VideoAssetConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    asset_id: str | None = Field(default=None, alias="assetId")
    asset_role: str = Field(default="user", alias="assetRole")
    action_type: str | None = Field(default=None, alias="actionType")
    camera_view: str | None = Field(default=None, alias="cameraView")
    handedness: str | None = None
    skeleton_version: str = Field(default="sam3db_v1", alias="skeletonVersion")
    render_float_dtype: Literal["float16", "float32"] = Field(
        default="float16",
        alias="renderFloatDtype",
    )
    render_include_masks: bool = Field(default=False, alias="renderIncludeMasks")
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoAssetStorageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: Literal["local"] = "local"
    output_dir: str | None = Field(default=None, alias="outputDir")
    prefix: str = ""


class VideoInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    video_path: str = Field(alias="videoPath")
    selection: VideoSelectionModel | None = None
    video_config: VideoExtractionConfigModel = Field(
        default_factory=VideoExtractionConfigModel,
        alias="videoConfig",
    )
    asset_config: VideoAssetConfigModel = Field(
        default_factory=VideoAssetConfigModel,
        alias="assetConfig",
    )
    storage: VideoAssetStorageModel = Field(default_factory=VideoAssetStorageModel)


class VideoInferenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    num_frames: int = Field(alias="numFrames")
    num_joints: int = Field(alias="numJoints")
    first_timestamp: float = Field(alias="firstTimestamp")
    last_timestamp: float = Field(alias="lastTimestamp")
    duration_sec: float = Field(alias="durationSec")
    source_fps: float = Field(alias="sourceFps")
    image_size_hw: list[int] = Field(alias="imageSizeHw")
    frame_indices: list[int] = Field(alias="frameIndices")


class VideoInferenceCameraModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str
    horizontal_fov_deg: list[float] = Field(alias="horizontalFovDeg")
    timestamps: list[float]

    @model_validator(mode="after")
    def validate_lengths(self) -> "VideoInferenceCameraModel":
        if len(self.horizontal_fov_deg) != len(self.timestamps):
            raise ValueError("horizontalFovDeg length must match timestamps length.")
        return self


class GeneratedAssetFileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    path: str
    relative_path: str | None = Field(default=None, alias="relativePath")
    fetch_url: str | None = Field(default=None, alias="fetchUrl")
    size_bytes: int = Field(alias="sizeBytes")


class GeneratedAssetFilesModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    skeleton: GeneratedAssetFileModel
    render: GeneratedAssetFileModel
    metadata: GeneratedAssetFileModel


class VideoInferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    asset_id: str = Field(alias="assetId")
    summary: VideoInferenceSummary
    camera: VideoInferenceCameraModel | None = None
    files: GeneratedAssetFilesModel
    manifest: dict[str, Any]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    status: str
    version: str
    model_loaded: bool = Field(alias="modelLoaded")
    model_load_error: str | None = Field(default=None, alias="modelLoadError")


def dump_alias_model(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(by_alias=True)
