from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sam_3d_body.technique_alignment import (
    AlignmentConfig,
    NormalizationConfig,
    SkeletonSequence,
)
from sam_3d_body.video_processor import VideoExtractionConfig


def _validate_bbox_xyxy(
    bbox: tuple[float, float, float, float],
    *,
    field_name: str,
) -> None:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{field_name} must satisfy x2 > x1 and y2 > y1.")


class VideoExtractionConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    target_fps: float = Field(default=12.0, alias="targetFps", gt=0.0)
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


class NormalizationConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    root_joint_index: int = Field(default=0, alias="rootJointIndex", ge=0)
    left_hip_index: int = Field(default=1, alias="leftHipIndex", ge=0)
    right_hip_index: int = Field(default=2, alias="rightHipIndex", ge=0)
    up_joint_index: int = Field(default=3, alias="upJointIndex", ge=0)
    smooth_window: int = Field(default=3, alias="smoothWindow", ge=1)
    eps: float = Field(default=1e-6, gt=0.0)

    def to_domain(self) -> NormalizationConfig:
        return NormalizationConfig(
            root_joint_index=self.root_joint_index,
            left_hip_index=self.left_hip_index,
            right_hip_index=self.right_hip_index,
            up_joint_index=self.up_joint_index,
            smooth_window=self.smooth_window,
            eps=self.eps,
        )


class AlignmentConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dtw_radius: int = Field(default=5, alias="dtwRadius", ge=1)
    use_fastdtw: bool = Field(default=True, alias="useFastdtw")
    normalization: NormalizationConfigModel = Field(default_factory=NormalizationConfigModel)

    def to_domain(self) -> AlignmentConfig:
        return AlignmentConfig(
            dtw_radius=self.dtw_radius,
            use_fastdtw=self.use_fastdtw,
            normalization=self.normalization.to_domain(),
        )


class SkeletonSequenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    keypoints_3d: list[list[list[float]]] = Field(alias="keypoints3d")
    timestamps: list[float]
    joint_names: list[str] | None = Field(default=None, alias="jointNames")

    def to_domain(self) -> SkeletonSequence:
        joint_names = tuple(self.joint_names) if self.joint_names is not None else None
        return SkeletonSequence(
            keypoints_3d=np.asarray(self.keypoints_3d, dtype=np.float32),
            timestamps=np.asarray(self.timestamps, dtype=np.float32),
            joint_names=joint_names,
        )

    @classmethod
    def from_domain(cls, sequence: SkeletonSequence) -> "SkeletonSequenceModel":
        return cls(
            keypoints3d=sequence.keypoints_3d.tolist(),
            timestamps=sequence.timestamps.tolist(),
            jointNames=list(sequence.joint_names) if sequence.joint_names is not None else None,
        )


class VideoInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    video_path: str = Field(alias="videoPath")
    selection_bbox_xyxy: tuple[float, float, float, float] | None = Field(
        default=None,
        alias="selectionBbox",
    )
    video_config: VideoExtractionConfigModel = Field(
        default_factory=VideoExtractionConfigModel,
        alias="videoConfig",
    )
    save_npz_path: str | None = Field(default=None, alias="saveNpzPath")

    @model_validator(mode="after")
    def validate_selection_bbox(self) -> "VideoInferenceRequest":
        if self.selection_bbox_xyxy is not None:
            _validate_bbox_xyxy(self.selection_bbox_xyxy, field_name="selectionBbox")
        return self


class VideoInferenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    num_frames: int = Field(alias="numFrames")
    num_joints: int = Field(alias="numJoints")
    first_timestamp: float = Field(alias="firstTimestamp")
    last_timestamp: float = Field(alias="lastTimestamp")


class VideoInferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sequence: SkeletonSequenceModel
    summary: VideoInferenceSummary
    saved_npz_path: str | None = Field(default=None, alias="savedNpzPath")


class SequenceSourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    npz_path: str | None = Field(default=None, alias="npzPath")
    sequence: SkeletonSequenceModel | None = None
    video_path: str | None = Field(default=None, alias="videoPath")
    selection_bbox_xyxy: tuple[float, float, float, float] | None = Field(
        default=None,
        alias="selectionBbox",
    )
    video_config: VideoExtractionConfigModel | None = Field(default=None, alias="videoConfig")

    @model_validator(mode="after")
    def validate_exactly_one_source(self) -> "SequenceSourceModel":
        num_sources = int(self.npz_path is not None)
        num_sources += int(self.sequence is not None)
        num_sources += int(self.video_path is not None)
        if num_sources != 1:
            raise ValueError("Exactly one of npzPath, sequence, videoPath must be provided.")

        if self.video_path is None and self.selection_bbox_xyxy is not None:
            raise ValueError("selectionBbox is only valid when videoPath is provided.")
        if self.video_path is None and self.video_config is not None:
            raise ValueError("videoConfig is only valid when videoPath is provided.")
        if self.selection_bbox_xyxy is not None:
            _validate_bbox_xyxy(self.selection_bbox_xyxy, field_name="selectionBbox")

        return self


class AlignmentInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    user: SequenceSourceModel
    reference: SequenceSourceModel
    alignment_config: AlignmentConfigModel = Field(
        default_factory=AlignmentConfigModel,
        alias="alignmentConfig",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    status: str
    version: str
    model_loaded: bool = Field(alias="modelLoaded")
    model_load_error: str | None = Field(default=None, alias="modelLoadError")


def build_video_summary(sequence: SkeletonSequence) -> VideoInferenceSummary:
    return VideoInferenceSummary(
        numFrames=sequence.num_frames,
        numJoints=sequence.num_joints,
        firstTimestamp=float(sequence.timestamps[0]),
        lastTimestamp=float(sequence.timestamps[-1]),
    )


def dump_alias_model(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(by_alias=True)
