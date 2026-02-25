from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .technique_alignment import (
    AlignmentConfig,
    SkeletonSequence,
    build_alignment_report,
    load_skeleton_sequence_npz,
    save_alignment_report_json,
)
from .video_processor import VideoExtractionConfig, extract_skeleton_sequence_from_video


@dataclass
class TechniquePipelineConfig:
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    user_video: VideoExtractionConfig = field(default_factory=VideoExtractionConfig)
    reference_video: VideoExtractionConfig = field(default_factory=VideoExtractionConfig)


class TechniqueAlignmentPipeline:
    def __init__(
        self,
        config: TechniquePipelineConfig | None = None,
    ) -> None:
        self.config = config or TechniquePipelineConfig()

    def align_from_sequences(
        self,
        user_sequence: SkeletonSequence,
        reference_sequence: SkeletonSequence,
    ) -> dict[str, Any]:
        return build_alignment_report(
            user_sequence=user_sequence,
            reference_sequence=reference_sequence,
            config=self.config.alignment,
        )

    def align_from_npz(
        self,
        user_npz: str | Path,
        reference_npz: str | Path,
    ) -> dict[str, Any]:
        user_sequence = load_skeleton_sequence_npz(user_npz)
        reference_sequence = load_skeleton_sequence_npz(reference_npz)
        return self.align_from_sequences(user_sequence, reference_sequence)

    def align_from_videos(
        self,
        estimator: SAM3DBodyEstimator,
        user_video_path: str | Path,
        reference_video_path: str | Path,
        user_selection_point_px: tuple[float, float] | None = None,
        reference_selection_point_px: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        user_sequence = extract_skeleton_sequence_from_video(
            user_video_path,
            estimator,
            config=self.config.user_video,
            selection_point_px=user_selection_point_px,
        )
        reference_sequence = extract_skeleton_sequence_from_video(
            reference_video_path,
            estimator,
            config=self.config.reference_video,
            selection_point_px=reference_selection_point_px,
        )
        return self.align_from_sequences(user_sequence, reference_sequence)

    def align_npz_to_json(
        self,
        user_npz: str | Path,
        reference_npz: str | Path,
        output_json: str | Path,
    ) -> dict[str, Any]:
        report = self.align_from_npz(user_npz=user_npz, reference_npz=reference_npz)
        save_alignment_report_json(report, output_json)
        return report
