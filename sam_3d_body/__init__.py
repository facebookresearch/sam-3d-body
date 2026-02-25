# Copyright (c) Meta Platforms, Inc. and affiliates.
__version__ = "1.0.0"

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .build_models import load_sam_3d_body, load_sam_3d_body_hf
from .technique_alignment import (
    AlignmentConfig,
    NormalizationConfig,
    SkeletonSequence,
    build_alignment_report,
    load_skeleton_sequence_npz,
    save_alignment_report_json,
)
from .technique_pipeline import TechniqueAlignmentPipeline, TechniquePipelineConfig
from .video_processor import VideoExtractionConfig, extract_skeleton_sequence_from_video

__all__ = [
    "__version__",
    "load_sam_3d_body",
    "load_sam_3d_body_hf",
    "SAM3DBodyEstimator",
    "SkeletonSequence",
    "NormalizationConfig",
    "AlignmentConfig",
    "build_alignment_report",
    "load_skeleton_sequence_npz",
    "save_alignment_report_json",
    "VideoExtractionConfig",
    "extract_skeleton_sequence_from_video",
    "TechniquePipelineConfig",
    "TechniqueAlignmentPipeline",
]
