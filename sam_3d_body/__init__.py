# Copyright (c) Meta Platforms, Inc. and affiliates.
__version__ = "1.0.0"

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .build_models import load_sam_3d_body, load_sam_3d_body_hf
from .video_processor import VideoExtractionConfig, extract_skeleton_sequence_from_video
from .reference_assets import (
    DEFAULT_METADATA_FILENAME,
    RENDER_ASSET_SCHEMA_VERSION,
    SUPPORTED_VIDEO_EXTENSIONS,
    ReferenceAssetBundle,
    ReferenceVideoEntry,
    build_reference_asset_bundle,
    build_reference_assets,
    build_reference_assets_metadata,
    discover_reference_videos,
    load_reference_manifest,
    save_render_asset_npz,
    save_reference_assets_metadata,
    save_skeleton_sequence_npz,
)

__all__ = [
    "__version__",
    "load_sam_3d_body",
    "load_sam_3d_body_hf",
    "SAM3DBodyEstimator",
    "VideoExtractionConfig",
    "extract_skeleton_sequence_from_video",
    "DEFAULT_METADATA_FILENAME",
    "RENDER_ASSET_SCHEMA_VERSION",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "ReferenceAssetBundle",
    "ReferenceVideoEntry",
    "build_reference_asset_bundle",
    "discover_reference_videos",
    "load_reference_manifest",
    "save_render_asset_npz",
    "save_reference_assets_metadata",
    "save_skeleton_sequence_npz",
    "build_reference_assets_metadata",
    "build_reference_assets",
]
