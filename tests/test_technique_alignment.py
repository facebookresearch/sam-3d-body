from __future__ import annotations

import numpy as np

from sam_3d_body.technique_alignment import (
    AlignmentConfig,
    NormalizationConfig,
    SkeletonSequence,
    align_sequences,
    build_alignment_report,
    normalize_skeleton_sequence,
)


def _make_sequence(frames: np.ndarray) -> SkeletonSequence:
    timestamps = np.arange(frames.shape[0], dtype=np.float32) / 10.0
    joint_names = tuple(f"joint_{idx}" for idx in range(frames.shape[1]))
    return SkeletonSequence(
        keypoints_3d=frames.astype(np.float32),
        timestamps=timestamps,
        joint_names=joint_names,
    )


def test_normalization_keeps_root_at_origin_and_aligns_hips() -> None:
    frames = np.array(
        [
            [
                [10.0, 5.0, 3.0],   # root
                [9.0, 5.0, 3.0],    # left hip
                [11.0, 5.0, 3.0],   # right hip
                [10.0, 6.0, 3.0],   # up joint
            ],
            [
                [20.0, 15.0, 8.0],  # translated + scaled
                [18.0, 15.0, 8.0],
                [22.0, 15.0, 8.0],
                [20.0, 17.0, 8.0],
            ],
        ],
        dtype=np.float32,
    )
    sequence = _make_sequence(frames)
    normalized = normalize_skeleton_sequence(
        sequence,
        NormalizationConfig(smooth_window=1),
    )

    assert normalized.shape == frames.shape
    assert np.allclose(normalized[:, 0, :], 0.0, atol=1e-5)

    # Hips should be primarily aligned with x-axis after orientation normalization.
    hips_vec = normalized[0, 2] - normalized[0, 1]
    assert hips_vec[0] > 0.5
    assert abs(hips_vec[1]) < 1e-3
    assert abs(hips_vec[2]) < 1e-3


def test_align_sequences_falls_back_to_exact_dtw() -> None:
    user_frames = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    reference_frames = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.05, 0.0, 0.0], [1.05, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    distance, path, algorithm = align_sequences(
        user_frames,
        reference_frames,
        use_fastdtw=False,
    )

    assert algorithm == "exact_dtw"
    assert distance >= 0
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)


def test_build_alignment_report_outputs_expected_sections() -> None:
    user_frames = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.2, 0.0], [3.0, 0.3, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.3, 0.0], [3.0, 0.4, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.3, 0.0], [2.0, 0.4, 0.0], [3.0, 0.5, 0.0]],
        ],
        dtype=np.float32,
    )
    reference_frames = user_frames * 1.05
    report = build_alignment_report(
        _make_sequence(user_frames),
        _make_sequence(reference_frames),
        config=AlignmentConfig(use_fastdtw=False),
    )

    assert report["algorithm"] == "exact_dtw"
    assert report["summary"]["numAlignedPairs"] >= 3
    assert len(report["alignmentPath"]) == report["summary"]["numAlignedPairs"]
    assert len(report["frameErrors"]) == report["summary"]["numAlignedPairs"]
    assert len(report["jointErrors"]) == user_frames.shape[1]
    assert report["frameErrors"][0]["phase"] in {
        "preparation",
        "acceleration",
        "contact",
        "follow_through",
    }
