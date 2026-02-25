from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from fastdtw import fastdtw as _fastdtw  # type: ignore
except Exception:  # pragma: no cover - exercised via fallback tests
    _fastdtw = None

PHASE_NAMES = ("preparation", "acceleration", "contact", "follow_through")


@dataclass
class SkeletonSequence:
    keypoints_3d: np.ndarray
    timestamps: np.ndarray
    joint_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints_3d, dtype=np.float32)
        timestamps = np.asarray(self.timestamps, dtype=np.float32)

        if keypoints.ndim != 3 or keypoints.shape[2] != 3:
            raise ValueError(
                "keypoints_3d must have shape [num_frames, num_joints, 3]"
            )
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be 1D")
        if keypoints.shape[0] != timestamps.shape[0]:
            raise ValueError(
                "timestamps length must match keypoints_3d num_frames"
            )
        if keypoints.shape[0] == 0:
            raise ValueError("sequence must contain at least one frame")

        self.keypoints_3d = keypoints
        self.timestamps = timestamps
        if self.joint_names is not None and len(self.joint_names) != keypoints.shape[1]:
            raise ValueError(
                "joint_names length must match keypoints_3d num_joints"
            )

    @property
    def num_frames(self) -> int:
        return int(self.keypoints_3d.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.keypoints_3d.shape[1])


@dataclass
class NormalizationConfig:
    root_joint_index: int = 0
    left_hip_index: int = 1
    right_hip_index: int = 2
    up_joint_index: int = 3
    smooth_window: int = 3
    eps: float = 1e-6


@dataclass
class AlignmentConfig:
    dtw_radius: int = 5
    use_fastdtw: bool = True
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


def _safe_joint_index(index: int, num_joints: int) -> int:
    if num_joints <= 0:
        raise ValueError("num_joints must be > 0")
    return max(0, min(num_joints - 1, index))


def _normalize_vector(vector: np.ndarray, eps: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def _orthonormal_basis(
    x_axis_hint: np.ndarray,
    y_axis_hint: np.ndarray,
    eps: float,
) -> np.ndarray:
    x_axis = _normalize_vector(x_axis_hint, eps)
    if np.linalg.norm(x_axis) < eps:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    y_candidate = y_axis_hint - np.dot(y_axis_hint, x_axis) * x_axis
    y_axis = _normalize_vector(y_candidate, eps)
    if np.linalg.norm(y_axis) < eps:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = _normalize_vector(z_axis, eps)
    if np.linalg.norm(z_axis) < eps:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = _normalize_vector(np.cross(z_axis, x_axis), eps)

    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _smooth_sequence(sequence: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or sequence.shape[0] <= 2:
        return sequence
    radius = window // 2
    padded = np.pad(
        sequence,
        ((radius, radius), (0, 0), (0, 0)),
        mode="edge",
    )
    smoothed = np.empty_like(sequence)
    for frame_idx in range(sequence.shape[0]):
        frame_slice = padded[frame_idx : frame_idx + window]
        smoothed[frame_idx] = frame_slice.mean(axis=0)
    return smoothed


def normalize_skeleton_sequence(
    sequence: SkeletonSequence,
    config: NormalizationConfig | None = None,
) -> np.ndarray:
    config = config or NormalizationConfig()
    keypoints = np.asarray(sequence.keypoints_3d, dtype=np.float32).copy()
    num_joints = keypoints.shape[1]

    root_idx = _safe_joint_index(config.root_joint_index, num_joints)
    lhip_idx = _safe_joint_index(config.left_hip_index, num_joints)
    rhip_idx = _safe_joint_index(config.right_hip_index, num_joints)
    up_idx = _safe_joint_index(config.up_joint_index, num_joints)

    # Translation normalization: put root joint at origin in each frame.
    keypoints -= keypoints[:, [root_idx], :]

    normalized = np.empty_like(keypoints)
    for frame_idx in range(keypoints.shape[0]):
        frame = keypoints[frame_idx]
        hips_vec = frame[rhip_idx] - frame[lhip_idx]
        up_vec = frame[up_idx] - frame[root_idx]
        basis = _orthonormal_basis(hips_vec, up_vec, config.eps)
        rotated = frame @ basis

        scale_from_hips = float(np.linalg.norm(hips_vec))
        rms_scale = float(np.sqrt(np.mean(np.sum(rotated**2, axis=1))))
        scale = max(scale_from_hips, rms_scale, config.eps)
        normalized[frame_idx] = rotated / scale

    return _smooth_sequence(normalized, config.smooth_window)


def _frame_distance(user_frame: np.ndarray, reference_frame: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(user_frame - reference_frame, axis=1)))


def _exact_dtw(
    user_sequence: np.ndarray,
    reference_sequence: np.ndarray,
) -> tuple[float, list[tuple[int, int]]]:
    n_user = user_sequence.shape[0]
    n_ref = reference_sequence.shape[0]
    costs = np.full((n_user + 1, n_ref + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0

    for user_idx in range(1, n_user + 1):
        for ref_idx in range(1, n_ref + 1):
            step_cost = _frame_distance(
                user_sequence[user_idx - 1],
                reference_sequence[ref_idx - 1],
            )
            costs[user_idx, ref_idx] = step_cost + min(
                costs[user_idx - 1, ref_idx],  # insertion
                costs[user_idx, ref_idx - 1],  # deletion
                costs[user_idx - 1, ref_idx - 1],  # match
            )

    path: list[tuple[int, int]] = []
    user_idx = n_user
    ref_idx = n_ref
    while user_idx > 0 and ref_idx > 0:
        path.append((user_idx - 1, ref_idx - 1))
        prev_candidates = (
            (costs[user_idx - 1, ref_idx], user_idx - 1, ref_idx),
            (costs[user_idx, ref_idx - 1], user_idx, ref_idx - 1),
            (costs[user_idx - 1, ref_idx - 1], user_idx - 1, ref_idx - 1),
        )
        _, user_idx, ref_idx = min(prev_candidates, key=lambda item: item[0])

    while user_idx > 0:
        user_idx -= 1
        path.append((user_idx, 0))
    while ref_idx > 0:
        ref_idx -= 1
        path.append((0, ref_idx))

    path.reverse()
    return float(costs[n_user, n_ref]), path


def align_sequences(
    user_sequence: np.ndarray,
    reference_sequence: np.ndarray,
    dtw_radius: int = 5,
    use_fastdtw: bool = True,
) -> tuple[float, list[tuple[int, int]], str]:
    if user_sequence.ndim != 3 or reference_sequence.ndim != 3:
        raise ValueError("Expected user/reference sequence shape [num_frames, num_joints, 3]")
    if user_sequence.shape[1:] != reference_sequence.shape[1:]:
        raise ValueError("User/reference sequence must have matching joint shape")

    if use_fastdtw and _fastdtw is not None:
        distance, path = _fastdtw(
            user_sequence,
            reference_sequence,
            radius=dtw_radius,
            dist=_frame_distance,
        )
        return float(distance), [(int(i), int(j)) for i, j in path], "fastdtw"

    distance, path = _exact_dtw(user_sequence, reference_sequence)
    return distance, path, "exact_dtw"


def _phase_from_user_frame_index(
    user_frame_index: int,
    num_user_frames: int,
) -> str:
    if num_user_frames <= 1:
        return PHASE_NAMES[0]
    progress = user_frame_index / max(1, num_user_frames - 1)
    phase_index = min(len(PHASE_NAMES) - 1, int(progress * len(PHASE_NAMES)))
    return PHASE_NAMES[phase_index]


def _default_joint_names(num_joints: int) -> tuple[str, ...]:
    return tuple(f"joint_{joint_idx}" for joint_idx in range(num_joints))


def build_alignment_report(
    user_sequence: SkeletonSequence,
    reference_sequence: SkeletonSequence,
    config: AlignmentConfig | None = None,
) -> dict[str, Any]:
    config = config or AlignmentConfig()
    normalized_user = normalize_skeleton_sequence(user_sequence, config.normalization)
    normalized_reference = normalize_skeleton_sequence(
        reference_sequence, config.normalization
    )
    distance, path, algorithm = align_sequences(
        normalized_user,
        normalized_reference,
        dtw_radius=config.dtw_radius,
        use_fastdtw=config.use_fastdtw,
    )

    joint_names = (
        user_sequence.joint_names
        if user_sequence.joint_names is not None
        else (
            reference_sequence.joint_names
            if reference_sequence.joint_names is not None
            else _default_joint_names(user_sequence.num_joints)
        )
    )
    joint_error_sum = np.zeros(user_sequence.num_joints, dtype=np.float64)
    joint_error_max = np.zeros(user_sequence.num_joints, dtype=np.float64)
    joint_error_count = np.zeros(user_sequence.num_joints, dtype=np.int32)

    alignment_path: list[dict[str, Any]] = []
    frame_errors: list[dict[str, Any]] = []
    per_frame_mean_errors: list[float] = []
    for user_frame_index, reference_frame_index in path:
        user_frame = normalized_user[user_frame_index]
        reference_frame = normalized_reference[reference_frame_index]
        per_joint_error = np.linalg.norm(user_frame - reference_frame, axis=1)
        mean_error = float(per_joint_error.mean())
        top_joint_index = int(np.argmax(per_joint_error))

        alignment_path.append(
            {
                "userFrameIndex": int(user_frame_index),
                "referenceFrameIndex": int(reference_frame_index),
                "userTimestamp": float(user_sequence.timestamps[user_frame_index]),
                "referenceTimestamp": float(
                    reference_sequence.timestamps[reference_frame_index]
                ),
                "distance": mean_error,
            }
        )
        frame_errors.append(
            {
                "userFrameIndex": int(user_frame_index),
                "referenceFrameIndex": int(reference_frame_index),
                "userTimestamp": float(user_sequence.timestamps[user_frame_index]),
                "referenceTimestamp": float(
                    reference_sequence.timestamps[reference_frame_index]
                ),
                "phase": _phase_from_user_frame_index(
                    user_frame_index, user_sequence.num_frames
                ),
                "meanError": mean_error,
                "topJointIndex": top_joint_index,
                "topJointName": joint_names[top_joint_index],
                "topJointError": float(per_joint_error[top_joint_index]),
            }
        )

        joint_error_sum += per_joint_error
        joint_error_max = np.maximum(joint_error_max, per_joint_error)
        joint_error_count += 1
        per_frame_mean_errors.append(mean_error)

    joint_errors: list[dict[str, Any]] = []
    for joint_index in range(user_sequence.num_joints):
        count = max(1, int(joint_error_count[joint_index]))
        joint_errors.append(
            {
                "jointIndex": joint_index,
                "jointName": joint_names[joint_index],
                "meanError": float(joint_error_sum[joint_index] / count),
                "maxError": float(joint_error_max[joint_index]),
            }
        )

    p95_error = float(np.percentile(per_frame_mean_errors, 95)) if per_frame_mean_errors else 0.0
    mean_error = float(np.mean(per_frame_mean_errors)) if per_frame_mean_errors else 0.0

    return {
        "algorithm": algorithm,
        "distance": float(distance),
        "alignmentPath": alignment_path,
        "frameErrors": frame_errors,
        "jointErrors": joint_errors,
        "summary": {
            "meanFrameError": mean_error,
            "p95FrameError": p95_error,
            "numAlignedPairs": len(path),
            "numUserFrames": user_sequence.num_frames,
            "numReferenceFrames": reference_sequence.num_frames,
        },
    }


def load_skeleton_sequence_npz(
    npz_path: str | Path,
    keypoints_key: str = "keypoints_3d",
    timestamps_key: str = "timestamps",
    joint_names_key: str = "joint_names",
) -> SkeletonSequence:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Skeleton npz not found: {path}")

    data = np.load(path, allow_pickle=True)
    if keypoints_key not in data:
        raise ValueError(f"Missing key in npz: {keypoints_key}")

    keypoints = data[keypoints_key]
    if timestamps_key in data:
        timestamps = data[timestamps_key]
    else:
        timestamps = np.arange(keypoints.shape[0], dtype=np.float32)

    joint_names: tuple[str, ...] | None = None
    if joint_names_key in data:
        raw_joint_names = data[joint_names_key].tolist()
        if isinstance(raw_joint_names, (list, tuple)):
            joint_names = tuple(str(name) for name in raw_joint_names)

    return SkeletonSequence(
        keypoints_3d=keypoints,
        timestamps=timestamps,
        joint_names=joint_names,
    )


def save_alignment_report_json(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
