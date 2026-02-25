from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from api.main import create_app
from api.models import AlignmentInferenceRequest, VideoInferenceRequest


def _write_dummy_video(path: Path, fps: float, num_frames: int) -> None:
    width, height = 120, 80
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to create test video")

    try:
        for frame_idx in range(num_frames):
            color = int((frame_idx * 19) % 255)
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class _DummyEstimator:
    def __init__(self) -> None:
        self.call_count = 0

    def process_one_image(self, _frame_rgb: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        value = float(self.call_count)
        self.call_count += 1
        keypoints = np.array(
            [
                [value, 0.0, 0.0],
                [value + 1.0, 0.0, 0.0],
                [value + 2.0, 0.0, 0.0],
                [value + 3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        return [
            {
                "bbox": np.array([10, 10, 60, 60], dtype=np.float32),
                "pred_keypoints_3d": keypoints,
            }
        ]


def _route_endpoint(app: Any, path: str, method: str = "POST") -> Any:
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise RuntimeError(f"Route not found: {method} {path}")


def test_health_endpoint_returns_service_status() -> None:
    app = create_app(estimator=_DummyEstimator())
    health_endpoint = _route_endpoint(app, "/health", method="GET")

    payload = health_endpoint()
    assert payload["status"] == "ok"
    assert payload["modelLoaded"] is True
    assert payload["modelLoadError"] is None


def test_infer_video_endpoint_returns_sequence_and_summary(tmp_path: Path) -> None:
    video_path = tmp_path / "user.mp4"
    output_npz = tmp_path / "user_sequence.npz"
    _write_dummy_video(video_path, fps=10.0, num_frames=10)

    app = create_app(estimator=_DummyEstimator())
    infer_video_endpoint = _route_endpoint(app, "/infer/video")
    request = VideoInferenceRequest.model_validate(
        {
            "videoPath": str(video_path),
            "videoConfig": {"targetFps": 5.0, "maxFrames": 3},
            "saveNpzPath": str(output_npz),
        }
    )

    payload = infer_video_endpoint(request)
    assert payload["summary"]["numFrames"] == 3
    assert payload["summary"]["numJoints"] == 4
    assert len(payload["sequence"]["timestamps"]) == 3
    assert len(payload["sequence"]["keypoints3d"]) == 3
    assert payload["savedNpzPath"] == str(output_npz)

    assert output_npz.exists()


def test_infer_alignment_endpoint_supports_mixed_sources(tmp_path: Path) -> None:
    video_path = tmp_path / "user.mp4"
    reference_npz = tmp_path / "reference.npz"
    _write_dummy_video(video_path, fps=10.0, num_frames=10)

    reference_keypoints = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            [
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )
    np.savez(
        reference_npz,
        keypoints_3d=reference_keypoints,
        timestamps=np.array([0.0, 0.2, 0.4], dtype=np.float32),
        joint_names=np.array(["joint_0", "joint_1", "joint_2", "joint_3"], dtype=object),
    )

    app = create_app(estimator=_DummyEstimator())
    infer_alignment_endpoint = _route_endpoint(app, "/infer/alignment")
    request = AlignmentInferenceRequest.model_validate(
        {
            "user": {
                "videoPath": str(video_path),
                "videoConfig": {"targetFps": 5.0, "maxFrames": 3},
            },
            "reference": {"npzPath": str(reference_npz)},
            "alignmentConfig": {"useFastdtw": False},
        }
    )

    payload = infer_alignment_endpoint(request)
    assert payload["algorithm"] == "exact_dtw"
    assert payload["summary"]["numUserFrames"] == 3
    assert payload["summary"]["numReferenceFrames"] == 3
    assert payload["summary"]["numAlignedPairs"] >= 3
