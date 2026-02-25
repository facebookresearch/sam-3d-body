from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sam_3d_body.video_processor import (
    VideoExtractionConfig,
    extract_skeleton_sequence_from_video,
)


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
            color = int((frame_idx * 17) % 255)
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class _SinglePersonEstimator:
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
        return [{"bbox": np.array([10, 10, 60, 60], dtype=np.float32), "pred_keypoints_3d": keypoints}]


class _TwoPersonEstimator:
    def process_one_image(self, _frame_rgb: np.ndarray, **_kwargs: Any) -> list[dict[str, Any]]:
        left = {
            "bbox": np.array([0, 0, 40, 40], dtype=np.float32),
            "pred_keypoints_3d": np.array(
                [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]],
                dtype=np.float32,
            ),
        }
        right = {
            "bbox": np.array([80, 0, 119, 40], dtype=np.float32),
            "pred_keypoints_3d": np.array(
                [[9.0, 0.0, 0.0], [9.1, 0.0, 0.0], [9.2, 0.0, 0.0]],
                dtype=np.float32,
            ),
        }
        return [left, right]


def test_extract_skeleton_sequence_samples_by_target_fps(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    _write_dummy_video(video_path, fps=10.0, num_frames=12)

    sequence = extract_skeleton_sequence_from_video(
        video_path=video_path,
        estimator=_SinglePersonEstimator(),  # type: ignore[arg-type]
        config=VideoExtractionConfig(
            target_fps=5.0,  # should sample every 2 frames from source fps 10
            max_frames=4,
        ),
    )

    assert sequence.num_frames == 4
    assert sequence.num_joints == 4
    assert np.allclose(sequence.timestamps, np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float32), atol=0.05)


def test_extract_skeleton_sequence_honors_selection_point(tmp_path: Path) -> None:
    video_path = tmp_path / "selection.mp4"
    _write_dummy_video(video_path, fps=8.0, num_frames=4)

    sequence = extract_skeleton_sequence_from_video(
        video_path=video_path,
        estimator=_TwoPersonEstimator(),  # type: ignore[arg-type]
        config=VideoExtractionConfig(target_fps=8.0, max_frames=2),
        selection_point_px=(100.0, 10.0),  # near right person bbox
    )

    assert sequence.num_frames == 2
    # First joint should come from the right-side person's synthetic keypoints.
    assert float(sequence.keypoints_3d[0, 0, 0]) == 9.0


def test_extract_skeleton_sequence_supports_http_video_path(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source_video = tmp_path / "remote-source.mp4"
    _write_dummy_video(source_video, fps=10.0, num_frames=8)
    source_bytes = source_video.read_bytes()

    class _FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._buffer = io.BytesIO(payload)

        def read(self, size: int = -1) -> bytes:
            return self._buffer.read(size)

        def close(self) -> None:
            self._buffer.close()

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *_args: Any) -> bool:
            self.close()
            return False

    def _fake_urlopen(url: str, timeout: int = 30) -> _FakeResponse:
        assert url == "https://example.com/test-video.mp4"
        assert timeout == 30
        return _FakeResponse(source_bytes)

    monkeypatch.setattr("sam_3d_body.video_processor.urlopen", _fake_urlopen)

    sequence = extract_skeleton_sequence_from_video(
        video_path="https://example.com/test-video.mp4",
        estimator=_SinglePersonEstimator(),  # type: ignore[arg-type]
        config=VideoExtractionConfig(
            target_fps=5.0,
            max_frames=3,
        ),
    )

    assert sequence.num_frames == 3
    assert sequence.num_joints == 4
