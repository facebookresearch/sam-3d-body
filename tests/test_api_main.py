from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.config import ApiSettings
from api.main import create_app
from api.models import VideoInferenceRequest


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
                "cam_intrinsics": np.array(
                    [[100.0, 0.0, 60.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                "camera_source": "moge2",
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


def test_infer_video_endpoint_returns_asset_manifest_and_local_files(tmp_path: Path) -> None:
    video_path = tmp_path / "user.mp4"
    _write_dummy_video(video_path, fps=10.0, num_frames=10)
    artifact_root = tmp_path / "artifacts"

    app = create_app(
        estimator=_DummyEstimator(),
        settings=ApiSettings(
            checkpoint_path="/tmp/model.ckpt",
            mhr_path="/tmp/mhr_model.pt",
            device="cpu",
            fov_name="moge2",
            fov_path="",
            artifact_root=str(artifact_root),
        ),
    )
    infer_video_endpoint = _route_endpoint(app, "/infer/video")
    request = VideoInferenceRequest.model_validate(
        {
            "videoPath": str(video_path),
            "selection": {"bbox": [10, 20, 60, 70]},
            "videoConfig": {"targetFps": 5.0, "maxFrames": 3},
            "assetConfig": {
                "assetId": "user_bundle",
                "actionType": "smash",
                "handedness": "right",
            },
            "storage": {
                "mode": "local",
                "prefix": "unit-tests",
            },
        }
    )

    payload = infer_video_endpoint(request)
    assert payload["assetId"] == "user_bundle"
    assert payload["summary"]["numFrames"] == 3
    assert payload["summary"]["numJoints"] == 4
    assert payload["summary"]["sourceFps"] == pytest.approx(10.0)
    assert payload["summary"]["frameIndices"] == [0, 2, 4]
    assert payload["camera"]["source"] == "moge2"
    assert len(payload["camera"]["horizontalFovDeg"]) == 3
    assert len(payload["camera"]["timestamps"]) == 3
    assert payload["manifest"]["assetCount"] == 1
    assert payload["manifest"]["assets"][0]["selectionBbox"] == [10.0, 20.0, 60.0, 70.0]

    skeleton_path = Path(payload["files"]["skeleton"]["path"])
    render_path = Path(payload["files"]["render"]["path"])
    metadata_path = Path(payload["files"]["metadata"]["path"])
    assert skeleton_path.exists()
    assert render_path.exists()
    assert metadata_path.exists()
    assert payload["files"]["skeleton"]["fetchUrl"] is not None
    assert payload["files"]["render"]["fetchUrl"] is not None
    assert payload["files"]["metadata"]["fetchUrl"] is not None

    client = TestClient(app)
    render_response = client.get(payload["files"]["render"]["fetchUrl"])
    assert render_response.status_code == 200
    assert len(render_response.content) > 0


def test_video_inference_request_uses_selection_bbox() -> None:
    payload = VideoInferenceRequest.model_validate(
        {
            "videoPath": "/tmp/video.mp4",
            "selection": {"bbox": [10, 20, 30, 40]},
        }
    )
    assert payload.selection is not None
    assert payload.selection.bbox_xyxy == (10.0, 20.0, 30.0, 40.0)


def test_video_inference_request_defaults_target_fps_to_30() -> None:
    payload = VideoInferenceRequest.model_validate({"videoPath": "/tmp/video.mp4"})

    assert payload.video_config.target_fps == 30.0


def test_video_inference_request_rejects_legacy_fields() -> None:
    with pytest.raises(ValidationError):
        VideoInferenceRequest.model_validate(
            {
                "videoPath": "/tmp/video.mp4",
                "jointNames": ["a", "b", "c"],
            }
        )

    with pytest.raises(ValidationError):
        VideoInferenceRequest.model_validate(
            {
                "videoPath": "/tmp/video.mp4",
                "selectionBbox": [10, 20, 30, 40],
            }
        )

    with pytest.raises(ValidationError):
        VideoInferenceRequest.model_validate(
            {
                "videoPath": "/tmp/video.mp4",
                "saveNpzPath": "/tmp/out.npz",
            }
        )
