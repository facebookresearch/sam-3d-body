#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start sam-3d-body FastAPI server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--fov-name",
        default=os.getenv("SAM3DBODY_FOV_NAME", "moge2"),
        help="FOV estimator name. Default: moge2",
    )
    parser.add_argument(
        "--fov-path",
        default=os.getenv("SAM3DBODY_FOV_PATH", ""),
        help="Optional path or HF repo id for FOV estimator weights.",
    )
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    args = _parse_args()
    os.environ["SAM3DBODY_FOV_NAME"] = args.fov_name
    os.environ["SAM3DBODY_FOV_PATH"] = args.fov_path
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
