#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sam_3d_body.technique_pipeline import TechniqueAlignmentPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align two 3D skeleton sequences and output error report.",
    )
    parser.add_argument(
        "--user-npz",
        required=True,
        help="Path to user skeleton sequence npz file.",
    )
    parser.add_argument(
        "--reference-npz",
        required=True,
        help="Path to reference skeleton sequence npz file.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path for alignment report JSON.",
    )
    parser.add_argument(
        "--dtw-radius",
        type=int,
        default=5,
        help="fastdtw radius (ignored when exact_dtw fallback is used).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = TechniqueAlignmentPipeline()
    pipeline.config.alignment.dtw_radius = args.dtw_radius

    if args.output_json:
        report = pipeline.align_npz_to_json(
            user_npz=args.user_npz,
            reference_npz=args.reference_npz,
            output_json=args.output_json,
        )
    else:
        report = pipeline.align_from_npz(
            user_npz=args.user_npz,
            reference_npz=args.reference_npz,
        )

    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
