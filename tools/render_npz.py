#!/usr/bin/env python3

import argparse

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from sam_3d_body import render_npz_to_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render SAM3D NPZ to mesh images")
    p.add_argument("--npz", required=True, type=str, help="Path to NPZ")
    p.add_argument("--output_dir", required=True, type=str, help="Output folder")
    p.add_argument("--image", default="", type=str, help="Optional background image path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mesh_path, side_path = render_npz_to_files(
        npz_path=args.npz,
        output_dir=args.output_dir,
        image_path=args.image if args.image else None,
    )
    print(f"mesh={mesh_path}")
    print(f"side={side_path}")


if __name__ == "__main__":
    main()
