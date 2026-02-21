#!/usr/bin/env python3

import argparse

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from sam_3d_body import open_npz_interactive_viewer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open interactive 3D viewer for SAM3D NPZ")
    p.add_argument("--npz", required=True, type=str, help="Path to NPZ file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    open_npz_interactive_viewer(args.npz)


if __name__ == "__main__":
    main()
