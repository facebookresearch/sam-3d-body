# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

os.environ["OMP_NUM_THREADS"] = "1"

from typing import List, Dict
from huggingface_hub import HfApi, hf_hub_download
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def download_file(repo_id: str, repo_type: str, local_dir: str, filename: str):
    revision = None  # e.g. "main" or a commit SHA
    token = None
    local_cache_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        token=token,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    local_file = os.path.join(local_dir, filename.rsplit("/", 1)[-1])
    os.replace(local_cache_file, local_file)
    print(f"Downloaded {filename}.")


def download_files_multiprocess(
    repo_id: str,
    repo_type: str,
    local_dir: str,
    filenames: List[str],
    max_workers: int,
) -> None:
    """Parallel download without returning results."""
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                download_file,
                repo_id,
                repo_type,
                local_dir,
                filename,
            )
            for filename in filenames
        ]

        for fut in as_completed(futures):
            # We don’t store results — just catch any exceptions for logging.
            try:
                fut.result()
            except Exception as e:
                print(f"✗ Exception during download: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset to a specific folder."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Folder path where the dataset will be saved.",
    )
    parser.add_argument(
        "--max_workers",
        default=16,
        type=int,
        help="Maximum number of processes.",
    )
    parser.add_argument(
        "--splits",
        default="",
        type=str,
        help="Dataset splits that will be downloaded.",
    )

    args = parser.parse_args()
    save_dir: str = args.save_dir
    splits: str = args.splits
    max_workers: int = args.max_workers

    repo_id = "facebook/sam-3d-body-dataset"
    repo_type = "dataset"

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    split_files = defaultdict(list)
    for file in files:
        if not file.endswith("parquet"):
            continue
        _, split, _ = file.split("/")
        split_files[split].append(file)

    if splits != "":
        splits: List[str] = sorted(
            [split for split in splits.split(",") if split in split_files]
        )
    else:
        splits: List[str] = sorted(list(split_files.keys()))

    split_files: Dict[str, List[str]] = {
        split: sorted(split_files[split]) for split in splits if split in split_files
    }

    for split, filenames in split_files.items():
        local_dir = os.path.join(save_dir, split)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        download_files_multiprocess(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            filenames=filenames,
            max_workers=max_workers,
        )
        local_cache_dir = os.path.join(local_dir, "data", split)
        if os.path.exists(local_cache_dir):
            os.removedirs(local_cache_dir)


if __name__ == "__main__":
    main()
