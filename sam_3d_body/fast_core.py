# Copyright (c) Meta Platforms, Inc. and affiliates.

import hashlib
import json
import os
import queue
import shutil
import threading
import time
from glob import glob
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np
import torch

from . import SAM3DBodyEstimator, load_sam_3d_body


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tiff")
_ESTIMATOR_CACHE: dict[tuple[str, str, str], SAM3DBodyEstimator] = {}
_ESTIMATOR_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def list_images(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob(os.path.join(input_path, ext)))
    return sorted(images)


def resize_keep_aspect(img_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    h, w = img_bgr.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return img_bgr, 1.0, 1.0
    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, new_w / float(w), new_h / float(h)


def pick_single_person(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(outputs) <= 1:
        return outputs
    areas = []
    for out in outputs:
        x1, y1, x2, y2 = out["bbox"]
        areas.append(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)))
    best_idx = int(np.argmax(np.asarray(areas)))
    return [outputs[best_idx]]


# ---------------------------------------------------------------------------
# NPZ payload
# ---------------------------------------------------------------------------

def to_npz_payload(
    outputs: list[dict[str, Any]],
    faces: np.ndarray,
    image_path: str,
    orig_w: int,
    orig_h: int,
    proc_w: int,
    proc_h: int,
    scale_x: float,
    scale_y: float,
    map_2d_to_original: bool,
    infer_ms: float,
) -> dict[str, np.ndarray]:
    payload = {
        "schema_version": np.asarray("sam3d.fast_npz.v2"),
        "image_path": np.asarray(image_path),
        "num_people": np.asarray(len(outputs), dtype=np.int32),
        "faces": faces.astype(np.int32),
        "orig_w": np.asarray(orig_w, dtype=np.int32),
        "orig_h": np.asarray(orig_h, dtype=np.int32),
        "proc_w": np.asarray(proc_w, dtype=np.int32),
        "proc_h": np.asarray(proc_h, dtype=np.int32),
        "scale_x": np.asarray(scale_x, dtype=np.float32),
        "scale_y": np.asarray(scale_y, dtype=np.float32),
        "infer_ms": np.asarray(infer_ms, dtype=np.float32),
    }

    for pid, out in enumerate(outputs):
        prefix = f"person_{pid:03d}_"

        bbox = out["bbox"].astype(np.float32)
        k2d = out["pred_keypoints_2d"].astype(np.float32)
        if map_2d_to_original and scale_x > 0 and scale_y > 0:
            bbox = bbox.copy()
            bbox[[0, 2]] /= scale_x
            bbox[[1, 3]] /= scale_y
            k2d = k2d.copy()
            k2d[:, 0] /= scale_x
            k2d[:, 1] /= scale_y

        payload[prefix + "bbox"] = bbox
        payload[prefix + "focal_length"] = np.asarray(out["focal_length"], dtype=np.float32)
        payload[prefix + "pred_cam_t"] = out["pred_cam_t"].astype(np.float32)
        payload[prefix + "pred_keypoints_2d"] = k2d
        payload[prefix + "pred_keypoints_3d"] = out["pred_keypoints_3d"].astype(np.float32)
        payload[prefix + "pred_vertices"] = out["pred_vertices"].astype(np.float32)

    return payload


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _file_sig(path: str) -> dict[str, Any]:
    """Content-stable file identity: path + size (no mtime — survives git ops)."""
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(st.st_size),
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _cache_key(image_path: str, context: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(_sha256_file(image_path).encode("utf-8"))
    h.update(json.dumps(context, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Estimator cache (thread-safe)
# ---------------------------------------------------------------------------

def _get_or_create_estimator(
    checkpoint_path: str,
    mhr_path: str,
) -> tuple[SAM3DBodyEstimator, str]:
    """Thread-safe estimator cache — model is loaded once per (ckpt, mhr, device)."""
    device = detect_device()
    device_str = str(device)
    key = (os.path.abspath(checkpoint_path), os.path.abspath(mhr_path), device_str)

    with _ESTIMATOR_LOCK:
        if key in _ESTIMATOR_CACHE:
            return _ESTIMATOR_CACHE[key], device_str

        print(f"Using device: {device}")
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path,
            device=device,
            mhr_path=mhr_path,
        )
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        _ESTIMATOR_CACHE[key] = estimator
        return estimator, device_str


# ---------------------------------------------------------------------------
# Background I/O helpers for batch mode
# ---------------------------------------------------------------------------

def _prefetch_image(
    image_path: str, max_side: int
) -> Optional[tuple[np.ndarray, int, int, np.ndarray, int, int, float, float]]:
    """Read and resize an image (intended for background thread)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    orig_h, orig_w = img_bgr.shape[:2]
    proc_bgr, scale_x, scale_y = resize_keep_aspect(img_bgr, max_side)
    proc_h, proc_w = proc_bgr.shape[:2]
    return (img_bgr, orig_w, orig_h, proc_bgr, proc_w, proc_h, scale_x, scale_y)


def _save_npz_worker(save_queue: queue.Queue) -> None:
    """Background thread: drain save_queue and write NPZ files."""
    while True:
        item = save_queue.get()
        if item is None:  # poison pill
            break
        out_npz, payload, compressed, cache_npz = item
        try:
            if compressed:
                np.savez_compressed(out_npz, **payload)
            else:
                np.savez(out_npz, **payload)
            if cache_npz:
                shutil.copy2(out_npz, cache_npz)
        except Exception as e:
            print(f"[save-error] {out_npz}: {e}")
        finally:
            save_queue.task_done()


# ---------------------------------------------------------------------------
# Main batch inference
# ---------------------------------------------------------------------------

def run_fast_infer(
    input_path: Union[str, list[str]],
    output_dir: str,
    checkpoint_path: str,
    mhr_path: str,
    max_side: int = 1280,
    single_person: bool = True,
    map_2d_to_original: bool = True,
    save_compressed: bool = False,
    cache_dir: str = "",
    bbox_thr: float = 0.8,
    inference_type: str = "full",
    progress_fn: Optional[Callable[[int, int, float], None]] = None,
) -> list[dict[str, Any]]:
    """
    Fast batch inference: JPEG/PNG → NPZ.

    Args:
        inference_type: "full" (body+hand decoders) or "body" (body only, faster).
        progress_fn: Optional callback(current_idx, total, elapsed_s) for progress.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(input_path, list):
        images = [str(p) for p in input_path]
    else:
        images = list_images(input_path)
    if len(images) == 0:
        raise RuntimeError(f"No images found for input: {input_path}")

    total = len(images)
    cache_enabled = len(cache_dir) > 0
    if cache_enabled:
        os.makedirs(cache_dir, exist_ok=True)

    # ---- Cache pass (only if cache enabled) ----
    cache_context: dict[str, Any] = {}
    if cache_enabled:
        cache_context = {
            "schema": "sam3d.fast_npz.v2",
            "checkpoint": _file_sig(checkpoint_path),
            "mhr": _file_sig(mhr_path),
            "max_side": int(max_side),
            "single_person": bool(single_person),
            "map_2d_to_original": bool(map_2d_to_original),
            "bbox_thr": float(bbox_thr),
            "save_compressed": bool(save_compressed),
            "inference_type": str(inference_type),
        }

    misses: list[tuple[str, str, str]] = []
    summary: list[dict[str, Any]] = []
    for image_path in images:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_npz = os.path.join(output_dir, f"{stem}.npz")
        cache_npz = ""

        if cache_enabled:
            t0 = time.perf_counter()
            key = _cache_key(image_path, cache_context)
            cache_npz = os.path.join(cache_dir, f"{key}.npz")
            if os.path.exists(cache_npz):
                shutil.copy2(cache_npz, out_npz)
                cache_ms = (time.perf_counter() - t0) * 1000.0
                print(f"[cache-hit] {os.path.basename(image_path)} -> {out_npz} | cache_ms={cache_ms:.1f}")
                summary.append(
                    {
                        "image_path": image_path,
                        "out_npz": out_npz,
                        "cache_hit": True,
                        "cache_ms": float(cache_ms),
                        "infer_ms": None,
                        "save_ms": None,
                        "num_people": None,
                    }
                )
                continue

        misses.append((image_path, out_npz, cache_npz))

    if len(misses) == 0:
        print("All inputs served from cache. Model load skipped.")
        return summary

    # ---- Load model (cached across calls) ----
    estimator, _device_str = _get_or_create_estimator(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
    )

    # ---- Background save thread ----
    save_q: queue.Queue = queue.Queue(maxsize=4)
    save_thread = threading.Thread(target=_save_npz_worker, args=(save_q,), daemon=True)
    save_thread.start()

    # ---- Prefetch first image ----
    from concurrent.futures import ThreadPoolExecutor
    prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    next_future = prefetch_pool.submit(_prefetch_image, misses[0][0], max_side)

    batch_t0 = time.perf_counter()
    miss_total = len(misses)

    for idx, (image_path, out_npz, cache_npz) in enumerate(misses):
        # ---- Get prefetched image ----
        prefetched = next_future.result()

        # ---- Submit prefetch for NEXT image ----
        if idx + 1 < miss_total:
            next_future = prefetch_pool.submit(_prefetch_image, misses[idx + 1][0], max_side)

        if prefetched is None:
            print(f"[skip] unreadable: {image_path}")
            continue

        _img_bgr, orig_w, orig_h, proc_bgr, proc_w, proc_h, scale_x, scale_y = prefetched

        # ---- Inference ----
        infer_t0 = time.perf_counter()
        if max_side > 0 and (proc_w != orig_w or proc_h != orig_h):
            proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
            outputs = estimator.process_one_image(
                proc_rgb,
                bbox_thr=bbox_thr,
                use_mask=False,
                inference_type=inference_type,
            )
        else:
            outputs = estimator.process_one_image(
                image_path,
                bbox_thr=bbox_thr,
                use_mask=False,
                inference_type=inference_type,
            )
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0

        if single_person:
            outputs = pick_single_person(outputs)

        # ---- Package payload ----
        payload = to_npz_payload(
            outputs=outputs,
            faces=estimator.faces,
            image_path=image_path,
            orig_w=orig_w,
            orig_h=orig_h,
            proc_w=proc_w,
            proc_h=proc_h,
            scale_x=scale_x,
            scale_y=scale_y,
            map_2d_to_original=map_2d_to_original,
            infer_ms=infer_ms,
        )

        # ---- Queue save to background thread ----
        save_q.put((out_npz, payload, save_compressed, cache_npz if cache_enabled else ""))

        # ---- Progress logging ----
        elapsed = time.perf_counter() - batch_t0
        done = idx + 1
        avg_s = elapsed / done
        eta_s = avg_s * (miss_total - done)
        print(
            f"[{done}/{miss_total}] {os.path.basename(image_path)} | "
            f"people={len(outputs)} infer_ms={infer_ms:.1f} "
            f"size={orig_w}x{orig_h}->{proc_w}x{proc_h} "
            f"avg={avg_s:.1f}s/img ETA={eta_s:.0f}s"
        )

        if progress_fn is not None:
            try:
                progress_fn(done, miss_total, elapsed)
            except Exception:
                pass

        summary.append(
            {
                "image_path": image_path,
                "out_npz": out_npz,
                "cache_hit": False,
                "cache_ms": None,
                "infer_ms": float(infer_ms),
                "save_ms": None,  # async save
                "num_people": int(len(outputs)),
            }
        )

    # ---- Wait for all saves to finish ----
    save_q.put(None)  # poison pill
    save_thread.join()
    prefetch_pool.shutdown(wait=False)

    total_s = time.perf_counter() - batch_t0
    print(
        f"[batch-done] {miss_total} images in {total_s:.1f}s "
        f"({total_s / max(miss_total, 1):.1f}s/img avg) "
        f"inference_type={inference_type}"
    )

    return summary
