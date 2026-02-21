# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import cv2
import gradio as gr

from sam_3d_body import render_npz_to_files, run_fast_infer

# ---------------------------------------------------------------------------
# PID file for stale-process detection
# ---------------------------------------------------------------------------
_PID_FILE = os.path.join(os.path.dirname(__file__), ".gui_pid")


def _write_pid() -> None:
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _kill_stale(port: int) -> None:
    """Best-effort kill of a previous GUI process occupying *port*."""
    try:
        out = subprocess.check_output(
            ["lsof", "-tiTCP:" + str(port), "-sTCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        for pid_str in out.splitlines():
            pid = int(pid_str)
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.3)
    except Exception:
        pass
    # Also check PID file
    if os.path.exists(_PID_FILE):
        try:
            old_pid = int(open(_PID_FILE).read().strip())
            if old_pid != os.getpid():
                os.kill(old_pid, signal.SIGTERM)
        except (ValueError, ProcessLookupError, PermissionError):
            pass


# ---------------------------------------------------------------------------
# Gradio value helpers
# ---------------------------------------------------------------------------

def _collect_paths(value: Any) -> list[str]:
    """Recursively extract filesystem paths from Gradio File component values."""
    paths: list[str] = []
    if value is None:
        return paths
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for k in ("path", "name", "file", "value"):
            if k in value:
                paths.extend(_collect_paths(value[k]))
        return paths
    if isinstance(value, (list, tuple)):
        for v in value:
            paths.extend(_collect_paths(v))
        return paths
    if hasattr(value, "name"):
        return [str(value.name)]
    return [str(value)]


def _img_tuple(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    return (img[:, :, ::-1], os.path.basename(path))


def _perf_str(summary: dict[str, Any]) -> str:
    """Format a single-image performance summary into a readable string."""
    if summary.get("cache_hit"):
        cache_ms = summary.get("cache_ms") or 0
        return f"cache_hit cache_ms={cache_ms:.1f}"
    infer_ms = summary.get("infer_ms") or 0
    save_ms = summary.get("save_ms")
    if save_ms is not None:
        return f"infer_ms={infer_ms:.1f} save_ms={save_ms:.1f}"
    return f"infer_ms={infer_ms:.1f} save=async"


# ---------------------------------------------------------------------------
# Core GUI class
# ---------------------------------------------------------------------------

class ThinGui:
    def __init__(self, checkpoint_path: str, mhr_path: str, output_dir: str):
        self.checkpoint_path = checkpoint_path
        self.mhr_path = mhr_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self, files, max_side, single_person, save_compressed, render_mesh, cache_dir, inference_type):
        if not files:
            return [], [], [], "No files selected.", []

        image_paths = [f.name if hasattr(f, "name") else str(f) for f in files]
        npz_dir = os.path.join(self.output_dir, "npz")
        mesh_dir = os.path.join(self.output_dir, "render")
        debug_log = os.path.join(self.output_dir, "debug.log")
        os.makedirs(npz_dir, exist_ok=True)
        os.makedirs(mesh_dir, exist_ok=True)

        with open(debug_log, "a") as f:
            f.write(
                f"\n[{datetime.now().isoformat()}] process start "
                f"files={len(image_paths)} render_mesh={bool(render_mesh)}\n"
            )
            for p in image_paths:
                f.write(f"  image={p}\n")

        # ---- Inference ----
        infer_type = str(inference_type).strip().lower() if inference_type else "full"
        if infer_type not in ("full", "body"):
            infer_type = "full"

        infer_summary = run_fast_infer(
            input_path=image_paths,
            output_dir=npz_dir,
            checkpoint_path=self.checkpoint_path,
            mhr_path=self.mhr_path,
            max_side=int(max_side),
            single_person=bool(single_person),
            map_2d_to_original=True,
            save_compressed=bool(save_compressed),
            cache_dir=cache_dir.strip(),
            bbox_thr=0.8,
            inference_type=infer_type,
        )
        summary_by_path = {str(entry.get("image_path")): entry for entry in infer_summary}

        npz_files: list[str] = []
        mesh_gallery: list[tuple] = []
        side_gallery: list[tuple] = []
        render_notes: list[str] = []

        for img_path in image_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            npz_path = os.path.join(npz_dir, f"{stem}.npz")

            if not os.path.exists(npz_path):
                with open(debug_log, "a") as f:
                    f.write(f"  npz_missing={npz_path}\n")
                continue

            npz_files.append(npz_path)
            perf_info = summary_by_path.get(img_path, {})
            perf = _perf_str(perf_info)

            with open(debug_log, "a") as f:
                f.write(f"  npz_ok={npz_path}\n")

            if not render_mesh:
                render_notes.append(f"{stem}: render=skipped ({perf})")
                continue

            # ---- Render via subprocess (required on macOS: pyrender needs main thread) ----
            try:
                out_dir = os.path.join(mesh_dir, stem)
                os.makedirs(out_dir, exist_ok=True)
                render_t0 = time.perf_counter()
                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "tools", "render_npz.py"),
                    "--npz", npz_path,
                    "--output_dir", out_dir,
                    "--image", img_path,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                render_ms = (time.perf_counter() - render_t0) * 1000.0

                mesh_path = os.path.join(out_dir, "mesh_overlay.jpg")
                side_path = os.path.join(out_dir, "mesh_side.jpg")

                with open(debug_log, "a") as f:
                    f.write(f"  render_cmd={' '.join(cmd)}\n")
                    f.write(f"  render_rc={proc.returncode} render_ms={render_ms:.1f}\n")
                    if proc.stderr:
                        f.write(f"  render_stderr={proc.stderr.strip()}\n")

                if proc.returncode != 0:
                    raise RuntimeError(f"render subprocess rc={proc.returncode}")
                if not (os.path.exists(mesh_path) and os.path.exists(side_path)):
                    raise RuntimeError("render outputs missing on disk")

                overlay_tuple = _img_tuple(mesh_path)
                side_tuple = _img_tuple(side_path)
                if overlay_tuple is not None:
                    mesh_gallery.append(overlay_tuple)
                if side_tuple is not None:
                    side_gallery.append(side_tuple)

                render_notes.append(
                    f"{stem}: render=ok render_ms={render_ms:.1f} ({perf})"
                )
            except Exception as e:
                with open(debug_log, "a") as f:
                    f.write(f"  render_exception={type(e).__name__}: {e}\n")
                render_notes.append(f"{stem}: render=failed ({type(e).__name__}: {e})")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = (
            f"Done {ts}. Processed {len(image_paths)} file(s). NPZ: {len(npz_files)}. "
            f"render_mesh={bool(render_mesh)}"
        )
        if render_notes:
            status += " | " + " ; ".join(render_notes)
        status += f" | debug_log={debug_log}"
        return npz_files, mesh_gallery, side_gallery, status, npz_files

    def open_viewer(self, npz_files):
        # Try to find a valid NPZ from Gradio value
        candidates = _collect_paths(npz_files)
        npz_path = ""
        for p in candidates:
            if p and os.path.exists(p):
                npz_path = p
                break

        # Fallback: most recently modified NPZ from output folder
        if not npz_path:
            npz_dir = os.path.join(self.output_dir, "npz")
            if os.path.isdir(npz_dir):
                npz_list = [
                    os.path.join(npz_dir, f)
                    for f in os.listdir(npz_dir)
                    if f.lower().endswith(".npz")
                ]
                if npz_list:
                    npz_path = max(npz_list, key=os.path.getmtime)

        if not npz_path:
            return "No NPZ file selected."

        # Preflight: trimesh interactive viewer requires pyglet<2
        check = subprocess.run(
            [sys.executable, "-c", "import trimesh.viewer.windowed; print('ok')"],
            capture_output=True, text=True,
        )
        if check.returncode != 0:
            return (
                "3D viewer dependency missing. "
                "Install with: pip install 'pyglet<2' (inside this venv). "
                f"details={check.stderr.strip() or check.stdout.strip()}"
            )

        try:
            subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "tools", "view_npz_3d.py"),
                    "--npz",
                    npz_path,
                ]
            )
            return f"Opened 3D viewer for: {os.path.basename(npz_path)}"
        except Exception as e:
            return (
                f"Failed to open viewer: {type(e).__name__}: {e}. "
                f"candidates={candidates}"
            )


# ---------------------------------------------------------------------------
# Gradio app builder
# ---------------------------------------------------------------------------

def build_app(checkpoint_path: str, mhr_path: str, output_dir: str):
    runner = ThinGui(checkpoint_path, mhr_path, output_dir)

    with gr.Blocks(title="SAM3D Thin GUI v3") as demo:
        gr.Markdown("## SAM3D Thin GUI v3\nIn-process render · Estimator caching · Thread-safe")
        gr.Markdown(
            f"**Build:** thin-gui-v3  |  **PID:** {os.getpid()}  |  "
            f"**Checkpoint:** `{os.path.basename(checkpoint_path)}`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Input JPEG/PNG", file_count="multiple", file_types=["image"])
                max_side = gr.Slider(0, 2000, value=1280, step=32, label="max_side (0 disables resize)")
                inference_type = gr.Radio(
                    choices=["full", "body"],
                    value="full",
                    label="Inference type (body=faster, no hand detail)",
                )
                single_person = gr.Checkbox(value=True, label="Single person mode")
                save_compressed = gr.Checkbox(value=False, label="Compress NPZ (slower)")
                render_mesh = gr.Checkbox(value=False, label="Also render mesh images")
                cache_dir = gr.Textbox(value="./output/fast_npz_cache", label="Cache directory (optional)")

                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    batch_preset_btn = gr.Button("⚡ Batch Preset", variant="secondary")

                status = gr.Textbox(label="Status")

            with gr.Column(scale=2):
                npz_files = gr.File(label="NPZ outputs", file_count="multiple")
                mesh_gallery = gr.Gallery(label="Mesh overlay", columns=2, height=300)
                side_gallery = gr.Gallery(label="Mesh side view", columns=2, height=300)
                open_3d_btn = gr.Button("Open interactive 3D viewer (first NPZ)")
                open_3d_status = gr.Textbox(label="3D Viewer")

        def apply_batch_preset():
            """Set controls for maximum batch throughput."""
            return (
                960,       # max_side
                "body",    # inference_type
                True,      # single_person
                False,     # save_compressed
                False,     # render_mesh
                "",        # cache_dir (skip hashing overhead)
            )

        batch_preset_btn.click(
            fn=apply_batch_preset,
            inputs=[],
            outputs=[max_side, inference_type, single_person, save_compressed, render_mesh, cache_dir],
            queue=False,
        )

        run_btn.click(
            fn=runner.process,
            inputs=[files, max_side, single_person, save_compressed, render_mesh, cache_dir, inference_type],
            outputs=[npz_files, mesh_gallery, side_gallery, status, npz_files],
            queue=False,
        )

        open_3d_btn.click(
            fn=runner.open_viewer,
            inputs=[npz_files],
            outputs=[open_3d_status],
            queue=False,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SAM3D thin GUI v3")
    p.add_argument("--checkpoint_path", default="./checkpoints/sam-3d-body-dinov3/model.ckpt", type=str)
    p.add_argument("--mhr_path", default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt", type=str)
    p.add_argument("--output_dir", default="./output/gui", type=str)
    p.add_argument("--host", default="127.0.0.1", type=str)
    p.add_argument("--port", default=7862, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _kill_stale(args.port)
    _write_pid()
    app = build_app(args.checkpoint_path, args.mhr_path, args.output_dir)
    app.launch(server_name=args.host, server_port=args.port)
