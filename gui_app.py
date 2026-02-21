# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import csv
import json
import multiprocessing as mp
import os
from datetime import datetime
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

ABBR_MAP = {
    "nose": "Nose",
    "eye": "Eye",
    "ear": "Ear",
    "shoulder": "Sh",
    "elbow": "Elb",
    "wrist": "Wr",
    "hip": "Hip",
    "knee": "Knee",
    "ankle": "Ank",
    "neck": "Neck",
    "thumb": "Th",
    "index": "Idx",
    "middle": "Mid",
    "ring": "Ring",
    "pinky": "Pky",
    "toe": "Toe",
    "heel": "Heel",
}


def _mesh_render_worker(conn, img_bgr, outputs, faces):
    """Run pyrender work in a subprocess main thread (macOS-safe for pyglet)."""
    try:
        all_depths = np.stack([tmp["pred_cam_t"] for tmp in outputs], axis=0)[:, 2]
        outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

        all_pred_vertices = []
        all_faces = []
        for pid, person_output in enumerate(outputs_sorted):
            all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
            all_faces.append(faces + len(person_output["pred_vertices"]) * pid)

        all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
        all_faces = np.concatenate(all_faces, axis=0)

        fake_pred_cam_t = (
            np.max(all_pred_vertices[-2 * 18439 :], axis=0)
            + np.min(all_pred_vertices[-2 * 18439 :], axis=0)
        ) / 2
        all_pred_vertices = all_pred_vertices - fake_pred_cam_t

        renderer = Renderer(focal_length=outputs_sorted[0]["focal_length"], faces=all_faces)

        img_mesh = (
            renderer(
                all_pred_vertices,
                fake_pred_cam_t,
                img_bgr.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_bgr) * 255
        img_mesh_side = (
            renderer(
                all_pred_vertices,
                fake_pred_cam_t,
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )

        conn.send(("ok", to_uint8(img_mesh), to_uint8(img_mesh_side)))
    except Exception as e:
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


class Sam3DGuiRunner:
    def __init__(self, checkpoint_path: str, mhr_path: str, output_dir: str = "output/gui"):
        self.checkpoint_path = checkpoint_path
        self.mhr_path = mhr_path
        self.output_dir = output_dir

        self.device = detect_device()
        self.model = None
        self.model_cfg = None
        self.estimator = None

        self.visualizer = SkeletonVisualizer(line_width=2, radius=5)
        self.visualizer.set_pose_meta(mhr70_pose_info)

    def ensure_loaded(self):
        if self.estimator is not None:
            return

        model, model_cfg = load_sam_3d_body(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            mhr_path=self.mhr_path,
        )

        self.model = model
        self.model_cfg = model_cfg
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )

    def render_keypoints(self, img_bgr: np.ndarray, outputs: list[dict[str, Any]]) -> np.ndarray:
        img_keypoints = img_bgr.copy()
        for person_output in outputs:
            keypoints_2d = person_output["pred_keypoints_2d"]
            keypoints_2d = np.concatenate(
                [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
            )
            img_keypoints = self.visualizer.draw_skeleton(img_keypoints, keypoints_2d)
            bbox = person_output["bbox"]
            img_keypoints = cv2.rectangle(
                img_keypoints,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )
        return img_keypoints

    def _keypoint_abbrev(self, idx: int) -> str:
        kp_name = mhr70_pose_info["keypoint_info"].get(idx, {}).get("name", f"kp{idx}")
        tokens = kp_name.split("_")
        prefix = ""
        if tokens and tokens[0] in ("left", "right"):
            prefix = "L" if tokens[0] == "left" else "R"
            tokens = tokens[1:]
        if not tokens:
            return f"{prefix}KP{idx}"
        head = tokens[0]
        base = ABBR_MAP.get(head, head[:3].capitalize())
        return f"{prefix}{base}"

    def render_keypoints_labeled(
        self, img_bgr: np.ndarray, outputs: list[dict[str, Any]]
    ) -> np.ndarray:
        img_labeled = self.render_keypoints(img_bgr, outputs)
        for person_output in outputs:
            keypoints_2d = person_output["pred_keypoints_2d"]
            for k_idx, pt in enumerate(keypoints_2d):
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or x >= img_labeled.shape[1] or y >= img_labeled.shape[0]:
                    continue
                text = self._keypoint_abbrev(k_idx)
                cv2.putText(
                    img_labeled,
                    text,
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img_labeled,
                    text,
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
        return img_labeled

    def render_mesh_views(
        self, img_bgr: np.ndarray, outputs: list[dict[str, Any]], faces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(outputs) == 0:
            white_img = np.ones_like(img_bgr) * 255
            return img_bgr.copy(), white_img

        all_depths = np.stack([tmp["pred_cam_t"] for tmp in outputs], axis=0)[:, 2]
        outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

        all_pred_vertices = []
        all_faces = []
        for pid, person_output in enumerate(outputs_sorted):
            all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
            all_faces.append(faces + len(person_output["pred_vertices"]) * pid)

        all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
        all_faces = np.concatenate(all_faces, axis=0)

        fake_pred_cam_t = (
            np.max(all_pred_vertices[-2 * 18439 :], axis=0)
            + np.min(all_pred_vertices[-2 * 18439 :], axis=0)
        ) / 2
        all_pred_vertices = all_pred_vertices - fake_pred_cam_t

        renderer = Renderer(focal_length=outputs_sorted[0]["focal_length"], faces=all_faces)

        img_mesh = (
            renderer(
                all_pred_vertices,
                fake_pred_cam_t,
                img_bgr.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_bgr) * 255
        img_mesh_side = (
            renderer(
                all_pred_vertices,
                fake_pred_cam_t,
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )
        return to_uint8(img_mesh), to_uint8(img_mesh_side)

    def render_mesh_views_subprocess(
        self, img_bgr: np.ndarray, outputs: list[dict[str, Any]], faces: np.ndarray, timeout_s: int = 45
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(outputs) == 0:
            white_img = np.ones_like(img_bgr) * 255
            return img_bgr.copy(), white_img

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_mesh_render_worker,
            args=(child_conn, img_bgr, outputs, faces),
            daemon=True,
        )
        proc.start()
        child_conn.close()

        try:
            if parent_conn.poll(timeout_s):
                payload = parent_conn.recv()
            else:
                proc.terminate()
                proc.join(timeout=3)
                raise TimeoutError("Mesh rendering worker timed out")
        finally:
            parent_conn.close()

        proc.join(timeout=3)

        if payload[0] == "ok":
            return payload[1], payload[2]
        raise RuntimeError(payload[1])

    def build_person_summary(self, outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary = []
        for idx, out in enumerate(outputs):
            summary.append(
                {
                    "person_id": idx,
                    "bbox": [float(v) for v in out["bbox"].tolist()],
                    "focal_length": float(out["focal_length"]),
                    "num_keypoints_2d": int(out["pred_keypoints_2d"].shape[0]),
                    "num_keypoints_3d": int(out["pred_keypoints_3d"].shape[0]),
                    "num_vertices": int(out["pred_vertices"].shape[0]),
                }
            )
        return summary

    def write_data_exports(
        self, save_dir: str, image_stem: str, outputs: list[dict[str, Any]]
    ) -> list[str]:
        export_paths: list[str] = []

        json_path = os.path.join(save_dir, f"{image_stem}_data.json")
        json_payload = {"people": []}
        for pid, out in enumerate(outputs):
            json_payload["people"].append(
                {
                    "person_id": pid,
                    "bbox": out["bbox"].tolist(),
                    "focal_length": float(out["focal_length"]),
                    "pred_cam_t": out["pred_cam_t"].tolist(),
                    "pred_keypoints_2d": out["pred_keypoints_2d"].tolist(),
                    "pred_keypoints_3d": out["pred_keypoints_3d"].tolist(),
                    "pred_vertices": out["pred_vertices"].tolist(),
                }
            )
        with open(json_path, "w") as f:
            json.dump(json_payload, f)
        export_paths.append(json_path)

        k2d_csv = os.path.join(save_dir, f"{image_stem}_keypoints2d.csv")
        with open(k2d_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["person_id", "kp_id", "x", "y"])
            for pid, out in enumerate(outputs):
                for kp_id, xy in enumerate(out["pred_keypoints_2d"]):
                    writer.writerow([pid, kp_id, float(xy[0]), float(xy[1])])
        export_paths.append(k2d_csv)

        k3d_csv = os.path.join(save_dir, f"{image_stem}_keypoints3d.csv")
        with open(k3d_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["person_id", "kp_id", "x", "y", "z"])
            for pid, out in enumerate(outputs):
                for kp_id, xyz in enumerate(out["pred_keypoints_3d"]):
                    writer.writerow([pid, kp_id, float(xyz[0]), float(xyz[1]), float(xyz[2])])
        export_paths.append(k3d_csv)

        npz_path = os.path.join(save_dir, f"{image_stem}_full.npz")
        npz_data = {}
        for pid, out in enumerate(outputs):
            prefix = f"person_{pid:03d}_"
            npz_data[prefix + "bbox"] = out["bbox"]
            npz_data[prefix + "pred_cam_t"] = out["pred_cam_t"]
            npz_data[prefix + "pred_keypoints_2d"] = out["pred_keypoints_2d"]
            npz_data[prefix + "pred_keypoints_3d"] = out["pred_keypoints_3d"]
            npz_data[prefix + "pred_vertices"] = out["pred_vertices"]
        np.savez_compressed(npz_path, **npz_data)
        export_paths.append(npz_path)

        return export_paths

    def process_files(self, files, render_mode, progress=gr.Progress(track_tqdm=False)):
        if not files:
            return [], [], [], [], [], {"error": "No files selected."}, "No files selected.", [], []

        self.ensure_loaded()
        os.makedirs(self.output_dir, exist_ok=True)

        originals = []
        keypoints_overlays = []
        keypoints_labeled = []
        mesh_overlays = []
        side_views = []
        all_metadata = []
        labeled_paths = []
        data_export_paths = []
        warnings = []

        for i, file_obj in enumerate(files):
            progress((i, len(files)), desc=f"Processing {i + 1}/{len(files)}")

            file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                all_metadata.append(
                    {
                        "file": file_path,
                        "error": "Could not read image",
                    }
                )
                continue

            outputs = self.estimator.process_one_image(file_path, bbox_thr=0.8, use_mask=False)

            originals.append((to_uint8(img_bgr), os.path.basename(file_path)))

            if len(outputs) == 0:
                keypoints_overlays.append((to_uint8(img_bgr.copy()), os.path.basename(file_path)))
                keypoints_labeled.append((to_uint8(img_bgr.copy()), os.path.basename(file_path)))
                mesh_overlays.append((to_uint8(img_bgr.copy()), os.path.basename(file_path)))
                side_views.append((np.ones_like(img_bgr) * 255, os.path.basename(file_path)))
                all_metadata.append(
                    {
                        "file": file_path,
                        "num_people": 0,
                        "message": "No people detected",
                    }
                )
                continue

            keypoints_img = self.render_keypoints(img_bgr, outputs)
            keypoints_lbl_img = self.render_keypoints_labeled(img_bgr, outputs)
            if render_mode == "Fast Pose (No Mesh)":
                mesh_img = to_uint8(img_bgr.copy())
                side_img = np.ones_like(img_bgr) * 255
            else:
                try:
                    mesh_img, side_img = self.render_mesh_views_subprocess(
                        img_bgr, outputs, self.estimator.faces
                    )
                except Exception as e:
                    # On macOS + Gradio worker threads, pyglet can fail to create
                    # an offscreen context. Keep GUI responsive and return keypoints.
                    mesh_img = to_uint8(img_bgr.copy())
                    side_img = np.ones_like(img_bgr) * 255
                    warnings.append(
                        f"{os.path.basename(file_path)}: mesh rendering skipped ({type(e).__name__}: {e})"
                    )
                    print(warnings[-1])

            keypoints_overlays.append((keypoints_img, os.path.basename(file_path)))
            keypoints_labeled.append((keypoints_lbl_img, os.path.basename(file_path)))
            mesh_overlays.append((mesh_img, os.path.basename(file_path)))
            side_views.append((side_img, os.path.basename(file_path)))

            person_summary = self.build_person_summary(outputs)
            cur_meta = {
                "file": file_path,
                "num_people": len(outputs),
                "people": person_summary,
            }
            if warnings:
                cur_meta["warnings"] = warnings[-1:]
            all_metadata.append(cur_meta)

            # Save rendered output bundle
            stem = os.path.splitext(os.path.basename(file_path))[0]
            save_dir = os.path.join(self.output_dir, stem)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, "original.jpg"), to_uint8(img_bgr))
            cv2.imwrite(os.path.join(save_dir, "keypoints.jpg"), to_uint8(keypoints_img))
            labeled_path = os.path.join(save_dir, "keypoints_labeled.jpg")
            cv2.imwrite(labeled_path, to_uint8(keypoints_lbl_img))
            cv2.imwrite(os.path.join(save_dir, "mesh_overlay.jpg"), to_uint8(mesh_img))
            cv2.imwrite(os.path.join(save_dir, "mesh_side.jpg"), to_uint8(side_img))
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(all_metadata[-1], f, indent=2)
            labeled_paths.append(labeled_path)
            data_export_paths.extend(self.write_data_exports(save_dir, stem, outputs))

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = (
            f"Done at {ts}. Device: {self.device}. Mode: {render_mode}. "
            f"Processed {len(files)} file(s)."
        )
        if warnings:
            status += " Mesh rendering skipped for some files (see Metadata warnings)."

        return (
            originals,
            keypoints_overlays,
            keypoints_labeled,
            mesh_overlays,
            side_views,
            {"runs": all_metadata},
            status,
            labeled_paths,
            data_export_paths,
        )


def build_app(checkpoint_path: str, mhr_path: str, output_dir: str):
    runner = Sam3DGuiRunner(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        output_dir=output_dir,
    )

    with gr.Blocks(title="SAM 3D Body GUI") as demo:
        gr.Markdown("## SAM 3D Body GUI\nDrag & drop or select one/more images, then process and inspect outputs in separate panels.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Input images",
                    file_count="multiple",
                    file_types=["image"],
                )
                render_mode = gr.Radio(
                    choices=["Fast Pose (No Mesh)", "Full Render (Mesh + Side View)"],
                    value="Fast Pose (No Mesh)",
                    label="Processing mode",
                    info="Use Fast Pose for fastest keypoint/data export workflow. Switch to Full Render when you need mesh visuals.",
                )
                run_btn = gr.Button("Process", variant="primary")
                status = gr.Textbox(label="Status", value=f"Ready. Device preference: {runner.device}")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Original"):
                        out_original = gr.Gallery(label="Original images", columns=2, height=350)
                    with gr.Tab("2D Keypoints"):
                        out_keypoints = gr.Gallery(label="Keypoints overlay", columns=2, height=350)
                    with gr.Tab("2D Keypoints + Labels"):
                        out_keypoints_labeled = gr.Gallery(
                            label="Labeled keypoints (click image to open large)",
                            columns=2,
                            height=500,
                        )
                    with gr.Tab("Mesh Overlay"):
                        out_mesh = gr.Gallery(label="Mesh overlay", columns=2, height=350)
                    with gr.Tab("Side View"):
                        out_side = gr.Gallery(label="Mesh side view", columns=2, height=350)
                    with gr.Tab("Metadata"):
                        out_meta = gr.JSON(label="Per-image / per-person summary")
                    with gr.Tab("Open Labeled Image"):
                        out_labeled_files = gr.File(
                            label="Open/download full-size labeled keypoint images",
                            file_count="multiple",
                        )
                    with gr.Tab("Data Exports"):
                        out_export_files = gr.File(
                            label="Download raw data (JSON/CSV/NPZ)",
                            file_count="multiple",
                        )

        run_btn.click(
            fn=runner.process_files,
            inputs=[file_input, render_mode],
            outputs=[
                out_original,
                out_keypoints,
                out_keypoints_labeled,
                out_mesh,
                out_side,
                out_meta,
                status,
                out_labeled_files,
                out_export_files,
            ],
            queue=False,
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="SAM 3D Body Gradio GUI")
    parser.add_argument(
        "--checkpoint_path",
        default="./checkpoints/sam-3d-body-dinov3/model.ckpt",
        type=str,
        help="Path to SAM 3D Body checkpoint",
    )
    parser.add_argument(
        "--mhr_path",
        default="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        type=str,
        help="Path to MHR model asset",
    )
    parser.add_argument(
        "--output_dir",
        default="./output/gui",
        type=str,
        help="Directory for GUI run outputs",
    )
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = build_app(args.checkpoint_path, args.mhr_path, args.output_dir)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
