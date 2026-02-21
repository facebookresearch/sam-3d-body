# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import trimesh

from .visualization.renderer import Renderer


def _load_people(npz_path: str) -> tuple[np.ndarray, list[dict[str, Any]], int, int]:
    z = np.load(npz_path, allow_pickle=False)
    num_people = int(z["num_people"])
    faces = z["faces"].astype(np.int32)
    proc_w = int(z["proc_w"])
    proc_h = int(z["proc_h"])

    people = []
    for pid in range(num_people):
        p = f"person_{pid:03d}_"
        people.append(
            {
                "pred_vertices": z[p + "pred_vertices"].astype(np.float32),
                "pred_cam_t": z[p + "pred_cam_t"].astype(np.float32),
                "focal_length": float(z[p + "focal_length"]),
            }
        )
    return faces, people, proc_w, proc_h


def _build_scene_mesh(people: list[dict[str, Any]], faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(people) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32), np.zeros(3, dtype=np.float32), 1000.0

    all_depths = np.stack([p["pred_cam_t"] for p in people], axis=0)[:, 2]
    people_sorted = [people[idx] for idx in np.argsort(-all_depths)]

    verts_list = []
    faces_list = []
    offset = 0
    for p in people_sorted:
        v = p["pred_vertices"] + p["pred_cam_t"]
        verts_list.append(v)
        faces_list.append(faces + offset)
        offset += v.shape[0]

    verts = np.concatenate(verts_list, axis=0)
    tri = np.concatenate(faces_list, axis=0)
    fake_cam_t = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts = verts - fake_cam_t
    focal = float(people_sorted[0]["focal_length"])
    return verts.astype(np.float32), tri.astype(np.int32), fake_cam_t.astype(np.float32), focal


def render_npz_views(
    npz_path: str,
    image_bgr: np.ndarray | None = None,
    mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
    bg_color=(1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    faces, people, proc_w, proc_h = _load_people(npz_path)
    if image_bgr is None:
        image_bgr = np.ones((proc_h, proc_w, 3), dtype=np.uint8) * 255
    else:
        # Camera params (focal, translation) were computed at proc_w x proc_h.
        # Resize background to match so the mesh overlay aligns with the figure.
        img_h, img_w = image_bgr.shape[:2]
        if img_w != proc_w or img_h != proc_h:
            image_bgr = cv2.resize(image_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    if len(people) == 0:
        return image_bgr.copy(), np.ones_like(image_bgr) * 255

    verts, tri, fake_cam_t, focal = _build_scene_mesh(people, faces)
    renderer = Renderer(focal_length=focal, faces=tri)

    overlay = (
        renderer(
            verts,
            fake_cam_t,
            image_bgr.copy(),
            mesh_base_color=mesh_base_color,
            scene_bg_color=bg_color,
        )
        * 255
    ).astype(np.uint8)

    white = np.ones_like(image_bgr) * 255
    side = (
        renderer(
            verts,
            fake_cam_t,
            white,
            mesh_base_color=mesh_base_color,
            scene_bg_color=bg_color,
            side_view=True,
        )
        * 255
    ).astype(np.uint8)

    return overlay, side


def open_npz_interactive_viewer(npz_path: str) -> None:
    faces, people, _proc_w, _proc_h = _load_people(npz_path)
    if len(people) == 0:
        raise RuntimeError("No people in NPZ")
    verts, tri, _fake_cam_t, _focal = _build_scene_mesh(people, faces)

    mesh = trimesh.Trimesh(vertices=verts, faces=tri, process=False)
    scene = trimesh.Scene(mesh)
    scene.show()


def render_npz_to_files(npz_path: str, output_dir: str, image_path: str | None = None) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path) if image_path else None
    mesh, side = render_npz_views(npz_path=npz_path, image_bgr=img)
    mesh_path = os.path.join(output_dir, "mesh_overlay.jpg")
    side_path = os.path.join(output_dir, "mesh_side.jpg")
    cv2.imwrite(mesh_path, mesh)
    cv2.imwrite(side_path, side)
    return mesh_path, side_path
