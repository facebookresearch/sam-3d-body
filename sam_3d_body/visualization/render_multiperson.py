"""Wrapper function to visualize multiple persons in a single image using the 
existing mesh renderer
"""

import numpy as np
import cv2
import torch
import trimesh
from sam_3d_body.utils.misc import to_numpy
from sam_3d_body.visualization.utils import draw_box

from sam_3d_body.visualization.renderer import Renderer
from typing import Dict, List, Optional

LIGHT_BLUE = (0.65098039,  0.74117647,  0.85882353)


def render_multiple(
    img: np.ndarray,
    pred_vertices: List[torch.Tensor],
    focal_length: torch.Tensor,
    pred_cam_t: List[torch.Tensor],
    faces,
    metrics_str: Optional[str] = None,
    img_num: Optional[str] = None,
    bboxes: Optional[Dict] = None,
):

    renderer = Renderer(focal_length=focal_length, faces=faces)

    focal_length = focal_length.cpu().numpy()
    img_cv2 = img.copy()

    # Set the render resolution to match the original image dimensions.
    render_res = [img_cv2.shape[1], img_cv2.shape[0]]

    # get zs to plot before sorting
    zs = [f"{v[2]:.2f}" for v in pred_cam_t]

    sorted_pairs = sorted(zip(pred_cam_t, pred_vertices),
                          key=lambda x: x[0][2], reverse=False)
    pred_cam_t_sorted, pred_vertices_sorted = zip(*sorted_pairs)
    # pred_cam_t --> torch.stack([tx + cx, ty + cy, tz], dim=-1)
    pred_cam_t = np.stack(to_numpy(list(pred_cam_t_sorted)))
    pred_vertices = np.stack(to_numpy(list(pred_vertices_sorted)))

    # center scene
    ############################
    points = trimesh.PointCloud(pred_cam_t[:, :3])
    bbox = points.bounding_box
    central_pt = bbox.centroid
    # use the focal length and bbox depth to calculate the zoom out factor
    # TODO: improve this
    bbox_diff_center = (bbox.vertices.max(0) - bbox.vertices.min(0)) / 2
    margin = 5.
    zo_h = (2 * bbox_diff_center[2] * focal_length / render_res[1]) + margin
    zo_w = 2 * bbox_diff_center[2] * focal_length / render_res[0] + margin
    # translate to origin
    pred_cam_t_cent = pred_cam_t - central_pt
    ############################

    trimesh_meshes = []
    for n, v in enumerate(pred_vertices):
        tmesh = renderer.vertices_to_trimesh(v, pred_cam_t_cent[n], LIGHT_BLUE)
        trimesh_meshes.append(tmesh)

    # Render the composite RGBA image with all meshes overlayed,
    composite_rgba = renderer.render_rgba_multiple(
        vertices=pred_vertices,
        cam_t=pred_cam_t,
        scene_bg_color=(1, 1, 1),
        render_res=render_res,
    )

    # for BEV view
    rot = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    transl = trimesh.transformations.translation_matrix([0, 0, zo_h])
    transform = transl @ rot
    pred_vertices_bev = [m.copy().apply_transform(
        transform).vertices for m in trimesh_meshes]
    composite_rgba_bev = renderer.render_rgba_multiple(
        vertices=pred_vertices_bev,
        cam_t=[np.array([0., 0., 0.])] * len(pred_vertices_bev),
        scene_bg_color=(1, 1, 1),
        render_res=render_res,
    )

    # for side view
    rot = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [0, 0, 1]) @ rot
    transl = trimesh.transformations.translation_matrix([0, 0, zo_w])
    transform = transl @ rot
    pred_vertices_side = [m.copy().apply_transform(
        transform).vertices for m in trimesh_meshes]
    composite_rgba_side = renderer.render_rgba_multiple(
        vertices=pred_vertices_side,
        cam_t=[np.array([0., 0., 0.])] * len(pred_vertices_bev),
        scene_bg_color=(1, 1, 1),
        render_res=render_res,
    )

    # draw bboxes
    if bboxes is not None:
        original_image = draw_annot_box(img_cv2.copy(),
                                        box_centers=bboxes["orig_center"],
                                        box_scales=bboxes["orig_scale"]
                                        )

        overlay_image = draw_annot_box(img_cv2.copy(),
                                       box_centers=bboxes["center"],
                                       box_scales=bboxes["scale"]
                                       )
    else:
        original_image = img_cv2.copy()
        overlay_image = img_cv2.copy()

    # Create a variable for the original image.
    overlay_image = overlay_image.copy() / 255.

    # Create a variable for the mesh overlay with a transparent background.
    color = composite_rgba.copy()
    valid_mask = (color[:, :, -1])[:, :, np.newaxis]
    output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * overlay_image)
    output_img = (output_img * 255).astype(np.uint8)
    
    composite_rgba_bev = (composite_rgba_bev[:, :, :3] * 255).astype(np.uint8)
    composite_rgba_side = (composite_rgba_side[:, :, :3] * 255).astype(np.uint8)

    # add tittle to the bev and side views
    add_text_to_image(composite_rgba_side, "Side View",
                      position=(10, 35), font_scale=1.2)
    add_text_to_image(composite_rgba_bev, "Bird's Eye View",
                      position=(10, 35), font_scale=1.2)
    add_text_to_image(composite_rgba_bev, f"focal length: {focal_length:.2f}", position=(10, 75), font_scale=1.2)
    add_text_to_image(composite_rgba_bev, f"zs: {zs}", position=(10, 125), font_scale=1.2)
    
    # show metrics on the image
    if metrics_str is not None:
        for n_str, metric in enumerate(metrics_str):
            add_text_to_image(composite_rgba_bev, metric, position=(
                10, 125 + 50 * (n_str + 1)), font_scale=1.2)

    # cv2.imwrite("/private/home/nugrinovic/code/3po/debug/mupots_pcod.png", composite_rgba_bev)
    if img_num is not None:
        img_num = img_num.split('-')[-1]
        cv2.putText(composite_rgba_side,
                    img_num,
                    (composite_rgba_side.shape[1] - 400,
                     composite_rgba_side.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA)

    # add a black stripe
    black_stripe = np.zeros(
        (original_image.shape[0], 5, 3), dtype=np.uint8)  # 10 pixels wide

    # Display the original and rendered images side by side
    cur_img = np.concatenate([original_image,
                              output_img,
                              black_stripe, composite_rgba_bev,
                              black_stripe, composite_rgba_side],
                             axis=1
                             )

    return cur_img


def add_text_to_image(
    image,
    text,
    position=(10, 25),
    font_scale=0.8,
    color=(0, 0, 0),
    thickness=2,
    add_box=False
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # add a box around the text
    if add_box:
        text_size = cv2.getTextSize(text, font, 0.30, 1)[0]
        cv2.rectangle(image, (10, 10 - text_size[1]),
                      (10 + text_size[0], 10 + 5),
                      (100, 100, 100),
                      -1)
    cv2.putText(image, text, position, font, font_scale,
                color, thickness, cv2.LINE_AA)


def draw_annot_box(img, box_centers, box_scales):
    for person_num, (center, scale) in enumerate(zip(box_centers, box_scales)):
        cx, cy = center
        sx, sy = scale
        pt1 = [int(cx - sx / 2),
               int(cy - sy / 2)]
        pt2 = [int(cx + sx / 2),
               int(cy + sy / 2)]
        # draw box
        img = draw_box(img, pt1 + pt2,
                       text=str(
                           person_num),
                       font_scale=2.,
                       font_thickness=3)
    return img
