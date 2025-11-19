# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def visualize_sample(img_cv2, outputs, faces):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_mesh.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img3 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img

def visualize_sample_together(img_cv2, outputs, faces):
    # Render everything together
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints.
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Then, put all meshes together as one super mesh
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    # Render front view
    renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )

    # Render side view
    white_img = np.ones_like(img_cv2) * 255
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

    cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)

    return cur_img
