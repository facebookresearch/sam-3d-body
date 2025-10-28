import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import cv2
import numpy as np
import pyrender
import torch
import trimesh
from sam_3d_body.visualization.renderer import Renderer
from torchvision.utils import make_grid

from .render_atlas import render_atlas


def create_raymond_lights():
    import pyrender

    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )

    return nodes


class MeshRenderer:

    def __init__(self, focal_length, img_res, faces=None):
        self.focal_length = focal_length
        self.img_res = img_res
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.img_res, viewport_height=self.img_res, point_size=1.0
        )

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces

    def visualize(
        self, vertices, camera_translation, images, focal_length=None, nrow=3, padding=2
    ):
        images_np = np.transpose(images, (0, 2, 3, 1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            fl = self.focal_length
            rend_img = torch.from_numpy(
                np.transpose(
                    self.__call__(
                        vertices[i],
                        camera_translation[i],
                        images_np[i],
                        focal_length=fl,
                        side_view=False,
                    ),
                    (2, 0, 1),
                )
            ).float()
            rend_img_side = torch.from_numpy(
                np.transpose(
                    self.__call__(
                        vertices[i],
                        camera_translation[i],
                        images_np[i],
                        focal_length=fl,
                        side_view=True,
                    ),
                    (2, 0, 1),
                )
            ).float()
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def visualize_gt_mesh(
        self, gt_verts, full_cam_t, image_batch, focal_length, camera_center
    ):
        rend_imgs = []
        images_np = np.transpose(image_batch, (0, 2, 3, 1))
        for i in range(gt_verts.shape[0]):
            img_mesh_crop_i = self.render_mesh(
                gt_verts[i],
                full_cam_t[i],
                255 * images_np[i],
                focal_length=focal_length[i],
                camera_center=camera_center[i],
                mesh_color=(0.8, 0.2, 0),
            )
            rend_imgs.append(
                torch.from_numpy(np.transpose(img_mesh_crop_i, (2, 0, 1))).float()
            )
        rend_imgs = make_grid(rend_imgs, nrow=1, padding=2)
        return rend_imgs

    def visualize_tensorboard(
        self,
        vertices,
        camera_translation,
        images,
        pred_keypoints,
        gt_mesh_img,
        gt_keypoints,
        focal_length=None,
        nrow=7,
        padding=2,
        pred_v2d=None,
        camera_center=None,
        masks=None,
    ):
        assert camera_center is not None

        images_np = np.transpose(images, (0, 2, 3, 1))
        rend_imgs = []
        pred_keypoints = np.concatenate(
            (pred_keypoints, np.ones_like(pred_keypoints)[:, :, [0]]), axis=-1
        )
        pred_keypoints = self.img_res * (pred_keypoints + 0.5)
        if pred_v2d is not None:
            pred_v2d = self.img_res * (pred_v2d + 0.5)
        gt_keypoints[:, :, :-1] = self.img_res * (gt_keypoints[:, :, :-1] + 0.5)
        keypoint_matches = [
            (1, 12),
            (2, 8),
            (3, 7),
            (4, 6),
            (5, 9),
            (6, 10),
            (7, 11),
            (8, 14),
            (9, 2),
            (10, 1),
            (11, 0),
            (12, 3),
            (13, 4),
            (14, 5),
        ]
        if focal_length is None:
            fl = self.focal_length
        else:
            fl = focal_length

        for i in range(vertices.shape[0]):
            # fl = self.focal_length
            rend_img = torch.from_numpy(
                np.transpose(
                    self.render_mesh(
                        vertices[i],
                        camera_translation[i],
                        255 * images_np[i].copy(),
                        focal_length=fl[i],
                        side_view=False,
                        camera_center=camera_center[i],
                    ),
                    (2, 0, 1),
                )
            ).float()
            rend_img_side = torch.from_numpy(
                np.transpose(
                    self.render_mesh(
                        vertices[i],
                        camera_translation[i],
                        255 * images_np[i].copy(),
                        focal_length=fl[i],
                        side_view=True,
                        camera_center=camera_center[i],
                    ),
                    (2, 0, 1),
                )
            ).float()
            pred_keypoints_img = (
                render_atlas(255 * images_np[i].copy(), pred_keypoints[i], bboxes=None)
                / 255
            )
            pred_v2d_img = pred_v2d[i]
            gt_keypoints_img = (
                render_atlas(255 * images_np[i].copy(), gt_keypoints[i], bboxes=None)
                / 255
            )
            # Also draw kps on gt mesh
            curr_gt_mesh_img_debordered = gt_mesh_img[
                :,
                2
                + (2 + gt_keypoints_img.shape[0]) * i : 2
                + (2 + gt_keypoints_img.shape[0]) * i
                + gt_keypoints_img.shape[0],
                2:-2,
            ]
            curr_gt_mesh_img_debordered_kps = (
                render_atlas(
                    (curr_gt_mesh_img_debordered.permute(1, 2, 0).numpy().copy()) * 255,
                    gt_keypoints[i],
                    bboxes=None,
                )
                / 255
            )

            rend_imgs.append(torch.from_numpy(images[i]))

            if masks is not None:
                mask = masks[i]
                mask = np.repeat(mask, 3, axis=0)
                rend_imgs.append(torch.from_numpy(mask))
            # rend_imgs.append(gt_mesh_image)
            if pred_v2d is not None:
                img_copy_np = 255 * images_np[i].copy()
                for i in range(pred_v2d_img.shape[0]):
                    center = pred_v2d_img[i, :2].astype(int)
                    cv2.circle(img_copy_np, tuple(center.tolist()), 3, (0, 255, 0), -1)
                rend_imgs.append(torch.from_numpy(img_copy_np).permute(2, 0, 1) / 255.0)
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
            rend_imgs.append(torch.from_numpy(pred_keypoints_img).permute(2, 0, 1))
            rend_imgs.append(torch.from_numpy(gt_keypoints_img).permute(2, 0, 1))
            rend_imgs.append(
                torch.from_numpy(curr_gt_mesh_img_debordered_kps).permute(2, 0, 1)
            )
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def __call__(
        self,
        vertices,
        camera_translation,
        image,
        focal_length=5000,
        text=None,
        resize=None,
        side_view=False,
        baseColorFactor=(1.0, 1.0, 0.9, 1.0),
        rot_angle=90,
    ):
        renderer = pyrender.OffscreenRenderer(
            viewport_width=image.shape[1],
            viewport_height=image.shape[0],
            point_size=1.0,
        )
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=baseColorFactor
        )

        vertices = vertices + camera_translation.unsqueeze(1)
        camera_translation[0] *= -1.0

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0]
            )
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3)
        )
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2.0, image.shape[0] / 2.0]
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length, fy=focal_length, cx=camera_center[0], cy=camera_center[1]
        )
        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        if not side_view:
            output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image
        else:
            output_img = color[:, :, :3]
        if resize is not None:
            output_img = cv2.resize(output_img, resize)

        output_img = output_img.astype(np.float32)
        renderer.delete()
        return output_img

    def render_mesh(
        self,
        vertices,
        camera_translation,
        image,
        focal_length=5000,
        text=None,
        resize=None,
        side_view=False,
        mesh_color=(0.65098039, 0.74117647, 0.85882353),
        baseColorFactor=(1.0, 1.0, 0.9, 1.0),
        rot_angle=90,
        camera_center=None,
    ):
        renderer = Renderer(focal_length=focal_length, faces=self.faces)

        # camera_translation[0] *= -1.
        if side_view:
            while_image = np.ones_like(image) * 255
            image = while_image
        img_mesh = renderer(
            vertices,
            camera_translation,
            image.copy(),
            side_view=side_view,
            mesh_base_color=mesh_color,
            scene_bg_color=(1, 1, 1),
            camera_center=camera_center,
        )
        return img_mesh

    def label_collage_rows_simple(
        self,
        collage_tensor,
        labels,
        mask_scores=None,
        font_scale=0.5,
        thickness=1,
        padding_px=20,
    ):
        """
        Add row labels at the top of each row directly onto the collage.

        Parameters:
            collage_tensor (torch.Tensor): (3, H, W) float tensor
            labels (List[str]): list of labels (length == number of rows)
            font_scale (float): cv2 font scale
            thickness (int): line thickness for text
            padding_px (int): vertical space added above each row

        Returns:
            torch.Tensor: labeled image (3, H', W)
        """
        c, h, w = collage_tensor.shape
        n_rows = len(labels)
        row_height = h // n_rows

        # Convert to uint8 for OpenCV
        img_np = (collage_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # New height = original + padding per row
        new_height = h + n_rows * padding_px
        labeled_img = (
            np.ones((new_height, w, 3), dtype=np.uint8) * 255
        )  # white background

        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, label in enumerate(labels):
            y_start = i * (row_height + padding_px)
            # Paste the original row
            labeled_img[y_start + padding_px : y_start + padding_px + row_height] = (
                img_np[i * row_height : (i + 1) * row_height]
            )
            # if mask_scores is not None:
            #     label = f"{label} - mask_score={mask_scores[i]:.2f}"

            # Put the text at the top-left corner of the row area
            cv2.putText(
                labeled_img,
                label,
                (5, y_start + padding_px - 5),  # 5px above the image
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        # Convert back to tensor
        labeled_tensor = torch.from_numpy(labeled_img).float() / 255.0
        return labeled_tensor.permute(2, 0, 1)
