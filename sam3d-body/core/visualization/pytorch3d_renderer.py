import cv2
import numpy as np
import torch
from pytorch3d.renderer import (
    Materials,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    roi_image[mask] = image[mask]
    out_image[bbox[1] : bbox[3], bbox[0] : bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype

    K = torch.zeros((K_org.shape[0], 4, 4)).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1

    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = right - left
    height = bottom - top

    new_left = torch.clamp(cx - width / 2 * scaleFactor, min=0, max=img_w - 1)
    new_right = torch.clamp(cx + width / 2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h - 1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = (
        torch.stack(
            (
                new_left.detach(),
                new_top.detach(),
                new_right.detach(),
                new_bottom.detach(),
            )
        )
        .int()
        .float()
        .T
    )

    return bbox


class Renderer:
    def __init__(self, width, height, focal_length, K=None, device="cuda", faces=None):

        self.width = width
        self.height = height
        self.focal_length = focal_length

        self.device = device
        if faces is not None:
            self.faces = (
                torch.from_numpy((faces).astype("int")).unsqueeze(0).to(self.device)
            )

        self.initialize_camera_params(K=K)
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()

    def create_renderer(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0], blur_radius=1e-5
                ),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            ),
        )

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False,
        )

    def initialize_camera_params(self, K=None, width=None, height=None):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""

        # Extrinsics
        self.R = (
            torch.diag(torch.tensor([1, 1, 1])).float().to(self.device).unsqueeze(0)
        )

        self.T = torch.tensor([0, 0, 0]).unsqueeze(0).float().to(self.device)

        # Intrinsics
        if K is None:
            self.K = (
                torch.tensor(
                    [
                        [self.focal_length, 0, self.width / 2],
                        [0, self.focal_length, self.height / 2],
                        [0, 0, 1],
                    ]
                )
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
        else:
            if isinstance(K, np.ndarray):
                self.K = torch.from_numpy(K).reshape(1, 3, 3).float().to(self.device)
            else:
                self.K = K.reshape(1, 3, 3).float().to(self.device)

        self.width = width if width is not None else self.width
        self.height = height if height is not None else self.height
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()

    def update_bbox(self, x3d, scale=2.0, mask=None):
        """Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """

        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(
                x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1)
            )

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(
        self,
    ):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, vertices, background, colors=[0.8, 0.8, 0.8]):
        self.update_bbox(vertices[::50], scale=1.2)
        vertices = vertices.unsqueeze(0)

        if colors[0] > 1:
            colors = [c / 255.0 for c in colors]
        verts_features = (
            torch.tensor(colors)
            .reshape(1, 1, 3)
            .to(device=vertices.device, dtype=vertices.dtype)
        )
        verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)

        mesh = Meshes(
            verts=vertices,
            faces=self.faces,
            textures=textures,
        )

        materials = Materials(device=self.device, specular_color=(colors,), shininess=0)

        results = torch.flip(
            self.renderer(
                mesh, materials=materials, cameras=self.cameras, lights=self.lights
            ),
            [1, 2],
        )

        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3

        image = overlay_image_onto_background(
            image, mask, self.bboxes, background.copy()
        )
        self.reset_bbox()
        return image

    def draw_bboxes(self, img, bboxes, convention="xywh"):
        # Draw bbox around the image

        for bbox in bboxes:
            bbox = bbox.reshape(-1)
            if convention == "xywh":
                x1, y1, w, h = bbox[:4].cpu().numpy()
                x2, y2 = x1 + w, y1 + h

            img = cv2.line(
                img,
                (int(x1), int(y1)),
                (int(x1), int(y2)),
                color=(255, 0, 0),
                thickness=5,
            )
            img = cv2.line(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y1)),
                color=(255, 0, 0),
                thickness=5,
            )
            img = cv2.line(
                img,
                (int(x2), int(y2)),
                (int(x2), int(y1)),
                color=(255, 0, 0),
                thickness=5,
            )
            img = cv2.line(
                img,
                (int(x2), int(y2)),
                (int(x1), int(y2)),
                color=(255, 0, 0),
                thickness=5,
            )
        return img
