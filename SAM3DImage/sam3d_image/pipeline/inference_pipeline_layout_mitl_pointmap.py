import os

import numpy as np
import torch
import trimesh
import open3d as o3d
from loguru import logger
from typing import Union
from PIL import Image
from omegaconf import OmegaConf

from sam3d_image.pipeline import preprocess_utils
from sam3d_image.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_image.pipeline.inference_utils import (
    get_pose_decoder,
    voxelize_mesh,
    preprocess_mesh,
    trimesh2o3d_mesh,
    update_layout,
)


class InferencePipelineLayoutMITLPointMap(InferencePipelinePointMap):
    def __init__(
        self,
        # layout model here
        layout_model_config_path,
        layout_model_ckpt_path,
        ss_encoder_config_path,
        ss_encoder_ckpt_path,
        depth_model,
        dtype="bfloat16",
        device="cuda",
        version="v0",
        layout_preprocessor=preprocess_utils.get_default_preprocessor(),
        layout_condition_input_mapping=["image"],
        pose_decoder_name="default",
        workspace_dir="",
        layout_post_optimization_method=None,
    ):
        self.device = torch.device(device)
        with self.device:
            self.version = version
            self.layout_preprocessor = layout_preprocessor
            self.layout_condition_input_mapping = layout_condition_input_mapping
            self.pose_decoder = get_pose_decoder(pose_decoder_name)
            self.force_shape_in_layout = True
            self.use_layout_result = True
            self.workspace_dir = workspace_dir
            if dtype == "bfloat16":
                self.dtype = torch.bfloat16
            elif dtype == "float16":
                self.dtype = torch.float16
            elif dtype == "float32":
                self.dtype = torch.float32
            else:
                raise NotImplementedError
            logger.info("Loading model weights...")

            layout_model = self.init_layout_model(
                layout_model_config_path, layout_model_ckpt_path
            )

            ss_encoder = self.init_ss_encoder(
                ss_encoder_config_path,
                ss_encoder_ckpt_path,
            )

            # Load conditioner embedder so that we only load it once
            layout_condition_embedder = self.init_layout_condition_embedder(
                layout_model_config_path, layout_model_ckpt_path
            )
            self.condition_embedders = {
                "layout_condition_embedder": layout_condition_embedder,
            }
            self.override_layout_model_cfg_config(layout_model)
            self.models = torch.nn.ModuleDict(
                {
                    "layout_model": layout_model,
                    "ss_encoder": ss_encoder,
                }
            )
            logger.info("Loading model weights completed!")
            self.depth_model = depth_model
            self.layout_post_optimization_method = layout_post_optimization_method

    def init_ss_encoder(self, ss_encoder_config_path, ss_encoder_ckpt_path):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_encoder_config_path)
        )
        if "pretrained_ckpt_path" in config:
            del config["pretrained_ckpt_path"]
        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, ss_encoder_ckpt_path),
            device=self.device,
        )

    def encode_ss_latent(self, ss):
        ss = ss.to(self.device).float()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                result = self.models["ss_encoder"](ss[None])
        return result["mean"].reshape(1, 8, -1).permute(0, 2, 1)

    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mesh_path: str,
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed=42,
        return_pm=False,  # for visualization
        to_halo=True,  # Halo Deployment
    ):
        # mesh = o3d.io.read_triangle_mesh(mesh_path)
        # o3d is not robust; use trimesh then convert
        trimesh_mesh = trimesh.load_mesh(mesh_path, force="mesh")
        if isinstance(trimesh_mesh, trimesh.Scene):
            trimesh_mesh = trimesh_mesh.dump(concatenate=True)
        mesh = trimesh2o3d_mesh(trimesh_mesh)
        image = self.merge_image_and_mask(image, mask)
        with self.device:
            pointmap_dict = self.compute_pointmap(image)
            pointmap = pointmap_dict["pointmap"]
            pointmap_scale = pointmap_dict["pointmap_scale"]
            pointmap_shift = pointmap_dict["pointmap_shift"]
            layout_input_dict = self.preprocess_image(
                image, self.layout_preprocessor, pointmap=pointmap
            )
            ss, scale, center = voxelize_mesh(mesh)

            logger.info("Encoding ss latents...")
            torch.manual_seed(seed)
            shape_latent = self.encode_ss_latent(ss)

            ss_return_dict = {"shape": shape_latent}

            logger.info("Sampling Layout...")
            layout_return_dict = self.run_layout_model(
                layout_input_dict, ss_return_dict
            )
            logger.info("Finished!")

            layout_return_dict.update(
                self.pose_decoder(
                    layout_return_dict,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )

            if self.layout_post_optimization_method is not None:
                trimesh_mesh = trimesh.load_mesh(mesh_path)
                trimesh_mesh = preprocess_mesh(trimesh_mesh)

                postprocessed_pose = self.run_post_optimization(
                    trimesh_mesh,
                    pointmap_dict["intrinsics"],
                    layout_return_dict,
                    layout_input_dict,
                )
                layout_return_dict.update(postprocessed_pose)

            updated_t, updated_s, updated_r = update_layout(
                layout_return_dict["translation"],
                layout_return_dict["scale"],
                layout_return_dict["quaternion"],
                center,
                scale,
                to_halo=to_halo,
            )
            layout_return_dict.update(
                {
                    "quaternion": updated_r[None],
                    "translation": updated_t,
                    "scale": updated_s,
                }
            )

            layout_return_dict.update(ss_return_dict)
            if return_pm:
                pts = type(self)._down_sample_img(pointmap)
                pts_colors = type(self)._down_sample_img(pointmap_dict["pts_color"])
                return {
                    **layout_return_dict,
                    "pointmap": pts.cpu().permute((1, 2, 0)),  # HxWx3
                    "pointmap_colors": pts_colors.cpu().permute((1, 2, 0)),  # HxWx3
                }
            else:
                return layout_return_dict
