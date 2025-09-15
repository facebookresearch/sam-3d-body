import os
import torch
from loguru import logger


def set_attention_backend():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    logger.info(f"GPU name is {gpu_name}")
    if "A100" in gpu_name or "H100" in gpu_name:
        os.environ["ATTN_BACKEND"] = "flash_attn"
    else:
        os.environ["ATTN_BACKEND"] = "sdpa"


# set attention backend before any import
# TODO Hao: Current does not see flash attention improve inference speed
# Disable it for now
# set_attention_backend()

from typing import List, Union
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np

from PIL import Image
from sam3d_image.pipeline import preprocess_utils
from sam3d_image.modality_preprocessor.img_and_mask_transforms import (
    get_mask,
)
from sam3d_image.pipeline.inference_utils import (
    get_pose_decoder,
    SLAT_MEAN,
    SLAT_STD,
    downsample_sparse_structure,
    prune_sparse_structure,
)

from sam3d_image.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)

from sam3d_image.model.backbone.modules import sparse as sp
from sam3d_image.model.backbone.utils import postprocessing_utils


class InferencePipeline:
    def __init__(
        self,
        ss_generator_config_path,
        ss_generator_ckpt_path,
        slat_generator_config_path,
        slat_generator_ckpt_path,
        ss_decoder_config_path,
        ss_decoder_ckpt_path,
        slat_decoder_gs_config_path,
        slat_decoder_gs_ckpt_path,
        slat_decoder_mesh_config_path,
        slat_decoder_mesh_ckpt_path,
        layout_model_config_path=None,
        layout_model_ckpt_path=None,
        decode_formats=["gaussian", "mesh"],
        dtype="float32",
        pad_size=1.0,
        device="cuda",  # TODO(Pierre) : Should default to "cpu", but leaving "cuda" as default for backward compatibility
        ss_preprocessor=preprocess_utils.get_default_preprocessor(),
        slat_preprocessor=preprocess_utils.get_default_preprocessor(),
        layout_preprocessor=preprocess_utils.get_default_preprocessor(),
        ss_condition_input_mapping=["image"],
        slat_condition_input_mapping=["image"],
        layout_condition_input_mapping=["image"],
        pose_decoder_name="default",
        use_layout_result=False,
        force_shape_in_layout=False,
        downsample_ss_dist=0,  # the distance we use to downsample
        ss_inference_steps=25,
        ss_rescale_t=3,
        ss_cfg_strength=7,
        ss_cfg_interval=[0, 500],
        slat_inference_steps=25,
        slat_rescale_t=3,
        slat_cfg_strength=5,
        slat_cfg_interval=[0, 500],
    ):
        self.device = torch.device(device)
        with self.device:
            self.decode_formats = decode_formats
            self.pad_size = pad_size
            self.ss_preprocessor = ss_preprocessor
            self.slat_preprocessor = slat_preprocessor
            self.layout_preprocessor = layout_preprocessor
            self.ss_condition_input_mapping = ss_condition_input_mapping
            self.slat_condition_input_mapping = slat_condition_input_mapping
            self.layout_condition_input_mapping = layout_condition_input_mapping
            self.pose_decoder = get_pose_decoder(pose_decoder_name)
            self.force_shape_in_layout = force_shape_in_layout
            self.downsample_ss_dist = downsample_ss_dist
            if dtype == "bfloat16":
                self.dtype = torch.bfloat16
            elif dtype == "float16":
                self.dtype = torch.float16
            elif dtype == "float32":
                self.dtype = torch.float32
            else:
                raise NotImplementedError

            logger.info("Loading model weights...")

            ss_generator = self.init_ss_generator(
                ss_generator_config_path, ss_generator_ckpt_path
            )
            slat_generator = self.init_slat_generator(
                slat_generator_config_path, slat_generator_ckpt_path
            )
            ss_decoder = self.init_ss_decoder(
                ss_decoder_config_path, ss_decoder_ckpt_path
            )
            slat_decoder_gs = self.init_slat_decoder_gs(
                slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
            )
            slat_decoder_mesh = self.init_slat_decoder_mesh(
                slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
            )
            layout_model = self.init_layout_model(
                layout_model_config_path, layout_model_ckpt_path
            )

            # Load conditioner embedder so that we only load it once
            ss_condition_embedder = self.init_ss_condition_embedder(
                ss_generator_config_path, ss_generator_ckpt_path
            )
            slat_condition_embedder = self.init_slat_condition_embedder(
                slat_generator_config_path, slat_generator_ckpt_path
            )
            layout_condition_embedder = self.init_layout_condition_embedder(
                layout_model_config_path, layout_model_ckpt_path
            )

            self.condition_embedders = {
                "ss_condition_embedder": ss_condition_embedder,
                "slat_condition_embedder": slat_condition_embedder,
                "layout_condition_embedder": layout_condition_embedder,
            }

            # override generator and condition embedder setting
            self.override_ss_generator_cfg_config(
                ss_generator,
                cfg_strength=ss_cfg_strength,
                inference_steps=ss_inference_steps,
                rescale_t=ss_rescale_t,
                cfg_interval=ss_cfg_interval,
            )
            self.override_slat_generator_cfg_config(
                slat_generator,
                cfg_strength=slat_cfg_strength,
                inference_steps=slat_inference_steps,
                rescale_t=slat_rescale_t,
                cfg_interval=slat_cfg_interval,
            )
            self.override_layout_model_cfg_config(
                layout_model,
                cfg_strength=ss_cfg_strength,
                inference_steps=ss_inference_steps,
                rescale_t=ss_rescale_t,
                cfg_interval=ss_cfg_interval,
            )

            self.models = torch.nn.ModuleDict(
                {
                    "ss_generator": ss_generator,
                    "slat_generator": slat_generator,
                    "ss_decoder": ss_decoder,
                    "slat_decoder_gs": slat_decoder_gs,
                    "slat_decoder_mesh": slat_decoder_mesh,
                    "layout_model": layout_model,
                }
            )
            logger.info("Loading model weights completed!")
            self.use_layout_result = use_layout_result

    def instantiate_and_load_from_pretrained(
        self,
        config,
        ckpt_path,
        state_dict_fn=None,
        state_dict_key="state_dict",
        device="cpu",
    ):
        model = instantiate(config)
        model = load_model_from_checkpoint(
            model,
            ckpt_path,
            strict=True,
            device="cpu",
            freeze=True,
            eval=True,
            state_dict_key=state_dict_key,
            state_dict_fn=state_dict_fn,
        )
        model = model.to(device)

        return model

    def init_ss_generator(self, ss_generator_config_path, ss_generator_ckpt_path):
        return self.instantiate_and_load_from_pretrained(
            OmegaConf.load(ss_generator_config_path)[
                "module"
            ]["generator"]["backbone"],
            os.path.join(ss_generator_ckpt_path),
            # state_dict_fn=remove_prefix_state_dict_fn("_base_models.generator."),
            state_dict_fn=filter_and_remove_prefix_state_dict_fn(
                "_base_models.generator."
            ),
            device=self.device,
        )

    def init_slat_generator(self, slat_generator_config_path, slat_generator_ckpt_path):
        config = OmegaConf.load(slat_generator_config_path)["module"]["generator"]["backbone"]
        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        )
        return self.instantiate_and_load_from_pretrained(
            config,
            slat_generator_ckpt_path,
            state_dict_fn=state_dict_prefix_func,
            device=self.device,
        )

    def init_ss_decoder(self, ss_decoder_config_path, ss_decoder_ckpt_path):
        # override to avoid problem loading
        config = OmegaConf.load(ss_decoder_config_path)
        if "pretrained_ckpt_path" in config:
            del config["pretrained_ckpt_path"]
        return self.instantiate_and_load_from_pretrained(
            config,
            ss_decoder_ckpt_path,
            device=self.device,
            state_dict_key=None,
        )

    def init_slat_decoder_gs(
        self, slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
    ):
        return self.instantiate_and_load_from_pretrained(
            OmegaConf.load(slat_decoder_gs_config_path),
            slat_decoder_gs_ckpt_path,
            device=self.device,
            state_dict_key=None,
        )

    def init_slat_decoder_mesh(
        self, slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
    ):
        return self.instantiate_and_load_from_pretrained(
            OmegaConf.load(slat_decoder_mesh_config_path),
            slat_decoder_mesh_ckpt_path,
            device=self.device,
        )

    def init_layout_model(self, layout_model_config_path, layout_model_ckpt_path):
        if layout_model_config_path is not None:
            assert layout_model_ckpt_path is not None
            hydra_config = OmegaConf.load(layout_model_config_path)["module"]["generator"]["backbone"]
            if self.force_shape_in_layout:
                hydra_config["_target_"] = (
                    "sam3d_image.model.backbone.generator.flow_matching.model.ConditionalFlowMatching"
                )
            layout_model = self.instantiate_and_load_from_pretrained(
                hydra_config,
                layout_model_ckpt_path,
                device=self.device,
                state_dict_fn=filter_and_remove_prefix_state_dict_fn(
                    "_base_models.generator."
                ),
            )
        else:
            layout_model = None

        return layout_model

    def init_ss_condition_embedder(
        self, ss_generator_config_path, ss_generator_ckpt_path
    ):
        conf = OmegaConf.load(
            ss_generator_config_path
        )
        if "condition_embedder" in conf["module"]:
            return self.instantiate_and_load_from_pretrained(
                conf["module"]["condition_embedder"]["backbone"],
                ss_generator_ckpt_path,
                state_dict_fn=filter_and_remove_prefix_state_dict_fn(
                    "_base_models.condition_embedder."
                ),
                device=self.device,
            )
        else:
            return None

    def init_slat_condition_embedder(
        self, slat_generator_config_path, slat_generator_ckpt_path
    ):
        return self.init_ss_condition_embedder(
            slat_generator_config_path, slat_generator_ckpt_path
        )

    def init_layout_condition_embedder(
        self, layout_model_config_path, layout_model_ckpt_path
    ):
        if layout_model_config_path is not None:
            assert layout_model_ckpt_path is not None
            return self.init_ss_condition_embedder(
                layout_model_config_path, layout_model_ckpt_path
            )
        else:
            return None

    def override_ss_generator_cfg_config(
        self,
        ss_generator,
        cfg_strength=7,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
    ):
        # override generator setting
        ss_generator.inference_steps = inference_steps
        ss_generator.reverse_fn.strength = cfg_strength
        ss_generator.reverse_fn.interval = cfg_interval
        ss_generator.rescale_t = rescale_t
        ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
        ss_generator.reverse_fn.unconditional_handling = "add_flag"

        logger.info(
            "ss_generator parameters: inference_steps={}, cfg_strength={}, cfg_interval={}, rescale_t={}",
            inference_steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
        )

    def override_slat_generator_cfg_config(
        self,
        slat_generator,
        cfg_strength=5,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
    ):
        slat_generator.inference_steps = inference_steps
        slat_generator.reverse_fn.strength = cfg_strength
        slat_generator.reverse_fn.interval = cfg_interval
        slat_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
        slat_generator.reverse_fn.backbone.force_zeros_cond = True
        slat_generator.rescale_t = rescale_t
        slat_generator.reverse_fn.backbone.condition_embedder.prenorm_features = True
        slat_generator.reverse_fn.unconditional_handling = "add_flag"

        logger.info(
            "slat_generator parameters: inference_steps={}, cfg_strength={}, cfg_interval={}, rescale_t={}",
            inference_steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
        )


    def override_layout_model_cfg_config(
        self,
        layout_model,
        cfg_strength=7,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
    ):
        if layout_model is not None:
            self.override_ss_generator_cfg_config(
                layout_model,
                cfg_strength=cfg_strength,
                inference_steps=inference_steps,
                rescale_t=rescale_t,
                cfg_interval=cfg_interval,
            )

    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed=42,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
    ) -> dict:
        """
        Parameters:
        - image (Image): The input image to be processed.
        - seed (int, optional): The random seed for reproducibility. Default is 42.
        - stage1_only (bool, optional): If True, only the sparse structure is sampled and returned. Default is False.
        - with_mesh_postprocess (bool, optional): If True, performs mesh post-processing. Default is True.
        - with_texture_baking (bool, optional): If True, applies texture baking to the 3D model. Default is True.
        Returns:
        - dict: A dictionary containing the GLB file and additional data from the sparse structure sampling.
        """
        # This should only happen if called from demo
        image = self.merge_image_and_mask(image, mask)
        with self.device:  # TODO(Pierre) make with context a decorator ?
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            layout_input_dict = self.preprocess_image(image, self.layout_preprocessor)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            torch.manual_seed(seed)
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict, inference_steps=stage1_inference_steps
            )

            # This is for decoupling oriented shape and layout model
            # ss_input_dict["x_shape_latent"] = ss_return_dict["shape"]
            layout_return_dict = self.run_layout_model(
                layout_input_dict,
                ss_return_dict,
                inference_steps=stage1_inference_steps,
            )
            ss_return_dict.update(layout_return_dict)
            ss_return_dict.update(self.pose_decoder(ss_return_dict))

            if stage1_only:
                logger.info("Finished!")
                return ss_return_dict

            coords = ss_return_dict["coords"]
            slat = self.sample_slat(
                slat_input_dict, coords, inference_steps=stage2_inference_steps
            )
            outputs = self.decode_slat(slat, self.decode_formats)

            # GLB files can be extracted from the outputs
            logger.info(
                f"Postprocessing mesh with option with_mesh_postprocess {with_mesh_postprocess}, with_texture_baking {with_texture_baking}..."
            )
            if "mesh" in outputs:
                glb = postprocessing_utils.to_glb(
                    outputs["gaussian"][0],
                    outputs["mesh"][0],
                    # Optional parameters
                    simplify=0.95,  # Ratio of triangles to remove in the simplification process
                    texture_size=1024,  # Size of the texture used for the GLB
                    verbose=False,
                    with_mesh_postprocess=with_mesh_postprocess,
                    with_texture_baking=with_texture_baking,
                )


            # glb.export("sample.glb")
            else:
                glb = None
            logger.info("Finished!")

            return {
                "glb": glb,
                "gs": outputs["gaussian"][0],
                **ss_return_dict,
                **outputs,
            }

    def merge_image_and_mask(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[None, np.ndarray, Image.Image],
    ):
        if mask is not None:
            if isinstance(image, Image.Image):
                image = np.array(image)

            mask = np.array(mask)
            if mask.ndim == 2:
                mask = mask[..., None]

            logger.info(f"Replacing alpha channel with the provided mask")
            assert mask.shape[:2] == image.shape[:2]
            image = np.concatenate([image[..., :3], mask], axis=-1)

        return image

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        logger.info("Decoding sparse latent...")
        ret = {}
        # with torch.autocast(device_type="cuda", dtype=self.dtype):
        # TODO: need to update flexicubes to make it fp16 compatible. The update is only local so disable fp16 for the decoder for now
        with torch.no_grad():
            if "mesh" in formats:
                ret["mesh"] = self.models["slat_decoder_mesh"](slat)
            if "gaussian" in formats:
                ret["gaussian"] = self.models["slat_decoder_gs"](slat)
        # if "radiance_field" in formats:
        #     ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    def is_mm_dit(self, model_name="ss_generator"):
        return hasattr(self.models[model_name].reverse_fn.backbone, "latent_mapping")

    def embed_condition(self, condition_embedder, *args, **kwargs):
        if condition_embedder is not None:
            logger.info("Running condition embedder ...")
            tokens = condition_embedder(*args, **kwargs)
            logger.info("Condition embedder finishes!")
            return tokens, None, None
        return None, args, kwargs

    def get_condition_input(self, condition_embedder, input_dict, input_mapping):
        condition_args = self.map_input_keys(input_dict, input_mapping)
        condition_kwargs = {
            k: v for k, v in input_dict.items() if k not in input_mapping
        }
        embedded_cond, condition_args, condition_kwargs = self.embed_condition(
            condition_embedder, *condition_args, **condition_kwargs
        )
        if embedded_cond is not None:
            condition_args = (embedded_cond,)
            condition_kwargs = {}

        return condition_args, condition_kwargs

    def sample_sparse_structure(self, ss_input_dict: dict, inference_steps=None):
        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]
        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps

        image = ss_input_dict["image"]
        bs = image.shape[0]
        logger.info(
            f"Sampling sparse structure {ss_generator.inference_steps} steps ..."
        )

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                if self.is_mm_dit():
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                else:
                    latent_shape_dict = (bs,) + (4096, 8)

                condition_args, condition_kwargs = self.get_condition_input(
                    self.condition_embedders["ss_condition_embedder"],
                    ss_input_dict,
                    self.ss_condition_input_mapping,
                )
                return_dict = ss_generator(
                    latent_shape_dict,
                    image.device,
                    *condition_args,
                    **condition_kwargs,
                )
                if not self.is_mm_dit():
                    return_dict = {"shape": return_dict}

                shape_latent = return_dict["shape"]
                ss = ss_decoder(
                    shape_latent.permute(0, 2, 1)
                    .contiguous()
                    .view(shape_latent.shape[0], 8, 16, 16, 16)
                )
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

                # downsample output
                return_dict["coords_original"] = coords
                original_shape = coords.shape
                if self.downsample_ss_dist > 0:
                    coords = prune_sparse_structure(
                        coords,
                        max_neighbor_axes_dist=self.downsample_ss_dist,
                    )
                coords = downsample_sparse_structure(coords)
                logger.info(
                    f"Downsampled coords from {original_shape[0]} to {coords.shape[0]}"
                )
                return_dict["coords"] = coords

        ss_generator.inference_steps = prev_inference_steps
        return return_dict

    def run_layout_model(
        self, ss_input_dict: dict, ss_return_dict: dict, inference_steps=None
    ):
        if self.models["layout_model"] is None:
            return {}
        ss_generator = self.models["layout_model"]
        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps
        logger.info(f"Sampling layout model {ss_generator.inference_steps} steps ...")

        image = ss_input_dict["image"]
        bs = image.shape[0]

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                if self.is_mm_dit("layout_model"):
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                else:
                    latent_shape_dict = (bs,) + (4096, 8)

                condition_args, condition_kwargs = self.get_condition_input(
                    self.condition_embedders["layout_condition_embedder"],
                    ss_input_dict,
                    self.layout_condition_input_mapping,
                )
                if self.force_shape_in_layout:
                    condition_kwargs["noise_override"] = {
                        "shape": ss_return_dict["shape"],
                    }
                return_dict = ss_generator(
                    latent_shape_dict,
                    image.device,
                    *condition_args,
                    **condition_kwargs,
                )
                if not self.is_mm_dit("layout_model"):
                    return_dict = {"shape": return_dict}

        if not self.use_layout_result:
            return_dict.pop("shape")
            if "quaternion" in return_dict:
                return_dict.pop("quaternion")
            if "rotation" in return_dict:
                return_dict.pop("rotation")

        ss_generator.inference_steps = prev_inference_steps
        return return_dict

    def sample_slat(
        self, slat_input: dict, coords: torch.Tensor, inference_steps=25
    ) -> sp.SparseTensor:
        image = slat_input["image"]
        DEVICE = image.device
        slat_generator = self.models["slat_generator"]
        latent_shape = (image.shape[0],) + (coords.shape[0], 8)
        prev_inference_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps

        logger.info(
            f"Sampling sparse latent {slat_generator.inference_steps} steps ..."
        )
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            with torch.no_grad():
                condition_args, condition_kwargs = self.get_condition_input(
                    self.condition_embedders["slat_condition_embedder"],
                    slat_input,
                    self.slat_condition_input_mapping,
                )
                condition_args += (coords.cpu().numpy(),)
                # condition_kwargs["coords"] = coords.cpu().numpy()
                slat = slat_generator(
                    latent_shape, DEVICE, *condition_args, **condition_kwargs
                )
                slat = sp.SparseTensor(
                    coords=coords,
                    feats=slat[0],
                ).to(DEVICE)
                slat = slat * SLAT_STD.to(DEVICE) + SLAT_MEAN.to(DEVICE)

        slat_generator.inference_steps = prev_inference_steps
        return slat

    def map_input_keys(self, item, condition_input_mapping):
        output = [item[k] for k in condition_input_mapping]

        return output

    def image_to_float(self, image):
        image = np.array(image)
        image = image / 255
        image = image.astype(np.float32)
        return image

    def preprocess_image(
        self, image: Union[Image.Image, np.ndarray], preprocessor
    ) -> torch.Tensor:
        # canonical type is numpy
        if not isinstance(input, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = (get_mask(rgba_image, None, "ALPHA_CHANNEL") > 0).float()

        processor_out = preprocessor(rgb_image, rgb_image_mask, None)

        item = {
            "mask": processor_out["mask"][None].to(self.device),
            "image": processor_out["image"][None].to(self.device),
            "rgb_image": processor_out["rgb_image"][None].to(self.device),
            "rgb_image_mask": processor_out["mask"][rgb_image_mask].to(self.device),
        }

        return item
