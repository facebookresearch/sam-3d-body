import torch
from loguru import logger
from PIL import Image

from sam3d_image.pipeline.inference_pipeline import InferencePipeline
from sam3d_image.model.backbone.trellis.utils import postprocessing_utils


class InferenceTextureSwapPipeline(InferencePipeline):
    def run(
        self,
        image: Image,
        texture_swap_image: Image,
        seed=42,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
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
        with self.device:  # TODO(Pierre) make with context a decorator ?
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            slat_input_dict = self.preprocess_image(
                texture_swap_image, self.slat_preprocessor
            )
            logger.info("Sampling sparse structure...")
            torch.manual_seed(seed)
            ss_return_dict = self.sample_sparse_structure(ss_input_dict)

            # This is for decoupling oriented shape and layout model
            # ss_input_dict["x_shape_latent"] = ss_return_dict["shape"]
            layout_return_dict = self.run_layout_model(ss_input_dict, ss_return_dict)
            ss_return_dict.update(layout_return_dict)
            ss_return_dict.update(self.pose_decoder(ss_return_dict))

            if stage1_only:
                logger.info("Finished!")
                return ss_return_dict

            coords = ss_return_dict["coords"]
            logger.info("Sampling sparse latent...")
            slat = self.sample_slat(slat_input_dict, coords)
            logger.info("Decoding sparse latent...")
            outputs = self.decode_slat(slat, self.decode_formats)

            # GLB files can be extracted from the outputs
            logger.info(
                f"Postprocessing mesh with option with_mesh_postprocess {with_mesh_postprocess}, with_texture_baking {with_texture_baking}..."
            )
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
            logger.info("Finished!")

            return {"glb": glb, "gs": outputs["gaussian"][0], **ss_return_dict}
