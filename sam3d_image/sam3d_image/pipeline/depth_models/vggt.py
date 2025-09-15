from sam3d_image.data.dataset.tdfy.img_processing import preprocess_img
from .base import DepthModel
import torch

class VGGT(DepthModel):
    @torch.no_grad()
    def __call__(self, image):
        processed_img = preprocess_img(image[None], img_target_shape=518)[0]
        output = self.model(processed_img.to(self.device))
        output["pointmaps"] = output["world_points"][0, 0].detach()
        return output