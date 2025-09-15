from .base import DepthModel

class MoGe(DepthModel):
    def __call__(self, image):
        output = self.model.infer(
            image.to(self.device), force_projection=False
        )
        pointmaps = output["points"]
        output["pointmaps"] = pointmaps
        return output