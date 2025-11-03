import os
from sam_3d_body.utils.checkpoint import load_state_dict
from sam_3d_body.utils.config import get_config

# from sam_3d_body.utils.config import get_config
from .sam3d_body import SAM3DBody
from .sam3d_body_triplet import SAM3DBodyTriplet
import torch 

def load_sam3d_body(checkpoint_path, proto_path, 
                    estimate_cam_int=True,
                    use_triplet=False,
                    use_twostage_for_hands=False,
                    use_face=False):
    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.ATLAS_HEAD.ZERO_FACE = True
    model_cfg.MODEL.ATLAS_HEAD.ATLAS_MODEL_PATH = proto_path
    model_cfg.freeze()

    if use_triplet: 
        model = SAM3DBodyTriplet(model_cfg, estimate_cam_int=estimate_cam_int)
        model.use_twostage_for_hands = use_twostage_for_hands
    else:
        model = SAM3DBody(model_cfg, estimate_cam_int=estimate_cam_int)
       
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)
    return model, model_cfg