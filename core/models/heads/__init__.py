from ..modules import to_2tuple
from .atlas_head import ATLAS46Head
from .camera_head import PerspectiveHead, WeakPerspectiveHead


def build_head(cfg, head_type="atlas46"):
    if head_type == "atlas46":
        return ATLAS46Head(
            input_dim=cfg.MODEL.DECODER.DIM,
            mlp_depth=cfg.MODEL.ATLAS_HEAD.get("MLP_DEPTH", 1),
            # num_body_joints=cfg.MODEL.ATLAS_HEAD.NUM_BODY_JOINTS,
            num_hand_shape_comps=cfg.MODEL.ATLAS_HEAD.NUM_HAND_SHAPE_COMPS,
            num_shape_comps=cfg.MODEL.ATLAS_HEAD.NUM_SHAPE_COMPS,
            num_scale_comps=cfg.MODEL.ATLAS_HEAD.NUM_SCALE_COMPS,
            atlas_model_path=cfg.MODEL.ATLAS_HEAD.ATLAS_MODEL_PATH,
            mesh_type=cfg.MODEL.ATLAS_HEAD.MESH_TYPE,
            fix_kps_eye_and_chin=cfg.MODEL.ATLAS_HEAD.get("FIX_KPS_EYE_AND_CHIN", True),
            znorm_fullbody_scales=cfg.MODEL.ATLAS_HEAD.get(
                "ZNORM_FULL_BODY_SCALES", True
            ),
            ffn_zero_bias=cfg.MODEL.ATLAS_HEAD.get("FFN_ZERO_BIAS", False),
            mlp_channel_div_factor=cfg.MODEL.ATLAS_HEAD.get(
                "MLP_CHANNEL_DIV_FACTOR", 8
            ),
            enable_slim_keypoint_mapping=cfg.MODEL.ATLAS_HEAD.get(
                "ENABLE_SLIM_KEYPOINT_MAPPING", False
            ),
            zero_face=cfg.MODEL.ATLAS_HEAD.get(
                "ZERO_FACE", False
            ),
        )
    elif head_type == "weak_perspective":
        return WeakPerspectiveHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            img_size=to_2tuple(cfg.MODEL.IMAGE_SIZE),
        )
    elif head_type == "perspective":
        return PerspectiveHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            img_size=to_2tuple(cfg.MODEL.IMAGE_SIZE),
            mlp_depth=cfg.MODEL.get("CAMERA_HEAD", dict()).get("MLP_DEPTH", 1),
            mlp_channel_div_factor=cfg.MODEL.get("CAMERA_HEAD", dict()).get(
                "MLP_CHANNEL_DIV_FACTOR", 8
            ),
        )
    else:
        raise ValueError("Invalid head type: ", head_type)
