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
            num_hand_comps=32 if not cfg.MODEL.get('DISABLE_HAND_PCA', False) else 54,
            num_shape_comps=cfg.MODEL.ATLAS_HEAD.NUM_SHAPE_COMPS,
            num_scale_comps=cfg.MODEL.ATLAS_HEAD.NUM_SCALE_COMPS,
            atlas_model_path=cfg.MODEL.ATLAS_HEAD.ATLAS_MODEL_PATH,
            mesh_type=cfg.MODEL.ATLAS_HEAD.MESH_TYPE,
            extra_joint_regressor=cfg.MODEL.SMPL_HEAD.EXTRA_JOINT,
            smpl_model_path=cfg.MODEL.SMPL_HEAD.SMPL_MODEL_PATH,
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
            zero_face_for_nonparam_losses=cfg.MODEL.ATLAS_HEAD.get(
                "ZERO_FACE_FOR_NONPARAM_LOSSES", False
            ),
            detach_face_for_nonparam_losses=cfg.MODEL.ATLAS_HEAD.get(
                "DETACH_FACE_FOR_NONPARAM_LOSSES", False
            ),
            pred_global_wrist_rot=cfg.MODEL.ATLAS_HEAD.get(
                "PRED_GLOBAL_WRIST_ROT", False
            ),
            replace_local_with_pred_global_wrist_rot=cfg.MODEL.ATLAS_HEAD.get(
                "REPLACE_LOCAL_WITH_PRED_GLOBAL_WRIST_ROT", False
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
