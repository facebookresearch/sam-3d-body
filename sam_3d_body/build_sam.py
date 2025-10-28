import os
from .estimator import SAM3DBodyEstimator


def build_sam_3d_body_model(
    checkpoint_path: str = "",
    proto_path: str = "",
    detector_path: str = "",
    moge_path: str = "",
    bbox_threshold: float = 0.5,
    sam_path: str = "",
    use_mask: bool = False,
    use_triplet: bool = False,
    scale_factor=None,
    just_left_hand=False,
    use_face=False,
    mode="eval"
):

    model = SAM3DBodyEstimator(
        checkpoint_path=checkpoint_path,
        proto_path=proto_path,
        detector_path=detector_path,
        moge_path=moge_path,
    )
    if mode == "eval":
        model.eval()
    return model

def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "last.ckpt"), os.path.join(local_dir, "assets")


def build_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mohr_path = _hf_download(repo_id)
    return build_sam_3d_body_model(checkpoint_path=ckpt_path, proto_path=mohr_path)
