import os
from .estimator import SAM3DBodyEstimator


def build_sam_3d_body_model(
    checkpoint_path: str = "",
    mhr_path: str = "",
    detector_path: str = "",
    moge_path: str = "",
):

    return SAM3DBodyEstimator(
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        detector_path=detector_path,
        moge_path=moge_path,
    )

def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "model.ckpt"), os.path.join(local_dir, "assets")


def build_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    return build_sam_3d_body_model(checkpoint_path=ckpt_path, mhr_path=mhr_path, **kwargs)
