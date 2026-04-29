# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch
from .models.meta_arch import SAM3DBody
from .utils.config import get_config
from .utils.checkpoint import load_state_dict


# ---------------------------------------------------------------------------
# CPU offload wrapper
# ---------------------------------------------------------------------------

class CPUOffloadWrapper(torch.nn.Module):
    """
    Keeps model weights in CPU RAM between inference calls.
    On each forward(), moves weights to `inference_device`, runs the pass,
    then moves everything back to CPU and clears the CUDA cache.

    This means your GPU only needs enough VRAM for the active forward pass,
    not to hold the full model persistently.
    """

    def __init__(self, model: torch.nn.Module, inference_device: torch.device):
        super().__init__()
        self.inner = model.cpu()
        self.inference_device = inference_device

    @staticmethod
    def _move_non_persistent_buffers(module: torch.nn.Module, device: torch.device):
        """
        Recursively move all non-persistent buffers to `device`.

        PyTorch's .to() intentionally skips buffers registered with
        persistent=False (e.g. image_mean / image_std in SAM3DBody).
        These are tracked in the internal _non_persistent_buffers_set set
        on each submodule. We move them manually so they match the device
        of the parameters during the forward pass.
        """
        for submodule in module.modules():
            non_persistent = getattr(submodule, "_non_persistent_buffers_set", set())
            for name in non_persistent:
                buf = getattr(submodule, name, None)
                if isinstance(buf, torch.Tensor):
                    setattr(submodule, name, buf.to(device))

    def forward(self, *args, **kwargs):
        # Move parameters + persistent buffers to GPU
        self.inner.to(self.inference_device)
        # Move non-persistent buffers (e.g. image_mean/image_std) separately —
        # .to() silently skips these, causing device mismatch errors.
        self._move_non_persistent_buffers(self.inner, self.inference_device)
        try:
            args = _to_device(args, self.inference_device)
            kwargs = _to_device(kwargs, self.inference_device)
            out = self.inner(*args, **kwargs)
        finally:
            # Always move everything back to CPU, even on exception.
            self.inner.cpu()
            self._move_non_persistent_buffers(self.inner, torch.device("cpu"))
            if self.inference_device.type == "cuda":
                torch.cuda.empty_cache()
        return out

    # Proxy attribute access so callers (e.g. estimator.faces) still work.
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


def _to_device(obj, device):
    """Recursively move tensors in nested dicts / lists / tuples to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_sam_3d_body(
    checkpoint_path: str = "",
    device: str = "cuda",
    mhr_path: str = "",
    offload_mode: str = "cpu",   # "none" | "cpu" | "auto"
):
    """
    Load the SAM 3D Body model with optional CPU/RAM offloading.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model .ckpt file.
    device : str
        Target device for inference (e.g. "cuda", "cuda:0", "cpu").
    mhr_path : str
        Path to the MHR model .pt file.
    offload_mode : str
        "none" – original behaviour, model fully resident on `device`.
        "cpu"  – weights stay in CPU RAM; moved to `device` only during
                 each forward() call, then returned to CPU afterwards.
                 Requires only enough VRAM for a single forward pass.
        "auto" – uses HuggingFace Accelerate to split layers across
                 GPU + CPU RAM automatically. Requires: pip install accelerate
    """
    print(f"Loading SAM 3D Body model... (offload_mode={offload_mode!r})")

    # ---- config --------------------------------------------------------
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )
    model_cfg = get_config(model_cfg)
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()

    # ---- build model skeleton (always on CPU) ---------------------------
    model = SAM3DBody(model_cfg)

    # ---- load checkpoint (always to CPU RAM first) ----------------------
    # map_location="cpu" ensures the raw checkpoint never occupies VRAM,
    # regardless of which offload_mode is chosen.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_state_dict(model, state_dict, strict=False)
    del checkpoint, state_dict   # free CPU RAM immediately

    model.eval()

    # ---- apply offload strategy ----------------------------------------
    inference_device = torch.device(device)

    if offload_mode == "none":
        # Original behaviour: move everything to the target device now.
        model = model.to(inference_device)

    elif offload_mode == "cpu":
        if inference_device.type == "cpu":
            # Nothing to offload; just stay on CPU.
            pass
        else:
            # Wrap: weights live on CPU, hop to GPU only during forward().
            model = CPUOffloadWrapper(model, inference_device)

    elif offload_mode == "auto":
        try:
            from accelerate import dispatch_model, infer_auto_device_map
            from accelerate.utils import get_balanced_memory
        except ImportError as exc:
            raise ImportError(
                "offload_mode='auto' requires the `accelerate` package.\n"
                "Install it with:  pip install accelerate"
            ) from exc

        max_memory = get_balanced_memory(model, no_split_module_classes=None)
        device_map = infer_auto_device_map(
            model, max_memory=max_memory, no_split_module_classes=None
        )
        print(f"Accelerate device_map: {device_map}")
        model = dispatch_model(model, device_map=device_map)

    else:
        raise ValueError(
            f"Unknown offload_mode {offload_mode!r}. "
            "Choose from: 'none', 'cpu', 'auto'."
        )

    return model, model_cfg


# ---------------------------------------------------------------------------
# HuggingFace helpers (unchanged from original, except offload_mode forwarded)
# ---------------------------------------------------------------------------

def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return (
        os.path.join(local_dir, "model.ckpt"),
        os.path.join(local_dir, "assets", "mhr_model.pt"),
    )


def load_sam_3d_body_hf(repo_id, offload_mode: str = "cpu", **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    return load_sam_3d_body(
        checkpoint_path=ckpt_path,
        mhr_path=mhr_path,
        offload_mode=offload_mode,
        **kwargs,
    )
