# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch


class FOVEstimator:
    def __init__(self, name="moge2", device="cuda", offload_mode="none", **kwargs):
        self.device = torch.device(device)
        self.offload_mode = offload_mode

        if name == "moge2":
            print(f"########### Using fov estimator: MoGe2... (offload_mode={offload_mode!r})")
            # Always load to CPU first regardless of offload_mode
            self.fov_estimator = load_moge("cpu", **kwargs)
            self.fov_estimator_func = run_moge
            self.fov_estimator.eval()

            if offload_mode == "none":
                # Original behaviour: move to target device now
                self.fov_estimator = self.fov_estimator.to(self.device)
            # offload_mode="cpu": stays in RAM, moved per-call in get_cam_intrinsics
        else:
            raise NotImplementedError

    def get_cam_intrinsics(self, img, **kwargs):
        if self.offload_mode == "cpu" and self.device.type != "cpu":
            self.fov_estimator.to(self.device)
            try:
                result = self.fov_estimator_func(self.fov_estimator, img, self.device, **kwargs)
            finally:
                self.fov_estimator.cpu()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            return result
        else:
            _device = next(self.fov_estimator.parameters()).device
            return self.fov_estimator_func(self.fov_estimator, img, _device, **kwargs)


def load_moge(device, path=""):
    from moge.model.v2 import MoGeModel

    if path == "":
        path = "Ruicheng/moge-2-vitl-normal"
    # Always load to CPU first; caller decides final device placement
    moge_model = MoGeModel.from_pretrained(path).to(device)
    return moge_model


def run_moge(model, input_image, device):
    # We expect the image to be RGB already
    H, W, _ = input_image.shape
    # Derive device from model parameters so this works with CPU offloading
    _device = next(model.parameters()).device
    input_image = torch.tensor(
        input_image / 255, dtype=torch.float32, device=_device
    ).permute(2, 0, 1)

    # Infer w/ MoGe2
    moge_data = model.infer(input_image)

    # get intrinsics
    intrinsics = denormalize_f(moge_data["intrinsics"].cpu().numpy(), H, W)
    v_focal = intrinsics[1, 1]

    # override hfov with v_focal
    intrinsics[0, 0] = v_focal
    # add batch dim
    cam_intrinsics = intrinsics[None]

    return cam_intrinsics


def denormalize_f(norm_K, height, width):
    # Extract cx and cy from the normalized K matrix
    cx_norm = norm_K[0][2]  # c_x is at K[0][2]
    cy_norm = norm_K[1][2]  # c_y is at K[1][2]

    fx_norm = norm_K[0][0]  # Normalized fx
    fy_norm = norm_K[1][1]  # Normalized fy
    # s_norm = norm_K[0][1]   # Skew (usually 0)

    # Scale to absolute values
    fx_abs = fx_norm * width
    fy_abs = fy_norm * height
    cx_abs = cx_norm * width
    cy_abs = cy_norm * height
    # s_abs = s_norm * width
    s_abs = 0

    # Construct absolute K matrix
    abs_K = torch.tensor(
        [[fx_abs, s_abs, cx_abs], [0.0, fy_abs, cy_abs], [0.0, 0.0, 1.0]]
    )
    return abs_K
