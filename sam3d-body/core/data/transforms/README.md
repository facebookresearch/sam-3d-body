## Table of Content

### bbox_utils.py
- `bbox_xyxy2xywh`
- `bbox_xywh2xyxy`
- `bbox_xyxy2cs`
- `bbox_xywh2cs`
- `bbox_cs2xyxy`
- `bbox_cs2xywh`
- `flip_bbox`
- `fix_aspect_ratio`
    - Reshape the bbox to a fixed aspect ratio.
- `get_udp_warp_matrix`
- `get_warp_matrix`
    - Calculate the affine transformation matrix that can warp the bbox area in the input image to the output size.

### common.py
- `Compose`
- `VisionTransformWrapper`
    - A wrapper to use torchvision transform functions in this codebase.
- `GetBBoxCenterScale`
- `SquarePad`
- `ToPIL`
- `ToCv2`
- `TopdownAffine`
    - Get the bbox image as the model input by affine transform. (crop the bbox image)
