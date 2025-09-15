## Table of Content

### backbones
- ViT-H from HMR2.0
    - `vit_hmr` | `hmr2` | `vit`
- Sapiens backbones
    - `sapiens_0.3b` | `sapiens_0.6b` | `sapiens_1b` | `sapiens_2b`
- DINOv2 backbones
    - `dinov2_vitb14` | `dinov2_vitl14` | `dinov2_vits14`

### decoders
- prompt_encoder
    - SAM-style prompt encoder
- promptable_decoder
    - SAM-style decoder

### heads
- smpl_head
    - an all-in-one implementation of SMPL head: given initial estimate and extracted features, return SMPL parameters, 3D joints/vertices and reprojected 2D joints (with optional extra_joint_regressor).
- atlas_head
- heatmap_head
- hmr2_head
    - for reproducing HMR2.0 only

### losses
- keypoint_loss
- param_loss
- heatmap_loss

### meta_arch
- promptable_hmr

### geometry_utils.py
- `aa_to_rotmat`
    - Convert axis-angle representation to rotation matrix. (roma.rotvec_to_rotmat)
- `rot6d_to_rotmat`
    - Convert 6D rotation representation to 3x3 rotation matrix.
- `transform_points`
    - Transform a set of 3D points given translation and rotation.
- `get_intrinsic_matrix`
    - Populate intrinsic camera matrix K given focal length and principle point.
- `perspective_projection`
    - Computes the perspective projection of a set of 3D points assuming the extrinsinc params have already been applied
- `inverse_perspective_projection`
    - Computes the inverse perspective projection of a set of points given an estimated distance.
