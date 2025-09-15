from fvcore.common.registry import Registry

FILTERS_REGISTRY = Registry("FILTERS")


@FILTERS_REGISTRY.register()
def atlasv1_itw_low():
    return dict(
        atlas_conf=0.1,
        kps3d_conf=0.1,
        suppress_kp_conf_thresh=0.3,
        filter_numkp=6,
        filter_bbox_size=60,
        filter_mae=15,
        filter_atlas_loss=0.008,
    )


@FILTERS_REGISTRY.register()
def atlasv1_gt_low():
    return dict(
        atlas_conf=0.5,
        kps3d_conf=0.5,
        filter_pve=90,
        filter_mae=15,
        filter_atlas_loss=0.008,
    )


@FILTERS_REGISTRY.register()
def atlas46_itw_lowest():
    # These are super permissive to allow basically all frames to contribute
    # (since we don't have PCBs, and loss now includes other stuff like limits, is L1 instad of euc, etc)
    # TODO (jinhyun1): Should also put in joint limits, etc
    return dict(
        filter_no_atlas=True,
    )

@FILTERS_REGISTRY.register()
def atlas46_empty():
    # These are super permissive to allow basically all frames to contribute
    # (since we don't have PCBs, and loss now includes other stuff like limits, is L1 instad of euc, etc)
    # TODO (jinhyun1): Should also put in joint limits, etc
    return dict(
        filter_no_atlas=False,
    )


@FILTERS_REGISTRY.register()
def atlasv1_gt_high():
    return dict(
        atlas_conf=0.5,
        kps3d_conf=0.5,
        filter_pve=60,
        filter_mae=5,
        filter_atlas_loss=0.006,
    )


@FILTERS_REGISTRY.register()
def itw_v1():
    return dict(
        smpl_conf=0.1,
        kps3d_conf=0.1,
        filter_bad_pose=True,
        suppress_betas_thresh=3.0,
        suppress_kp_conf_thresh=0.3,
        filter_numkp=4,
        filter_reproj_thresh=31000,
        atlas_fitting_thresh=20,
    )


@FILTERS_REGISTRY.register()
def itw_v2():
    # No filtering on bad poses based on prior
    return dict(
        smpl_conf=0.1,
        kps3d_conf=0.1,
        filter_bad_pose=False,
        suppress_betas_thresh=3.0,
        suppress_kp_conf_thresh=0.3,
        filter_numkp=4,
        filter_reproj_thresh=31000,
        atlas_fitting_thresh=20,
    )


@FILTERS_REGISTRY.register()
def itw_v3():
    # More aggressive reduction on pseudo-label loss
    return dict(
        smpl_conf=0.1,
        kps3d_conf=0.0,
        filter_bad_pose=True,
        suppress_betas_thresh=3.0,
        suppress_kp_conf_thresh=0.3,
        filter_numkp=4,
        filter_reproj_thresh=31000,
        atlas_fitting_thresh=20,
    )


@FILTERS_REGISTRY.register()
def itw_v4():
    # More aggressive reduction on pseudo-label loss
    return dict(
        smpl_conf=0.1,
        kps3d_conf=0.01,
        filter_bad_pose=True,
        suppress_betas_thresh=3.0,
        suppress_kp_conf_thresh=0.3,
        filter_numkp=4,
        filter_reproj_thresh=31000,
        atlas_fitting_thresh=20,
    )


@FILTERS_REGISTRY.register()
def hmr2():
    return dict(
        kps3d_conf=0.3,
        filter_bad_pose=True,
        suppress_betas_thresh=3.0,
        suppress_kp_conf_thresh=0.3,
        filter_numkp=4,
        filter_reproj_thresh=31000,
        atlas_fitting_thresh=20,
    )


@FILTERS_REGISTRY.register()
def none():
    return {}
