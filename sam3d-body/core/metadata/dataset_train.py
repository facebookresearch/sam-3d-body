import os

from . import DEFAULT_DATA_ROOT


train_cfg = dict(
    egohumans_train_20250827=dict(
        type="threepo_full",
        filter_empty_annos=True,
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_taoshaf/egohumans_train_20250827/{000000..000124}.tar",
        ),
        size=275000,
        name="egohumans_train_20250827",
        metrics=["2d", "3d"],
    ),
    egohumans_train_20250821=dict(
        type="threepo_full",
        filter_empty_annos=True,
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_taoshaf/egohumans_train_20250821/{000000..000124}.tar",
        ),
        size=275000,
        name="egohumans_train_20250821",
        metrics=["2d", "3d"],
    ),
    harmony4d_train_20250815=dict(
        type="threepo_full",
        filter_empty_annos=True,
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_taoshaf/harmony4d_train_20250815/{00000..00199}.tar",
        ),
        size=200000,
        name="harmony4d_train_20250815",
        metrics=["2d", "3d"],
    ),
    egoexo4d_physical_train_20250815=dict(
        type="threepo_full",
        filter_empty_annos=True,
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_taoshaf/egoexo4d_physical_train_20250815/{00000..00399}.tar",
        ),
        size=400000,
        name="egoexo4d_physical_train_20250815",
        metrics=["2d", "3d"],
    ),
    egoexo4d_procedure_train_20250815=dict(
        type="threepo_full",
        filter_empty_annos=True,
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_taoshaf/egoexo4d_procedure_train_20250815/{00000..00199}.tar",
        ),
        size=200000,
        name="egoexo4d_procedure_train_20250815",
        metrics=["2d", "3d"],
    ),
    atlas46_250718_goliath_train_resized_shuffled=dict(
        type="threepo_full",
        urls="/checkpoint/3po/xyang35/data/goliath_train_resized_0813/{000000..000966}.tar",
        size=966120,
        name="atlas46_250718_goliath_train_resized_shuffled",
        metrics=["2d", "3d"],
    ),
    # mask + v4.6; 250827 ############################
    egohumans_train_20250821_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/egohumans_train_20250821/{000000..000124}.tar",
        ),
        size=275000,
        name="egohumans_train_20250821_mask",
    ),
    harmony4d_train_mp_20250815_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/harmony4d_train_mp_20250815/{000000..000199}.tar",
        ),
        size=200000,
        name="harmony4d_train_mp_20250815_mask",
    ),
    # mask + v4.6; 250813 ############################
    harmony4d_train_20250805_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/harmony4d_train_20250805/{000000..000199}.tar",
        ),
        size=285908,
        name="harmony4d_train_20250805_mask",
    ),
    egoexo4d_physical_train_20250803_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/egoexo4d_physical_train_20250803/{000000..000399}.tar",
        ),
        size=285908,
        name="egoexo4d_physical_train_20250803_mask",
    ),
    egoexo4d_procedure_train_20250803_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/egoexo4d_procedure_train_20250803/{000000..000199}.tar",
        ),
        size=285908,
        name="egoexo4d_procedure_train_20250803_mask",
    ),
    atlas46_250718_metasim_scale11_1_7m_body_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/atlas46_250718_metasim_scale11_1_7M_body/{000000..001694}.tar",
        ),
        size=285908,
        name="atlas46_250718_metasim_scale11_1_7m_body_mask",
    ),
    goliath_train_resized_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/goliath_train_resized/{000000..000966}.tar",
        ),
        size=285908,
        name="goliath_train_resized_mask",
    ),
    sa1b_atlas46_250709_fitting_dense_ppv_90_jinhyun1_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/sa1b_atlas46_250709_fitting_dense_ppv_90_jinhyun1/{000000..000184}.tar",
        ),
        size=285908,
        name="sa1b_atlas46_250709_fitting_dense_ppv_90_jinhyun1_mask",
    ),
    atlas46_250729_3dpw_train_full_filtered_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/atlas46_250729_3dpw_train_full_filtered/{000000..000017}.tar",
        ),
        size=285908,
        name="atlas46_250729_3dpw_train_full_filtered_mask",
    ),
    atlas46_250731_agora_train_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/atlas46_250731_agora_train/{000000..000014}.tar",
        ),
        size=285908,
        name="atlas46_250731_agora_train_mask",
    ),
    coco_atlas46_250718_fitting_dense_wds_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/coco_atlas46_250718_fitting_dense_wds_mask/{000000..000023}.tar",
        ),
        size=285908,
        name="coco_atlas46_250718_fitting_dense_wds_mask",
    ),
    mpii_atlas46_250718_fitting_dense_wds_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/mpii_atlas46_250718_fitting_dense_wds_mask/{000000..000005}.tar",
        ),
        size=285908,
        name="mpii_atlas46_250718_fitting_dense_wds_mask",
    ),
    # mask + v4.6; 250626 ############################
    bedlam_atlas46_250626_fitting_jinhyun1_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/bedlam_atlas46_250626_fitting_jinhyun1_mask/{000000..000285}.tar",
        ),
        size=285908,
        name="bedlam_atlas46_250626_fitting_jinhyun1_mask",
    ),
    mpii_atlas46_250626_fitting_jinhyun1_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/mpii_atlas46_250626_fitting_jinhyun1_mask/{000000..000005}.tar",
        ),
        size=5441,
        name="mpii_atlas46_250626_fitting_jinhyun1_mask",
    ),
    coco_atlas46_250626_fitting_jinhyun1_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/coco_atlas46_250626_fitting_jinhyun1_mask/{000000..000023}.tar",
        ),
        size=23800,
        name="coco_atlas46_250626_fitting_jinhyun1_mask",
    ),
    aic_atlas46_250626_fitting_jinhyun1_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/aic_atlas46_250626_fitting_jinhyun1_mask/{000000..000172}.tar",
        ),
        size=172132,
        name="aic_atlas46_250626_fitting_jinhyun1_mask",
    ),
    sa1b_atlas46_250628_fitting_dense_jinhyun1_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/sa1b_atlas46_250628_fitting_dense_jinhyun1_mask/{000000..001840}.tar",
        ),
        size=1849833,  # Not actually, since a good bit don't have GT
        name="sa1b_atlas46_250628_fitting_dense_jinhyun1_mask",
    ),
    # The params in here for hands are wrong btw, but doesn't matter since val is basically just used as a glorified image loader.
    coco_val_atlas46_250626_fitting_nlf_wilor_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/coco-val_atlas46_250626_fitting_nlf+wilor_mask/{000000..000000}.tar",
        ),
        size=1985,
        metrics=["2d"],
        name="coco_val_atlas46_250626_fitting_nlf_wilor_mask",
    ),
    sa1b_atlas46_250628_fitting_val_debug_mask=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_mask/sa1b_atlas46_250628_fitting_dense_jinhyun1_mask/{001841..001849}.tar",
        ),
        size=1849833,  # Not actually, since a good bit don't have GT
        metrics=["2d", "3d"],
        name="sa1b_atlas46_250628_fitting_val_debug_mask",
    ),
    # v4.6; 2508012 ############################
    atlas46_250718_goliath_train_resized=dict(
        type="threepo_full",
        urls="/checkpoint/3po/xyang35/data/goliath_train_resized/{000000..000966}.tar",
        size=966120,
        name="atlas46_250718_goliath_train_resized",
        metrics=["2d", "3d"],
    ),
    atlas46_250729_3dpw_train_sizehack=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250729_3dpw_train/{000000..000017}.tar",
        ),
        size=180000000,
        name="atlas46_250729_3dpw_train",
        metrics=["2d", "3d"],
    ),
    atlas46_250729_3dpw_train_full=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250729_3dpw_train_full/{000000..000017}.tar",
        ),
        size=18000,
        name="atlas46_250729_3dpw_train_full",
        metrics=["2d", "3d"],
    ),
    atlas46_250718_metasim_scale11_1_7m_body=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_scale11_1_7M_body/{000000..001694}.tar",
        ),
        filter_empty_annos=True,
        size=1633795,
        name="atlas46_250718_metasim_scale11_1_7M_body",
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
    ),
    atlas46_250718_metasim_scale11_1_7m_lefthand=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_scale11_1_7M_lefthand/{000000..001694}.tar",
        ),
        filter_empty_annos=True,
        size=1633795,
        name="atlas46_250718_metasim_scale11_1_7M_lefthand",
        metrics=["hand-2d", "hand-3d"],
    ),
    atlas46_250718_metasim_scale11_1_7m_righthand=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_scale11_1_7M_righthand/{000000..001694}.tar",
        ),
        filter_empty_annos=True,
        size=1633795,
        name="atlas46_250718_metasim_scale11_1_7M_righthand",
        metrics=["hand-2d", "hand-3d"],
    ),
    atlas46_250729_3dpw_train_full_filtered=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250729_3dpw_train_full_filtered/{000000..000017}.tar",
        ),
        filter_empty_annos=True,
        name="atlas46_250729_3dpw_train_full_filtered",
        size=170000,
        metrics=["2d"],
    ),
    harmony4d_train_20250805=dict(
        type="threepo_full",
        filter_empty_annos=True,
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/harmony4d_train_20250805/{000000..000199}.tar",
        ),
        size=200000,
        name="harmony4d_train_20250805",
        metrics=["2d", "3d"],
    ),
    egoexo4d_physical_train_20250803=dict(
        type="threepo_full",
        filter_empty_annos=True,
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/egoexo4d_physical_train_20250803/{000000..000399}.tar",
        ),
        size=400000,
        name="egoexo4d_physical_train_20250803",
        metrics=["2d", "3d"],
    ),
    egoexo4d_procedure_train_20250803=dict(
        type="threepo_full",
        filter_empty_annos=True,
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/egoexo4d_procedure_train_20250803/{000000..000199}.tar",
        ),
        size=200000,
        name="egoexo4d_procedure_train_20250803",
        metrics=["2d", "3d"],
    ),
    # v4.6; 250707 ############################
    # This is pseudo-labeled with 2-click 25_07_01_16. Why didn't I do insta?
    # Good question -> we're probably transitioning away from it anyway.
    mpii_atlas46_250707_fitting_pseudo_wds_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/mpii_atlas46_250707_fitting_pseudo_wds_jinhyun1/{000000..000005}.tar",
        ),
        size=5441,
    ),
    coco_atlas46_250707_fitting_pseudo_wds_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/coco_atlas46_250707_fitting_pseudo_wds_jinhyun1/{000000..000023}.tar",
        ),
        size=23800,
    ),
    aic_atlas46_250707_fitting_pseudo_wds_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/aic_atlas46_250707_fitting_pseudo_wds_jinhyun1/{000000..000172}.tar",
        ),
        size=172132,
    ),
    # v4.6; 250626 ############################
    # Newest version; znormed full body scales, new hand
    bedlam_shuffled_atlas46_250626_fitting_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/bedlam_shuffled_atlas46_250626_fitting_jinhyun1/{000000..000285}.tar",
        ),
        size=285908,
        name="bedlam_shuffled_atlas46_250626_fitting_jinhyun1",
    ),
    bedlam_atlas46_250626_fitting_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/bedlam_atlas46_250626_fitting_jinhyun1/{000000..000285}.tar",
        ),
        size=285908,
        name="bedlam_atlas46_250626_fitting_jinhyun1",
    ),
    mpii_atlas46_250626_fitting_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/mpii_atlas46_250626_fitting_jinhyun1/{000000..000005}.tar",
        ),
        size=5441,
        name="mpii_atlas46_250626_fitting_jinhyun1",
    ),
    coco_atlas46_250626_fitting_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/coco_atlas46_250626_fitting_jinhyun1/{000000..000023}.tar",
        ),
        size=23800,
        name="coco_atlas46_250626_fitting_jinhyun1",
    ),
    insta_atlas46_250626_fitting_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/insta_atlas46_250626_fitting_jinhyun1/{000000..001765}.tar",
        ),
        size=1747799,
        name="insta_atlas46_250626_fitting_jinhyun1",
    ),
    aic_atlas46_250626_fitting_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/aic_atlas46_250626_fitting_jinhyun1/{000000..000172}.tar",
        ),
        size=172132,
        name="aic_atlas46_250626_fitting_jinhyun1",
    ),
    ################ 0807: SA1b datasets of different versions ############
    sa1b_atlas46_250628_fitting_dense_jinhyun1=dict(
        type="threepo_full",
        filter_empty_annos=True,  # This one has some empty annotations
        filter_no_atlas=True,  # This one removes those without 3D atlas annotations
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_atlas46_250628_fitting_dense_jinhyun1/{000000..001840}.tar",
        ),
        size=1249718,  # Not actually, since a good bit don't have GT
        name="sa1b_atlas46_250628_fitting_dense_jinhyun1",
    ),
    sa1b_atlas46_250614_fitting_dense_jinhyun1=dict(
        type="threepo_full",
        filter_empty_annos=True,  # This one has some empty annotations
        filter_no_atlas=True,  # This one removes those without 3D atlas annotations
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_atlas46_250614_fitting_dense_jinhyun1/{000000..001840}.tar",
        ),
        size=1249718,  # Not actually, since a good bit don't have GT
        name="sa1b_atlas46_250614_fitting_dense_jinhyun1",
    ),
    sa1b_atlas46_250628_fitting_dense_jinhyun1_cropped768_lanczos_no2donly=dict(
        type="threepo_full",
        filter_empty_annos=True,  # This one has some empty annotations
        filter_no_atlas=True,  # This one removes those without 3D atlas annotations
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_atlas46_250628_fitting_dense_jinhyun1_cropped768_lanczos_no2donly/{000000..001840}.tar",
        ),
        size=1249718,  # Not actually, since a good bit don't have GT
        name="sa1b_atlas46_250628_fitting_dense_jinhyun1_cropped768_lanczos_no2donly",
    ),
    sa1b_atlas46_250628_fitting_dense_full1049=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_atlas46_250628_fitting_dense_jinhyun1/{000000..001849}.tar",
        ),
        size=1249718,  # Not actually, since a good bit don't have GT
        name="sa1b_atlas46_250628_fitting_dense_full1049",
    ),
    sa1b_atlas46_250709_fitting_dense_ppv_150_jinhyun1=dict(
        type="threepo_full",
        filter_empty_annos=True,  # This one has some empty annotations
        filter_no_atlas=True,  # This one removes those without 3D atlas annotations
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_atlas46_250709_fitting_dense_ppv_150_jinhyun1/{000000..000462}.tar",
        ),
        size=1249718,  # Not actually, since a good bit don't have GT
        name="sa1b_atlas46_250709_fitting_dense_ppv_150_jinhyun1",
    ),
    sa1b_atlas46_250709_fitting_dense_ppv_90_jinhyun1=dict(
        type="threepo_full",
        filter_empty_annos=True,  # This one has some empty annotations
        filter_no_atlas=True,  # This one removes those without 3D atlas annotations
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_atlas46_250709_fitting_dense_ppv_90_jinhyun1/{000000..000184}.tar",
        ),
        size=1249718,  # Not actually, since a good bit don't have GT
        name="sa1b_atlas46_250709_fitting_dense_ppv_90_jinhyun1",
    ),
    sa1b_atlas46_250709_fitting_dense_ppv_45_jinhyun1=dict(
        type="threepo_full",
        filter_empty_annos=True,  # This one has some empty annotations
        filter_no_atlas=True,  # This one removes those without 3D atlas annotations
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_atlas46_250709_fitting_dense_ppv_45_jinhyun1/{000000..000092}.tar",
        ),
        size=1249718,  # Not actually, since a good bit don't have GT
        name="sa1b_atlas46_250709_fitting_dense_ppv_45_jinhyun1",
    ),
    sa1b_atlas46_250709_fitting_dense_ppv_5_jinhyun1=dict(
        type="threepo_full",
        filter_empty_annos=True,  # This one has some empty annotations
        filter_no_atlas=True,  # This one removes those without 3D atlas annotations
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_atlas46_250709_fitting_dense_ppv_5_jinhyun1/{000000..000046}.tar",
        ),
        size=1249718,  # Not actually, since a good bit don't have GT
        name="sa1b_atlas46_250709_fitting_dense_ppv_5_jinhyun1",
    ),
    ####################### 0716-0718 fits of COCO, AIC and MPII ########################
    mpii_atlas46_250716_fitting_dense_wds_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/mpii_atlas46_250716_fitting_dense_wds_jinhyun1/{000000..000005}.tar",
        ),
        size=5441,
        name="mpii_atlas46_250716_fitting_dense_wds_jinhyun1",
    ),
    coco_atlas46_250716_fitting_dense_wds_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/coco_atlas46_250716_fitting_dense_wds_jinhyun1/{000000..000023}.tar",
        ),
        size=23800,
        name="coco_atlas46_250716_fitting_dense_wds_jinhyun1",
    ),
    mpii_atlas46_250717_fitting_dense_wds_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/mpii_atlas46_250717_fitting_dense_wds_jinhyun1/{000000..000005}.tar",
        ),
        size=5441,
        name="mpii_atlas46_250717_fitting_dense_wds_jinhyun1",
    ),
    mpii_atlas46_250718_fitting_dense_wds_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/mpii_atlas46_250718_fitting_dense_wds_jinhyun1/{000000..000005}.tar",
        ),
        size=5441,
        name="mpii_atlas46_250718_fitting_dense_wds_jinhyun1",
    ),
    coco_atlas46_250718_fitting_dense_wds_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/coco_atlas46_250718_fitting_dense_wds_jinhyun1/{000000..000023}.tar",
        ),
        size=23800,
        name="coco_atlas46_250718_fitting_dense_wds_jinhyun1",
    ),
    aic_atlas46_250718_fitting_dense_wds_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/aic_atlas46_250718_fitting_dense_wds_jinhyun1/{000000..000172}.tar",
        ),
        size=172132,
        name="aic_atlas46_250718_fitting_dense_wds_jinhyun1",
    ),
    mpii_atlas46_250718_fitting_dense_wds_updatekps2d_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/mpii_atlas46_250718_fitting_dense_wds_updatekps2d_jinhyun1/{000000..000005}.tar",
        ),
        size=5441,
        name="mpii_atlas46_250718_fitting_dense_wds_updatekps2d_jinhyun1",
    ),
    coco_atlas46_250718_fitting_dense_wds_updatekps2d_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/coco_atlas46_250718_fitting_dense_wds_updatekps2d_jinhyun1/{000000..000023}.tar",
        ),
        size=23800,
        name="coco_atlas46_250718_fitting_dense_wds_updatekps2d_jinhyun1",
    ),
    aic_atlas46_250718_fitting_dense_wds_updatekps2d_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/aic_atlas46_250718_fitting_dense_wds_updatekps2d_jinhyun1/{000000..000172}.tar",
        ),
        size=172132,
        name="aic_atlas46_250718_fitting_dense_wds_updatekps2d_jinhyun1",
    ),
    ##############################################
    threepo_train_100_0614=dict(
        type="threepo_full_airstore",
        airstore_uri="airstore://threepo_train_100_atlas_no_user_data_rsc",
        ingestion_id=99866,
        size=1400000,
    ),
    threepo_train_100_0614_downscaled=dict(
        type="threepo_full_airstore",
        airstore_uri="airstore://threepo_train_100_atlas_downscaled_no_user_data_rsc",
        size=1400000,
    ),
    threepo_train_100_0518=dict(
        type="threepo_full_airstore",
        airstore_uri="airstore://threepo_train_100_atlas_no_user_data_rsc",
        ingestion_id=97028,
        size=1000000,
    ),
    threepo_train_100_atlas_no_user_data_rsc=dict(
        type="threepo_full_airstore",
        airstore_uri="airstore://threepo_train_100_atlas_no_user_data_rsc",
        size=549249,
    ),
    threepo_val_atlas_no_user_data_rsc=dict(
        type="threepo_full_airstore",
        airstore_uri="airstore://threepo_val_atlas_no_user_data_rsc",
        size=61012,
    ),
    sa1b_atlas3_250628_fitting_dense_jinhyun1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_atlas3_250628_dense/{000000..001839}.tar",
        ),
        size=1849833,
    ),
    bedlam_camerahmr_full=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/bedlam_camerahmr/{000000..000285}.tar"
        ),
        size=285908,
    ),
    bedlam_camerahmr=dict(
        type="topdown_atlas",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/bedlam_camerahmr/{000000..000285}.tar"
        ),
        size=285908,
        metrics=["2d", "3d"],
    ),
    mpii_camerahmr_full=dict(
        type="threepo_full",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/mpii_camerahmr/{000000..000005}.tar"),
        size=5441,
    ),
    mpii_camerahmr=dict(
        type="topdown_atlas",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/mpii_camerahmr/{000000..000005}.tar"),
        size=5441,
    ),
    coco_camerahmr_full=dict(
        type="threepo_full",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/coco_camerahmr/{000000..000023}.tar"),
        size=23800,
    ),
    coco_camerahmr=dict(
        type="topdown_atlas",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/coco_camerahmr/{000000..000023}.tar"),
        size=23800,
    ),
    aic_camerahmr_full=dict(
        type="threepo_full",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/aic_camerahmr/{000000..000172}.tar"),
        size=172485,
    ),
    aic_camerahmr=dict(
        type="topdown_atlas",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/aic_camerahmr/{000000..000172}.tar"),
        size=172485,
    ),
    insta_camerahmr_full=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/insta_camerahmr/{000000..001765}.tar"
        ),
        size=1766000,
    ),
    insta_camerahmr=dict(
        type="topdown_atlas",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/insta_camerahmr/{000000..001765}.tar"
        ),
        size=1766000,
    ),
    #########################################################################################
    ## airstore variants ##
    goliath_atlasv1_full_airstore=dict(  # Downsampled version
        type="threepo_full_airstore",
        airstore_uri="airstore://threepo_goliath_atlas_train_041625_no_user_data_rsc",
        size=1016395,
    ),
    goliath_atlasv1_val_full_airstore=dict(  # Downsampled version
        type="threepo_full_airstore",
        airstore_uri="airstore://threepo_goliath_atlas_val_041625_no_user_data_rsc",
        size=1016395,
    ),
    goliath_atlasv1_airstore=dict(  # Downsampled version
        type="topdown_atlas_airstore",
        airstore_uri="airstore://threepo_goliath_atlas_train_041625_no_user_data_rsc",
        size=1016395,
    ),
    goliath_atlasv1_val_airstore=dict(  # Downsampled version
        type="topdown_atlas_airstore",
        airstore_uri="airstore://threepo_goliath_atlas_val_041625_no_user_data_rsc",
        size=1016395,
    ),
    bedlam_atlasv1=dict(
        type="topdown_atlas",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/bedlam_atlasv1/{000000..000285}.tar"),
        size=285919,
    ),
    mpi_atlasv1=dict(
        type="topdown_atlas",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/mpi-inf_atlasv1/{000000..000012}.tar"
        ),
        size=12251,
    ),
    mpii_atlasv1=dict(
        type="topdown_atlas",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/mpii_atlasv1/{000000..000009}.tar"),
        size=9899,
    ),
    aic_atlasv1=dict(
        type="topdown_atlas",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/aic_atlasv1/{000000..000209}.tar"),
        size=209827,
    ),
    threedpw_atlasv1=dict(
        type="topdown_atlas",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/threedpw_atlasv1/{000000..000023}.tar"
        ),
        size=23177,
    ),
    coco_atlasv1=dict(
        type="topdown_atlas",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/coco_atlasv1/{000000..000046}.tar"),
        size=46185,
    ),
    threedpw_val_atlasv1=dict(
        type="topdown_atlas",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/threedpw-val_atlasv1/{000000..000000}.tar"
        ),
        size=2326,
    ),
    threedpw_val_atlasv1_cam=dict(
        type="topdown_atlas",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/threedpw-val_atlasv1_cam/{000000..000000}.tar"
        ),
        size=23177,
    ),
    coco_val_atlasv1=dict(
        type="topdown_atlas",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/coco-val_atlasv1/{000000..000000}.tar"
        ),
        size=1985,
    ),
    coco_val_atlasv1_cam=dict(
        type="topdown_atlas",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/coco-val_atlasv1_cam/{000000..000000}.tar"
        ),
        size=1985,
    ),
    ########################################################################################
    bedlam_1223=dict(
        type="topdown_smpl",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/bedlam_1223/{000000..000285}.tar"),
        size=285919,
    ),
    mpi_1223=dict(
        type="topdown_smpl",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/mpi_1223/{000000..000012}.tar"),
        size=12251,
    ),
    coco_1223=dict(
        type="topdown_smpl",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/coco_1223/{000000..000017}.tar"),
        size=17747,
    ),
    mpii_1223=dict(
        type="topdown_smpl",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/mpii_1223/{000000..000009}.tar"),
        size=9899,
    ),
    coco_pseudo_1223=dict(
        type="topdown_smpl",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/coco_pseudo_1223/{000000..000044}.tar"
        ),
        size=44200,
    ),
    aic_1223=dict(
        type="topdown_smpl",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/aic_1223/{000000..000209}.tar"),
        size=209827,
    ),
    insta_1223=dict(
        type="topdown_smpl",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/insta_1223/{000000..003656}.tar"),
        size=3656992,
    ),
    coco_val=dict(
        type="topdown_smpl",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds/coco_val/{000000..000000}.tar"),
    ),
    cmu_mocap=dict(
        dataset_file=os.path.join(DEFAULT_DATA_ROOT, "misc/cmu_mocap.npz"),
    ),
    ## airstore variants ##
    bedlam_1223_airstore=dict(
        type="topdown_smpl_airstore",
        airstore_uri="airstore://threepo_bedlam_1223_no_user_data_rsc",
        size=285919,
    ),
    mpi_1223_airstore=dict(
        type="topdown_smpl_airstore",
        airstore_uri="airstore://threepo_mpi_1223_no_user_data_rsc",
        size=12251,
    ),
    coco_1223_airstore=dict(
        type="topdown_smpl_airstore",
        airstore_uri="airstore://threepo_coco_1223_no_user_data_rsc",
        size=17747,
    ),
    mpii_1223_airstore=dict(
        type="topdown_smpl_airstore",
        airstore_uri="airstore://threepo_mpii_1223_no_user_data_rsc",
        size=9899,
    ),
    coco_pseudo_1223_airstore=dict(
        type="topdown_smpl_airstore",
        airstore_uri="airstore://threepo_coco_pseudo_1223_no_user_data_rsc",
        size=44200,
    ),
    aic_1223_airstore=dict(
        type="topdown_smpl_airstore",
        airstore_uri="airstore://threepo_aic_1223_no_user_data_rsc",
        size=209827,
    ),
    ## 3PO annotation exports ##
    threepo_annotations_100_2025_03_25_airstore=dict(
        type="topdown_smpl_airstore",
        airstore_uri="airstore://threepo_annotations_100_2025_03_25_no_user_data_rsc",
        size=549249,
    ),
    threepo_annotations_val_2025_03_25_airstore=dict(
        type="topdown_smpl_airstore",
        airstore_uri="airstore://threepo_annotations_val_2025_03_25_no_user_data_rsc",
        size=61012,
    ),
    ##### Hand-Centric Datasets #####
    # v4.6; 250707 hand #######################
    # Refer to David's paste P1865071861 for some details on the dataset.
    dexycb_atlas46_250707_train_s1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/dexycb_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody/{000000..000145}.tar",
        ),
        filter_empty_annos=True,
        name="dexycb_atlas46_250707_train_s1",
        dexycb_split="s1",
        size=145472,
    ),
    dexycb_atlas46_250707_train_s2=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/dexycb_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody/{000000..000145}.tar",
        ),
        filter_empty_annos=True,
        name="dexycb_atlas46_250707_train_s2",
        dexycb_split="s2",
        size=145472,
    ),
    mpii_atlas46_newfits_0718=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/mpii_atlas46_250718_fitting_dense_wds_withhands_jinhyun1/{000000..000005}.tar",
        ),
        filter_empty_annos=True,
        name="mpii_atlas46_newfits_0718",
        size=145472,
    ),
    coco_atlas46_newfits_0718=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/coco_atlas46_250718_fitting_dense_wds_withhands_jinhyun1/{000000..000023}.tar",
        ),
        filter_empty_annos=True,
        name="coco_atlas46_newfits_0718",
        size=145472,
    ),
    # Interhand datasets
    # Refer to David's paste P1873317025 for some details
    interhand_atlas46_250707_train=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/interhand_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody_train/{000000..001319}.tar",
        ),
        filter_empty_annos=True,
        name="interhand_atlas46_250707_train",
        size=1319670,
    ),
    atlas46_250718_metasim_400k_body=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_400k_body/{000000..000402}.tar",
        ),
        filter_empty_annos=True,
        size=402000,
        name="atlas46_250718_metasim_400k_body",
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
    ),
    atlas46_250718_metasim_400k_righthand=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_400k_righthand/{000000..000402}.tar",
        ),
        filter_empty_annos=True,
        name="atlas46_250718_metasim_400k_righthand",
        size=402000,
    ),
    atlas46_250718_metasim_400k_lefthand=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_400k_lefthand/{000000..000402}.tar",
        ),
        filter_empty_annos=True,
        name="atlas46_250718_metasim_400k_lefthand",
        size=402000,
    ),
    atlas46_250718_metasim_scale11_nomaxscale_1m_body=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_scale11_nomaxscale_1M_body/{000000..001125}.tar",
        ),
        filter_empty_annos=True,
        size=402000,
        name="atlas46_250718_metasim_scale11_nomaxscale_1M_body",
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
    ),
    atlas46_250718_metasim_scale11_nomaxscale_1m_righthand=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_scale11_nomaxscale_1M_righthand/{000000..001125}.tar",
        ),
        filter_empty_annos=True,
        size=402000,
        name="atlas46_250718_metasim_scale11_nomaxscale_1M_righthand",
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
    ),
    atlas46_250718_metasim_scale11_nomaxscale_1m_lefthand=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_scale11_nomaxscale_1M_lefthand/{000000..001125}.tar",
        ),
        filter_empty_annos=True,
        size=402000,
        name="atlas46_250718_metasim_scale11_nomaxscale_1M_lefthand",
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
    ),
    atlas46_250731_agora_train=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250731_agora_train/{000000..000014}.tar",
        ),
        name="atlas46_250731_agora_train",
        size=1000000,
    ),
    atlas46_250801_arctic_1e5_all_stage1sub15_body=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250801_arctic_1e5_all_stage1sub15_body_fixed/{000000..001731}.tar",
        ),
        filter_empty_annos=True,
        size=1000000,
        name="atlas46_250801_arctic_1e5_all_stage1sub15_body",
    ),
    atlas46_250801_arctic_1e5_all_stage1sub15_lefthand=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250801_arctic_1e5_all_stage1sub15_lefthand_fixed/{000000..001731}.tar",
        ),
        filter_empty_annos=True,
        size=1000000,
        name="atlas46_250801_arctic_1e5_all_stage1sub15_lefthand",
    ),
    atlas46_250801_arctic_1e5_all_stage1sub15_righthand=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250801_arctic_1e5_all_stage1sub15_righthand_fixed/{000000..001731}.tar",
        ),
        filter_empty_annos=True,
        size=1000000,
        name="atlas46_250801_arctic_1e5_all_stage1sub15_righthand",
    ),
    atlas46_250718_metasim_scale11_1m_body=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_scale11_1M_body/{000000..001125}.tar",
        ),
        filter_empty_annos=True,
        size=1125000,
        name="atlas46_250718_metasim_scale11_1m_body",
        metrics=["2d", "3d"],
    ),
    atlas46_250718_metasim_1m_body=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_1M_body/{000000..001125}.tar",
        ),
        filter_empty_annos=True,
        size=1125000,
        name="atlas46_250718_metasim_1m_body",
        metrics=["2d", "3d"],
    ),
)

############################################################################################

val_cfg = dict(
    coco_val_atlasv1_fov=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds/coco-val_atlasv1_cam/{000000..000000}.tar",
        ),
        name="coco_val_atlasv1_fov",
        size=2000,
    ),
    threedpw_val_atlasv1_fov=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds/threedpw-val_atlasv1_cam/{000000..000000}.tar",
        ),
        name="threedpw_val_atlasv1_fov",
        size=2000,
    ),
    sa1b_atlas46_250628_fitting_val_debug=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_atlas46_250628_fitting_dense_jinhyun1/{001841..001849}.tar",
        ),
        size=1849833,  # Not actually, since a good bit don't have GT
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
        name="sa1b_atlas46_250628_fitting_val_debug",
    ),
    threedpw_val_atlasv1_full=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/threedpw-val_atlasv1/{000000..000000}.tar"
        ),
        size=2326,
        metrics=["2d"],
    ),
    coco_val_atlasv1_full=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT, "wds/coco-val_atlasv1/{000000..000000}.tar"
        ),
        size=1985,
        metrics=["2d"],
    ),
    dexycb_atlas46_250707_val_s1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/dexycb_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody/{000087..000101}.tar",
        ),  # 72-88 tar files, roughly contains the subject #6
        filter_empty_annos=True,
        name="dexycb_atlas46_250707_val_s1",
        dexycb_split="s1",
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
        size=20000,
    ),
    dexycb_atlas46_250707_test_s1=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/dexycb_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody/{000000..000145}.tar",
        ),
        filter_empty_annos=True,
        name="dexycb_atlas46_250707_test_s1",
        dexycb_split="s1",
        size=145472,
    ),
    dexycb_atlas46_250707_val_s2=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/dexycb_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody/{000000..000145}.tar",
        ),
        filter_empty_annos=True,
        name="dexycb_atlas46_250707_val_s2",
        dexycb_split="s2",
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
        size=145472,
    ),
    dexycb_atlas46_250707_test_s2=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/dexycb_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody/{000000..000145}.tar",
        ),
        filter_empty_annos=True,
        name="dexycb_atlas46_250707_test_s2",
        dexycb_split="s2",
        size=145472,
    ),
    sa1b_atlas46_250628_fitting_dense_val=dict(
        type="threepo_full",
        filter_empty_annos=False,
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/sa1b_250726_valtest/val_000000.tar",
        ),
        size=1849833,  # Not actually, since a good bit don't have GT
        metrics=["2d"],
        name="sa1b_atlas46_250628_fitting_dense_val",
    ),
    # The params in here for hands are wrong btw, but doesn't matter since val is basically just used as a glorified image loader.
    coco_val_atlas46_250626_fitting_nlf_wilor=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/coco-val_atlas46_250626_fitting_nlf+wilor/{000000..000000}.tar",
        ),
        size=1985,
        metrics=["2d"],
        name="coco_val_atlas46_250626_fitting_nlf_wilor",
    ),
    interhand_atlas46_250707_val=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/interhand_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody_val/{000000..000338}.tar",
        ),
        filter_empty_annos=True,
        name="interhand_atlas46_250707_val",
        size=1319670,
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
    ),
    interhand_atlas46_250707_test=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/interhand_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody_test/{000000..000708}.tar",
        ),
        filter_empty_annos=True,
        name="interhand_atlas46_250707_test",
        size=1319670,
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
    ),
    atlas46_250718_metasim_1m_body_test=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_scale11_nomaxscale_1M_body_test/{000000..000013}.tar",
        ),
        filter_empty_annos=True,
        size=1000000,
        name="atlas46_250718_metasim_1m_body_test",
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
    ),
    atlas46_250718_metasim_1m_righthand_test=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_scale11_nomaxscale_1M_righthand_test/{000000..000013}.tar",
        ),
        filter_empty_annos=True,
        size=1000000,
        name="atlas46_250718_metasim_1m_righthand_test",
        metrics=["hand-2d", "hand-3d"],
    ),
    atlas46_250718_metasim_1m_lefthand_test=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250718_metasim_scale11_nomaxscale_1M_lefthand_test/{000000..000013}.tar",
        ),
        filter_empty_annos=True,
        size=1000000,
        name="atlas46_250718_metasim_1m_lefthand_test",
        metrics=["hand-2d", "hand-3d"],
    ),
    atlas46_250731_challenging_interhand=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/interhand_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody_test/fps_256x4_verybadbuggytellpark.tar",
        ),
        filter_empty_annos=True,
        name="atlas46_250731_challenging_interhand",
        size=170000,
        metrics=["hand-2d", "hand-3d"],
    ),
    atlas46_250801_arctic_1e5_all_stage1sub15_body_val=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250801_arctic_1e5_all_stage1sub15_body_fixed_val/{000000..000232}.tar",
        ),
        filter_empty_annos=True,
        size=1000000,
        name="atlas46_250801_arctic_1e5_all_stage1sub15_body_val",
        metrics=["hand-2d", "hand-3d"],
    ),
    atlas46_250801_arctic_1e5_all_stage1sub15_lefthand_val=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250801_arctic_1e5_all_stage1sub15_lefthand_fixed_val/{000000..000232}.tar",
        ),
        filter_empty_annos=True,
        size=1000000,
        name="atlas46_250801_arctic_1e5_all_stage1sub15_lefthand_val",
        metrics=["hand-2d", "hand-3d"],
    ),
    atlas46_250801_arctic_1e5_all_stage1sub15_righthand_val=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250801_arctic_1e5_all_stage1sub15_righthand_fixed_val/{000000..000232}.tar",
        ),
        filter_empty_annos=True,
        size=1000000,
        name="atlas46_250801_arctic_1e5_all_stage1sub15_righthand_val",
        metrics=["hand-2d", "hand-3d"],
    ),
    atlas46_250731_agora_val=dict(
        type="threepo_full",
        urls=os.path.join(
            DEFAULT_DATA_ROOT,
            "wds_jinhyun1/atlas46_250731_agora_val/{000000..000000}.tar",
        ),
        name="atlas46_250731_agora_val",
        size=1000000,
        metrics=["2d", "3d", "hand-2d", "hand-3d"],
    ),
)

dataset_cfg = {**train_cfg, **val_cfg}
