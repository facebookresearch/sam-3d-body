import os

from . import with_dataroot, OPENPOSE_TO_COCO
from . import DEFAULT_DATA_ROOT

HMR2_KEYPOINTS_3D_LIST = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 43]
HMR2_KEYPOINTS_2D_LIST = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

dataset_cfg = dict(
    threepo_annotations_test_040425_airstore=dict(
        type="topdown_threepo_eval_airstore",
        annot_file="airstore://threepo_annotations_test_040425_no_user_data_rsc",
        img_dir="",
        keypoint_2d_list=OPENPOSE_TO_COCO,
    ),
    ############################################
    threedpw=dict(
        type="topdown_smpl_eval",
        annot_file=with_dataroot("eval_annot/3dpw_test.npz"),
        img_dir=with_dataroot("images/3dpw/"),
        keypoint_3d_list=HMR2_KEYPOINTS_3D_LIST,
        keypoint_2d_list=OPENPOSE_TO_COCO,
        use_hips=False,
        body_model_type="smpl",
    ),
    coco=dict(
        type="topdown_smpl_eval",
        annot_file=with_dataroot("eval_annot/coco_val.npz"),
        img_dir=with_dataroot("images/coco/"),
        keypoint_2d_list=OPENPOSE_TO_COCO,
        use_hips=False,
    ),
    coco_ap=dict(
        type="topdown_smpl_eval",
        annot_file=with_dataroot("eval_annot/coco_val_ap.npz"),
        img_dir=with_dataroot("images/coco/val2017/"),
        keypoint_2d_list=None,
        use_hips=False,
        coco_annot=with_dataroot("images/coco/annotations/person_keypoints_val2017.json"),
    ),
    lspet=dict(
        type="topdown_smpl_eval",
        annot_file=with_dataroot("eval_annot/hr-lspet_train.npz"),
        img_dir=with_dataroot("images/hr-lspet/"),
        keypoint_2d_list=list(range(25, 44)),
        use_hips=False,
    ),
    ochuman_test=dict(
        type="topdown_smpl_eval",
        annot_file=with_dataroot("eval_annot/ochuman_test_ap.npz"),
        img_dir=with_dataroot("images/OCHuman/images"),
        keypoint_2d_list=None,
        use_hips=False,
        coco_annot=with_dataroot("images/OCHuman/annotations/ochuman_coco_format_test_range_0.00_1.00.json"),
    ),
    ochuman_val=dict(
        type="topdown_smpl_eval",
        annot_file=with_dataroot("eval_annot/ochuman_val_ap.npz"),
        img_dir=with_dataroot("images/OCHuman/images"),
        keypoint_2d_list=None,
        use_hips=False,
        coco_annot=with_dataroot("images/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json"),
    ),
    emdb=dict(
        type="topdown_smpl_eval",
        annot_file=with_dataroot("eval_annot/emdb_for_hmr2.npz"),
        img_dir=with_dataroot("images/emdb"),
        keypoint_3d_list=list(range(24)),  # From TokenHMR
        keypoint_2d_list=list(range(25)),
        use_hips=False,
        bbox_scale=200.0,
        body_model_type="smpl",
    ),
    spec=dict(
        type="topdown_smpl_eval",
        annot_file=with_dataroot("eval_annot/spec_test.npz"),
        img_dir=with_dataroot("images/spec-syn"),
        keypoint_3d_list=list(range(24)),  # From TokenHMR
        keypoint_2d_list=list(range(25)),
        use_hips=False,
        bbox_scale=200.0,
        body_model_type="smpl",
    ),
    agora=dict(
        type="topdown_smpl_eval",
        annot_file=with_dataroot("eval_annot/agora_val.npz"),
        img_dir=with_dataroot("images/AGORA_test"),
        keypoint_3d_list=list(range(24)),  # From TokenHMR
        keypoint_2d_list=list(range(25)),
        use_hips=False,
        bbox_scale=200.0,
        body_model_type="smplx",
    ),
    rich=dict(
        type="topdown_smpl_eval",
        annot_file=with_dataroot("eval_annot/rich_test_6fps.npz"),
        img_dir=with_dataroot("images/RICH"),
        keypoint_3d_list=list(range(24)),  # From TokenHMR
        keypoint_2d_list=list(range(25)),
        use_hips=False,
        bbox_scale=200.0,
        body_model_type="smplx",
    ),

    ### Hand-Centric data
    # Interhand datasets
    # Refer to David's paste P1873317025 for some details 
    interhand_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody_val=dict(
        type="threepo_full",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds_jinhyun1/interhand_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody_val/{000000..000338}.tar"),
        filter_empty_annos=True,
        name="interhand_atlas46_250707_val",
        size=1319670,
    ),
    interhand_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody_test=dict(
        type="threepo_full",
        urls=os.path.join(DEFAULT_DATA_ROOT, "wds_jinhyun1/interhand_atlas46_250707_fitting_pseudo_wds_jinhyun1_fullbody_test/{000000..000708}.tar"),
        filter_empty_annos=True,
        name="interhand_atlas46_250707_test",
        size=1319670,
    ),
)
