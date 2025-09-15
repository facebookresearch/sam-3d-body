import numpy as np


def filter_no_atlas(x):
    return (
        len(x["annotation.pyd"]) > 0
        and "atlas_params" in x["annotation.pyd"][0]
        and x["annotation.pyd"][0]["atlas_params"] is not None
        and not np.isnan(x["annotation.pyd"][0]["atlas_params"]["body_pose_params"]).any()
    )

ATLAS70_BODY_IDX = list(range(15)) + [41, 62]


def filter_numkp(x, thresh):
    for person_annot in x["annotation.pyd"]:
        if (person_annot["keypoints_2d"][ATLAS70_BODY_IDX, -1] > 0.25).sum() < thresh:
            return False
    return True
