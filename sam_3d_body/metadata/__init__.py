# Copyright (c) Meta Platforms, Inc. and affiliates.

import os


OPENPOSE_TO_COCO = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]

# Mapping the J19 used in HMR2.0 to the 14 common points for evaluation
# J19 is defined as the first 19 keypoints in https://github.com/nkolot/SPIN/blob/master/constants.py#L42
# The first 14 keypoints in J19 are LSP keypoints
J19_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]

# Mapping from 14 LSP keypoints to 17 COCO keypoints
# Key: coco_idx, value: lsp_idx
LSP_TO_COCO = {
    5: 9,
    6: 8,
    7: 10,
    8: 7,
    9: 11,
    10: 6,
    11: 3,
    12: 2,
    13: 4,
    14: 1,
    15: 5,
    16: 0,
}

# fmt: off
OPENPOSE_PERMUTATION = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
J19_PERMUTATION = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
COCO_PERMUTATION = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
# fmt: on

# Mapping the 70 MHR keypoints to OpenPose (COCO included)
# key: OpenPose, value: mhr_idx
MHR70_TO_OPENPOSE = {
    0: 0,
    1: 69,
    2: 6,
    3: 8,
    4: 41,
    5: 5,
    6: 7,
    7: 62,
    9: 10,
    10: 12,
    11: 14,
    12: 9,
    13: 11,
    14: 13,
    15: 2,
    16: 1,
    17: 4,
    18: 3,
    19: 15,
    20: 16,
    21: 17,
    22: 18,
    23: 19,
    24: 20,
}

# fmt: off
MHR70_PERMUTATION = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 18, 19, 20, 15, 16, 17, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 64, 63, 66, 65, 68, 67, 69]
# fmt: on
MHR70_TO_LSP = {
    0: 14,
    1: 12,
    2: 10,
    3: 9,
    4: 11,
    5: 13,
    6: 41,
    7: 8,
    8: 6,
    9: 5,
    10: 7,
    11: 62,
    12: 69,
}
