# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from yacs.config import CfgNode as CN

##---------------------------------------
_C = CN()
_C.INVALID_ARIAS = []
_C.INVALID_EXOS = []
_C.SEQUENCE_TOTAL_TIME = -1

_C.CALIBRATION = CN()
_C.CALIBRATION.ANCHOR_EGO_CAMERA = 'aria01' ## all cameras are transformed to this camera's world

_C.POSE2D = CN()
_C.POSE2D.RGB_THRES = 0.5 
_C.POSE2D.RGB_VIS_THRES = 0.5 
_C.POSE2D.GRAY_THRES = 0.5 
_C.POSE2D.GRAY_VIS_THRES = 0.5 
_C.POSE2D.MIN_VIS_KEYPOINTS = 5
_C.POSE2D.OVERLAP_OKS_THRES = 0.8 

_C.POSE2D.VIS = CN()
_C.POSE2D.VIS.RADIUS = CN()
_C.POSE2D.VIS.RADIUS.EXO_RGB = 10
_C.POSE2D.VIS.RADIUS.EGO_RGB = 5
_C.POSE2D.VIS.RADIUS.EGO_LEFT = 2
_C.POSE2D.VIS.RADIUS.EGO_RIGHT = 2

_C.POSE2D.VIS.THICKNESS = CN()
_C.POSE2D.VIS.THICKNESS.EXO_RGB = 10
_C.POSE2D.VIS.THICKNESS.EGO_RGB = 5
_C.POSE2D.VIS.THICKNESS.EGO_LEFT = 2
_C.POSE2D.VIS.THICKNESS.EGO_RIGHT = 2


##---------------------------------------
def update_config(cfg, config_file):
    cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.freeze()
