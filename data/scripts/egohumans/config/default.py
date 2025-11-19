# Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

##---------------------------------------
_C = CN()
_C.SEQUENCE = "001_tagging"
_C.INVALID_ARIAS = []
_C.INVALID_EXOS = []
_C.SEQUENCE_TOTAL_TIME = -1
_C.EXO_CALIBRATION_ROOT = ""

_C.GEOMETRY = CN()
_C.GEOMETRY.MANUAL_GROUND_PLANE_POINTS = ""

_C.CALIBRATION = CN()
_C.CALIBRATION.MANUAL_EXO_CAMERAS = []
_C.CALIBRATION.MANUAL_EGO_CAMERAS = []
_C.CALIBRATION.MANUAL_INTRINSICS_OF_EXO_CAMERAS = []
_C.CALIBRATION.MANUAL_INTRINSICS_FROM_EXO_CAMERAS = []
_C.CALIBRATION.ANCHOR_EGO_CAMERA = (
    "aria01"  ## all cameras are transformed to this camera's world
)


##---------------------------------------
def update_config(cfg, config_file):
    cfg.defrost()
    cfg.merge_from_file(config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
