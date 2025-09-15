from fvcore.common.registry import Registry

from .base.threepo_full_train import get_threepo_full_wds
from .base.topdown_atlas_train import get_topdown_atlas_wds

DATASET_REGISTRY = Registry("DATASET")

@DATASET_REGISTRY.register()
def fov_train(cfg, dataset_cfg, **kwargs):
    from .base.fov_train import get_fov_train_wds
    
    return get_fov_train_wds(cfg, dataset_cfg, **kwargs)

@DATASET_REGISTRY.register()
def threepo_full_eval_wds(dataset_cfg, model_cfg, eval_cfg, annot=None):
    from .base.threepo_full_eval import get_threepo_full_eval_wds

    return get_threepo_full_eval_wds(dataset_cfg, model_cfg, eval_cfg)

@DATASET_REGISTRY.register()
def mp_full_eval(dataset_cfg, model_cfg, eval_cfg, annot=None):
    from .base.mp_full_eval import MPFullEvalDataset

    return MPFullEvalDataset(dataset_cfg, model_cfg, eval_cfg, annot)

@DATASET_REGISTRY.register()
def phmr_full_eval(dataset_cfg, model_cfg, eval_cfg, annot=None):
    from .base.phmr_full_eval import MPFullEvalDataset

    return MPFullEvalDataset(dataset_cfg, model_cfg, eval_cfg, annot)


@DATASET_REGISTRY.register()
def threepo_full_airstore_eval(dataset_cfg, model_cfg, eval_cfg, annot=None):
    from .base.threepo_full_airstore_eval import ThreePOAIRStoreEvalDataset

    return ThreePOAIRStoreEvalDataset(
        dataset_cfg, model_cfg=model_cfg, eval_cfg=eval_cfg
    )


@DATASET_REGISTRY.register()
def threepo_full_eval(dataset_cfg, model_cfg, eval_cfg, annot=None):
    from .base.threepo_full_eval import ThreePOFullEvalDataset

    return ThreePOFullEvalDataset(dataset_cfg, model_cfg, eval_cfg, annot)


@DATASET_REGISTRY.register()
def threepo_full_airstore(cfg, dataset_cfg, mode):
    from .base.threepo_full_airstore_train import ThreePOAIRStoreTrainDataset

    return ThreePOAIRStoreTrainDataset(dataset_cfg, cfg=cfg, mode=mode)


@DATASET_REGISTRY.register()
def threepo_full(cfg, dataset_cfg, **kwargs):
    return get_threepo_full_wds(cfg, dataset_cfg, **kwargs)
