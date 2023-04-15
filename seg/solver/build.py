import torch
from seg.config.config import CfgNode
from seg.utils.optimizer_utils import LinearWarmupCosineAnnealingLR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from seg.solver.edice_loss import EDiceLoss, EDiceLoss_Val
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
from monai.losses import DiceLoss


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    if cfg.SOLVER.OPTIM_ALGO == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg.SOLVER.LR,
                                     weight_decay=cfg.SOLVER.WEIGHT_DEACY)
    elif cfg.SOLVER.OPTIM_ALGO == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg.SOLVER.LR,
                                      weight_decay=cfg.SOLVER.WEIGHT_DEACY)
    elif cfg.SOLVER.OPTIM_ALGO == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.SOLVER.LR,
                                    momentum=cfg.SOLVER.MOMENTUM,
                                    nesterov=True,
                                    weight_decay=cfg.SOLVER.WEIGHT_DEACY)
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(cfg.SOLVER.OPTIM_ALGO))
    return optimizer


def build_lr_schedule(cfg, optimizer):
    """
    Build a LR scheduler from config.
    """
    if cfg.SOLVER.LR_SCHEDULE == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
                                                  max_epochs=cfg.SOLVER.EPOCHS)
    elif cfg.SOLVER.LR_SCHEDULE == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=cfg.SOLVER.EPOCHS)
    elif cfg.SOLVER.LR_SCHEDULE == "multistepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.LR_PARAMS.MILESTONES,
                                                         gamma=cfg.SOLVER.LR_PARAMS.GAMMA)
    else:
        scheduler = None
    return scheduler


def build_loss(cfgs):
    if cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
        return DiceLoss(to_onehot_y=False, sigmoid=True)
    elif cfgs.DATASETS.NAME == "MSD":
        return DiceLoss(to_onehot_y=True, softmax=True)
    if cfgs.SOLVER.LOSS == "BDICE":
        return EDiceLoss().cuda()
    elif cfgs.SOLVER.LOSS == "CEDICE":
        return DiceCELoss(to_onehot_y=True,
                          softmax=True,
                          squared_pred=True,
                          smooth_nr=cfgs.SOLVER.SMOOTH_NR,
                          smooth_dr=cfgs.SOLVER.SMOOTH_DR)


def build_loss_val():
    return EDiceLoss_Val().cuda()


def build_val_post_processing(cfgs):
    post_label = AsDiscrete(to_onehot=cfgs.MODEL.OUT_CHANNEL)
    if cfgs.DATASETS.NAME == "BraTS_2021" or cfgs.DATASETS.NAME == "BraTS_2021_VT_UNET":
        post_pred = Compose(
            [EnsureType(), Activations(sigmoid=True), AsDiscrete(logit_thresh=0.5, threshold_values=True)]
        )
    elif cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
        post_pred = AsDiscrete(argmax=False, logit_thresh=0.5, threshold_values=True)
    elif cfgs.DATASETS.NAME == "MSD":
        if cfgs.DATASETS.MSD_TYPE == "Spleen":
            post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=cfgs.MODEL.OUT_CHANNEL)])
        elif cfgs.DATASETS.MSD_TYPE == "Liver":
            post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=cfgs.MODEL.OUT_CHANNEL)])
        elif cfgs.DATASETS.MSD_TYPE == "Pancreas":
            post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=cfgs.MODEL.OUT_CHANNEL)])
    else:
        post_pred = AsDiscrete(argmax=True,
                               to_onehot=cfgs.MODEL.OUT_CHANNEL)
    if cfgs.DATASETS.NAME == "BraTS_2021" or cfgs.DATASETS.NAME == "BraTS_2021_VT_UNET":
        acc_func = DiceMetric(include_background=True,
                              reduction=MetricReduction.MEAN
                              )
    elif cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
        acc_func = DiceMetric(include_background=True,
                              reduction=MetricReduction.MEAN_BATCH,
                              get_not_nans=True)
    elif cfgs.DATASETS.NAME == "MSD":
        if cfgs.DATASETS.MSD_TYPE == "Pancreas" or cfgs.DATASETS.MSD_TYPE == "Liver":
            acc_func = DiceMetric(include_background=False,
                                  reduction=MetricReduction.MEAN_CHANNEL)
        elif cfgs.DATASETS.MSD_TYPE == "Spleen":
            acc_func = DiceMetric(include_background=False,
                                  reduction=MetricReduction.MEAN)
    else:
        acc_func = DiceMetric(include_background=True,
                              reduction=MetricReduction.MEAN,
                              get_not_nans=True)
    if cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
        post_label = Activations(sigmoid=True)
    elif cfgs.DATASETS.NAME == "MSD":
        if cfgs.DATASETS.MSD_TYPE == "Spleen":
            post_label = Compose([EnsureType(), AsDiscrete(to_onehot=cfgs.MODEL.OUT_CHANNEL)])
        else:
            post_label = AsDiscrete(to_onehot=cfgs.MODEL.OUT_CHANNEL)
    return post_label, post_pred, acc_func