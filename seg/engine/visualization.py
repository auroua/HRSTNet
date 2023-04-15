import os
import time
import shutil
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from seg.utils.dist_utils import distributed_all_gather
import seg.utils.dist_utils as dist_utils
import torch.utils.data.distributed
from monai.data import decollate_batch
from datasets.builder import get_dataloader
from seg.models.builder import get_model
import logging
from seg.solver.build import build_optimizer, build_lr_schedule, build_loss, build_val_post_processing, build_loss_val
from seg.engine.defaults import create_ddp_model
from seg.checkpoint.seg_checkpoint import SegCheckpointer
from functools import partial
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from monai import transforms, data
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from monai.data import load_decathlon_datalist
from monai.transforms import LoadImage
from seg.utils.solver_utils import dice, AverageMeter
from seg.utils.solver_utils import hd
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
from monai.utils.enums import MetricReduction
import nibabel as nib
from einops import rearrange
import SimpleITK as sitk
from numpy import logical_and as l_and, logical_not as l_not
import pprint
import pandas as pd
import glob


def visualization_3d(cfgs):
    train_images = sorted(
        glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(cfgs.INPUT.SPACING.X, cfgs.INPUT.SPACING.Y, cfgs.INPUT.SPACING.Z),
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MIN,
                a_max=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MAX,
                b_min=cfgs.INPUT.SCALE_INTENSITY.TARGET_MIN,
                b_max=cfgs.INPUT.SCALE_INTENSITY.TARGET_MAX,
                clip=True,
            ),
            transforms.EnsureTyped(keys=["image", "label"]),
        ]
    )

    if cfgs.DATASETS.NAME == "MSD":
        if cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas" \
                or cfgs.DATASETS.MSD_TYPE == "Spleen":
            train_files, val_files, test_files = data_dicts[:cfgs.DATASETS.MSD.TRAIN_NUM], \
                                                 data_dicts[cfgs.DATASETS.MSD.TRAIN_NUM: cfgs.DATASETS.MSD.VAL_NUM], \
                                                 data_dicts[cfgs.DATASETS.MSD.VAL_NUM:]
    test_ds = data.Dataset(data=test_files, transform=val_transforms)
    test_loader = data.DataLoader(test_ds,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                  sampler=None,
                                  pin_memory=True,
                                  persistent_workers=True)
    post_transforms = transforms.Compose([
        transforms.EnsureTyped(keys="pred"),
        transforms.Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        )
    ])

    with torch.no_grad():
        for test_data in test_loader:
            test_data["image"] = test_data["image"].cuda()
            img_name = test_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))
            test_data["pred"] = sliding_window_inference(test_data["image"],
                                                         roi_size=(cfgs.INPUT.RAND_CROP.ROI.X,
                                                                   cfgs.INPUT.RAND_CROP.ROI.Y,
                                                                   cfgs.INPUT.RAND_CROP.ROI.Z),
                                                         sw_batch_size=cfgs.SOLVER.SW_BATCH_SIZE_TEST,
                                                         predictor=self.model,
                                                         overlap=self.cfg.SOLVER.INFER_OVERLAP)
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]