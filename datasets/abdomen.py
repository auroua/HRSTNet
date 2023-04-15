# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from monai import transforms, data
from monai.data import load_decathlon_datalist
from seg.utils.dist_utils import Sampler
from seg.utils.dist_utils import get_world_size
import numpy as np
import random
import json
import glob
import pathlib
from datasets.vt_unet_brats.brats import Brats
import torch


def get_loader_abdomen(cfgs):
    data_dir = cfgs.DATASETS.DATA_DIR
    datalist_json = os.path.join(data_dir, cfgs.DATASETS.JSON)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"],
                                    axcodes=cfgs.INPUT.ORIENTATION),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(cfgs.INPUT.SPACING.X, cfgs.INPUT.SPACING.Y, cfgs.INPUT.SPACING.Z),
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MIN,
                                            a_max=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MAX,
                                            b_min=cfgs.INPUT.SCALE_INTENSITY.TARGET_MIN,
                                            b_max=cfgs.INPUT.SCALE_INTENSITY.TARGET_MAX,
                                            clip=True),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"],
                                   spatial_size=(cfgs.INPUT.RAND_CROP.ROI.X, cfgs.INPUT.RAND_CROP.ROI.Y, cfgs.INPUT.RAND_CROP.ROI.Z)
                                   ),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(cfgs.INPUT.RAND_CROP.ROI.X, cfgs.INPUT.RAND_CROP.ROI.Y, cfgs.INPUT.RAND_CROP.ROI.Z),
                pos=cfgs.INPUT.RAND_CROP.POS,
                neg=cfgs.INPUT.RAND_CROP.NEG,
                num_samples=cfgs.INPUT.RAND_CROP.SAMPLES,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=cfgs.INPUT.RAND.FLIP_AXIS_PROB,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=cfgs.INPUT.RAND.FLIP_AXIS_PROB,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=cfgs.INPUT.RAND.FLIP_AXIS_PROB,
                                 spatial_axis=2),
            transforms.RandRotate90d(
                keys=["image", "label"],
                prob=cfgs.INPUT.RAND.ROTATE90_PROB,
                max_k=3,
            ),
            transforms.RandScaleIntensityd(keys="image",
                                           factors=0.1,
                                           prob=cfgs.INPUT.RAND.SCALE_INTENSITY_PROB),
            transforms.RandShiftIntensityd(keys="image",
                                           offsets=0.1,
                                           prob=cfgs.INPUT.RAND.SHIFT_INTENSITY_PROB),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"],
                                    axcodes=cfgs.INPUT.ORIENTATION),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(cfgs.INPUT.SPACING.X, cfgs.INPUT.SPACING.Y, cfgs.INPUT.SPACING.Z),
                                mode=("bilinear", "nearest")),
            # transforms.Spacingd(keys="image",
            #                     pixdim=(cfgs.INPUT.SPACING.X, cfgs.INPUT.SPACING.Y, cfgs.INPUT.SPACING.Z),
            #                     mode="bilinear"),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MIN,
                                            a_max=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MAX,
                                            b_min=cfgs.INPUT.SCALE_INTENSITY.TARGET_MIN,
                                            b_max=cfgs.INPUT.SCALE_INTENSITY.TARGET_MAX,
                                            clip=True),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if cfgs.MODE == "test":
        test_files = load_decathlon_datalist(datalist_json,
                                             True,
                                             cfgs.DATASETS.TEST_TYPE,
                                             base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if get_world_size() > 1 else None
        test_loader = data.DataLoader(test_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                      sampler=test_sampler,
                                      pin_memory=True,
                                      persistent_workers=True)
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json,
                                           True,
                                           "training",
                                           base_dir=data_dir)
        if cfgs.DATASETS.TYPE == "CACHE":
            train_ds = data.CacheDataset(
                data=datalist,
                transform=train_transform,
                cache_num=cfgs.DATASETS.CACHE.NUM,
                cache_rate=cfgs.DATASETS.CACHE.RATE,
                num_workers=cfgs.DATALOADER.TRAIN_WORKERS,
            )
        else:
            train_ds = data.Dataset(data=datalist, transform=train_transform)

        train_sampler = Sampler(train_ds) if get_world_size() > 1 else None
        train_loader = data.DataLoader(train_ds,
                                       batch_size=cfgs.SOLVER.BATCH_SIZE,
                                       shuffle=(train_sampler is None),
                                       num_workers=cfgs.DATALOADER.TRAIN_WORKERS,
                                       sampler=train_sampler,
                                       pin_memory=True,
                                       persistent_workers=True)
        val_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "validation",
                                            base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if get_world_size() > 1 else None
        val_loader = data.DataLoader(val_ds,
                                     batch_size=cfgs.SOLVER.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        loader = [train_loader, val_loader]

    return loader


def get_loader_brats(cfgs):
    data_dir = cfgs.DATASETS.DATA_DIR
    datalist_json = os.path.join(data_dir, cfgs.DATASETS.JSON)
    train_transform = transforms.Compose(
        [
            # load 4 Nifti images and stack them together
            transforms.LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"]),
            transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
            MinMaxNormalization(keys=["t1", "t1ce", "t2", "flair"]),
            CropForegroundCombined(keys=["t1", "t1ce", "t2", "flair", "seg"],
                                   roi_size=(cfgs.INPUT.RAND_CROP.ROI.X,
                                             cfgs.INPUT.RAND_CROP.ROI.Y,
                                             cfgs.INPUT.RAND_CROP.ROI.Z)
                                   ),
            transforms.RandCropByPosNegLabeld(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                label_key="seg",
                spatial_size=(cfgs.INPUT.RAND_CROP.ROI.X, cfgs.INPUT.RAND_CROP.ROI.Y, cfgs.INPUT.RAND_CROP.ROI.Z),
                pos=cfgs.INPUT.RAND_CROP.POS,
                neg=cfgs.INPUT.RAND_CROP.NEG,
                num_samples=cfgs.INPUT.RAND_CROP.SAMPLES,
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["t1", "t1ce", "t2", "flair", "seg"],
                                 prob=cfgs.INPUT.RAND.FLIP_AXIS_PROB,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["t1", "t1ce", "t2", "flair", "seg"],
                                 prob=cfgs.INPUT.RAND.FLIP_AXIS_PROB,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["t1", "t1ce", "t2", "flair", "seg"],
                                 prob=cfgs.INPUT.RAND.FLIP_AXIS_PROB,
                                 spatial_axis=2),
            transforms.RandRotate90d(
                keys=["t1", "t1ce", "t2", "flair", "seg"],
                prob=cfgs.INPUT.RAND.ROTATE90_PROB,
                max_k=3,
            ),
            transforms.RandScaleIntensityd(keys=["t1", "t1ce", "t2", "flair"],
                                           factors=0.1,
                                           prob=cfgs.INPUT.RAND.SCALE_INTENSITY_PROB),
            transforms.RandShiftIntensityd(keys=["t1", "t1ce", "t2", "flair"],
                                           offsets=0.1,
                                           prob=cfgs.INPUT.RAND.SCALE_INTENSITY_PROB),
            transforms.ToTensord(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["t1", "t1ce", "t2", "flair", "seg"]),
            transforms.EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
            MinMaxNormalization(keys=["t1", "t1ce", "t2", "flair"]),
            transforms.ToTensord(keys=["t1", "t1ce", "t2", "flair", "seg"]),
        ]
    )
    if cfgs.MODE == "train":
        # here we don't cache any data in case out of memory issue
        datalist = load_decathlon_datalist(datalist_json,
                                           True,
                                           "training",
                                           base_dir=data_dir)
        if cfgs.DATASETS.TYPE == "CACHE":
            train_ds = data.CacheDataset(
                data=datalist,
                transform=train_transform,
                cache_num=cfgs.DATASETS.CACHE.NUM,
                cache_rate=cfgs.DATASETS.CACHE.RATE,
                num_workers=cfgs.DATALOADER.TRAIN_WORKERS,
            )
        else:
            train_ds = data.Dataset(data=datalist, transform=train_transform)

        train_sampler = Sampler(train_ds) if get_world_size() > 1 else None
        train_loader = data.DataLoader(train_ds,
                                       batch_size=cfgs.SOLVER.BATCH_SIZE,
                                       shuffle=(train_sampler is None),
                                       num_workers=cfgs.DATALOADER.TRAIN_WORKERS,
                                       sampler=train_sampler,
                                       pin_memory=True,
                                       persistent_workers=True)
        val_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "validation",
                                            base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if get_world_size() > 1 else None
        val_loader = data.DataLoader(val_ds,
                                     batch_size=cfgs.SOLVER.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        loader = [train_loader, val_loader]

        return loader
    elif cfgs.MODE == "test":
        val_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "test",
                                            base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if get_world_size() > 1 else None
        val_loader = data.DataLoader(val_ds,
                                     batch_size=cfgs.SOLVER.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        return val_loader


class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # label 4 is ET
            result.append(d[key] == 4)
            # merge label 1 and label 4 to construct TC
            result.append(np.logical_or(d[key] == 1, d[key] == 4))
            # merge labels 1, 2 and 4 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 4), d[key] == 1
                )
            )
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class MinMaxNormalization(transforms.MapTransform):
    def __call__(self, data, low_perc=1, high_perc=99):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            non_zeros = img > 0
            low, high = np.percentile(img[non_zeros], [low_perc, high_perc])
            img = np.clip(img, low, high)
            img = normalize(img)
            d[key] = img
        return d


class CropForegroundCombined(transforms.MapTransform):
    def __init__(self, keys, roi_size):
        super(CropForegroundCombined, self).__init__(keys)
        assert len(roi_size) == 3, "the length of roi size is wrong!"
        self.roi_size = roi_size

    def __call__(self, data):
        d = dict(data)
        imgs = []
        img_keys = ['t1', 't1ce', 't2', 'flair']
        for key in self.keys:
            if key in img_keys:
                imgs.append(data[key])
        patient_image = np.stack(imgs).squeeze(1)
        z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
        zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
        d['t1'] = d['t1'][:, zmin:zmax, ymin:ymax, xmin:xmax]
        d['t1ce'] = d['t1ce'][:, zmin:zmax, ymin:ymax, xmin:xmax]
        d['t2'] = d['t2'][:, zmin:zmax, ymin:ymax, xmin:xmax]
        d['flair'] = d['flair'][:, zmin:zmax, ymin:ymax, xmin:xmax]
        d['seg'] = d['seg'][:, zmin:zmax, ymin:ymax, xmin:xmax]

        _, D, H, W = d["t1"].shape
        todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(self.roi_size, [D, H, W])]
        padlist = [(0, 0)]  # channel dim
        for to_pad in todos:
            if to_pad[0]:
                padlist.append((to_pad[1], to_pad[2]))
            else:
                padlist.append((0, 0))
        d['t1'] = np.pad(d['t1'], padlist)
        d['t1ce'] = np.pad(d['t1ce'], padlist)
        d['t2'] = np.pad(d['t2'], padlist)
        d['flair'] = np.pad(d['flair'], padlist)
        d['seg'] = np.pad(d['seg'], padlist)

        return d


def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image


def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def get_loader_brats_monai(cfgs):
    data_dir = cfgs.DATASETS.DATA_DIR
    datalist_json = os.path.join(data_dir, cfgs.DATASETS.JSON)
    train_files = datafold_read_monai_new(datalist=datalist_json,
                                          basedir=data_dir,
                                          key="training")
    validation_files = datafold_read_monai_new(datalist=datalist_json,
                                               basedir=data_dir,
                                               key="validation")
    test_files = datafold_read_monai_new(datalist=datalist_json,
                                         basedir=data_dir,
                                         key="test")

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(keys=["image", "label"],
                                       source_key="image",
                                       k_divisible=[cfgs.INPUT.RAND_CROP.ROI.X,
                                                    cfgs.INPUT.RAND_CROP.ROI.Y,
                                                    cfgs.INPUT.RAND_CROP.ROI.Z]),
            transforms.RandSpatialCropd(keys=["image", "label"],
                                        roi_size=[cfgs.INPUT.RAND_CROP.ROI.X,
                                                  cfgs.INPUT.RAND_CROP.ROI.Y,
                                                  cfgs.INPUT.RAND_CROP.ROI.Z],
                                        random_size=False),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=cfgs.INPUT.RAND.FLIP_AXIS_PROB,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=cfgs.INPUT.RAND.FLIP_AXIS_PROB,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=cfgs.INPUT.RAND.FLIP_AXIS_PROB,
                                 spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image",
                                           nonzero=True,
                                           channel_wise=True),
            transforms.RandScaleIntensityd(keys="image",
                                           factors=0.1,
                                           prob=cfgs.INPUT.RAND.SCALE_INTENSITY_PROB),
            transforms.RandShiftIntensityd(keys="image",
                                           offsets=0.1,
                                           prob=cfgs.INPUT.RAND.SHIFT_INTENSITY_PROB),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if cfgs.MODE == "test":
        val_ds = data.Dataset(data=test_files, transform=test_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if get_world_size() > 1 else None
        test_loader = data.DataLoader(val_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                      sampler=val_sampler,
                                      pin_memory=True,
                                      )
        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)
        train_sampler = Sampler(train_ds) if get_world_size() > 1 else None
        train_loader = data.DataLoader(train_ds,
                                       batch_size=cfgs.SOLVER.BATCH_SIZE,
                                       shuffle=(train_sampler is None),
                                       num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                       sampler=train_sampler,
                                       pin_memory=True,
                                       )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if get_world_size() > 1 else None
        val_loader = data.DataLoader(val_ds,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     )
        loader = [train_loader, val_loader]
    return loader


def datafold_read(datalist,
                  basedir,
                  fold=0,
                  key='training'):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if 'fold' in d and d['fold'] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def datafold_read_monai_new(datalist,
                            basedir,
                            key='training'):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if "id" in k:
                continue
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    return json_data


def get_loader_msd(cfgs):
    train_images = sorted(
        glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    if cfgs.DATASETS.NAME == "MSD":
        if cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas" \
                or cfgs.DATASETS.MSD_TYPE == "Spleen":
            train_files, val_files, test_files = data_dicts[:cfgs.DATASETS.MSD.TRAIN_NUM], \
                                                 data_dicts[cfgs.DATASETS.MSD.TRAIN_NUM: cfgs.DATASETS.MSD.VAL_NUM], \
                                                 data_dicts[cfgs.DATASETS.MSD.VAL_NUM:]
        else:
            train_num = int(0.7*len(data_dicts))
            train_files, val_files = data_dicts[:train_num], data_dicts[train_num:]

    train_transforms = transforms.Compose(
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
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            PadWithZeros(keys=["image", "label"],
                         roi_size=(cfgs.INPUT.RAND_CROP.ROI.X,
                                   cfgs.INPUT.RAND_CROP.ROI.Y,
                                   cfgs.INPUT.RAND_CROP.ROI.Z)
                         ),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(cfgs.INPUT.RAND_CROP.ROI.X, cfgs.INPUT.RAND_CROP.ROI.Y, cfgs.INPUT.RAND_CROP.ROI.Z),
                pos=cfgs.INPUT.RAND_CROP.POS,
                neg=cfgs.INPUT.RAND_CROP.NEG,
                num_samples=cfgs.INPUT.RAND_CROP.SAMPLES,
                image_key="image",
                image_threshold=0,
            ),
            transforms.EnsureTyped(keys=["image", "label"]),
        ]
    )
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
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.EnsureTyped(keys=["image", "label"]),
        ]
    )

    if cfgs.MODE == "train":
        train_ds = data.CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cfgs.DATASETS.CACHE.RATE,
            num_workers=cfgs.DATALOADER.TRAIN_WORKERS,
        )
        # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

        # use batch_size=2 to load images and use RandCropByPosNegLabeld
        # to generate 2 x 4 images for network training
        train_sampler = Sampler(train_ds) if get_world_size() > 1 else None
        train_loader = data.DataLoader(train_ds,
                                       batch_size=cfgs.SOLVER.BATCH_SIZE,
                                       shuffle=(train_sampler is None),
                                       sampler=train_sampler,
                                       num_workers=cfgs.DATALOADER.TRAIN_WORKERS,
                                       pin_memory=True,
                                       persistent_workers=True)
        val_ds = data.Dataset(
            data=val_files,
            transform=val_transforms
        )
        val_sampler = Sampler(val_ds, shuffle=False) if get_world_size() > 1 else None
        val_loader = data.DataLoader(val_ds,
                                     batch_size=cfgs.SOLVER.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        return train_loader, val_loader
    elif cfgs.MODE == "test":
        test_ds = data.Dataset(
            data=test_files,
            transform=val_transforms
        )
        test_sampler = Sampler(test_ds, shuffle=False) if get_world_size() > 1 else None
        test_loader = data.DataLoader(test_ds,
                                      batch_size=cfgs.SOLVER.BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                      sampler=test_sampler,
                                      pin_memory=True,
                                      persistent_workers=True)
        return test_loader
    else:
        raise NotImplementedError("This config mode does not support at present!")


class PadWithZeros(transforms.MapTransform):
    def __init__(self, keys, roi_size):
        super(PadWithZeros, self).__init__(keys)
        assert len(roi_size) == 3, "the length of roi size is wrong!"
        self.roi_size = roi_size

    def __call__(self, data):
        d = dict(data)

        _, D, H, W = d["image"].shape
        todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(self.roi_size, [D, H, W])]
        padlist = [(0, 0)]  # channel dim
        for to_pad in todos:
            if to_pad[0]:
                padlist.append((to_pad[1], to_pad[2]))
            else:
                padlist.append((0, 0))
        d['image'] = np.pad(d['image'], padlist)
        d['label'] = np.pad(d['label'], padlist)

        return d


def get_loader_brats_vt_unet(cfgs):
    data_dir = cfgs.DATASETS.DATA_DIR
    datalist_json = os.path.join(data_dir, cfgs.DATASETS.JSON)
    normalisation = "minmax"
    train_datalist = load_decathlon_datalist(datalist_json,
                                             True,
                                             "training",
                                             base_dir=data_dir)
    val_datalist = load_decathlon_datalist(datalist_json,
                                           True,
                                           "validation",
                                           base_dir=data_dir)
    test_datalist = load_decathlon_datalist(datalist_json,
                                            True,
                                            "test",
                                            base_dir=data_dir)
    train_list = [pathlib.Path(d["id"]) for d in train_datalist]
    val_list = [pathlib.Path(d["id"]) for d in val_datalist]
    test_list = [pathlib.Path(d["id"]) for d in test_datalist]

    if cfgs.MODE == "train":
        train_dataset = Brats(train_list, training=True,
                              normalisation=normalisation, cfg=cfgs)
        train_sampler = Sampler(train_dataset) if get_world_size() > 1 else None
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=cfgs.SOLVER.BATCH_SIZE,
                                       shuffle=(train_sampler is None),
                                       sampler=train_sampler,
                                       num_workers=cfgs.DATALOADER.TRAIN_WORKERS,
                                       pin_memory=True,
                                       persistent_workers=True)

        val_dataset = Brats(val_list, training=False, data_aug=False,
                            normalisation=normalisation, cfg=cfgs)

        val_sampler = Sampler(val_dataset, shuffle=False) if get_world_size() > 1 else None
        val_loader = data.DataLoader(val_dataset,
                                     batch_size=cfgs.SOLVER.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
        #                                            num_workers=4, pin_memory=True, drop_last=True)
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
        #                                          pin_memory=True, num_workers=2)
        loader = [train_loader, val_loader]
    elif cfgs.MODE == "test":
        test_dataset = Brats(test_list, training=False, benchmarking=True,
                             normalisation=normalisation, cfg=cfgs)
        loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=cfgs.SOLVER.BATCH_SIZE,
                                             num_workers=cfgs.DATALOADER.TEST_WORKERS)
    return loader


def get_loader_total_segmentator(cfgs):
    data_dir = cfgs.DATASETS.DATA_DIR
    datalist_json = os.path.join(data_dir, cfgs.DATASETS.JSON)








    train_images = sorted(
        glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    if cfgs.DATASETS.NAME == "MSD":
        if cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas" \
                or cfgs.DATASETS.MSD_TYPE == "Spleen":
            train_files, val_files, test_files = data_dicts[:cfgs.DATASETS.MSD.TRAIN_NUM], \
                                                 data_dicts[cfgs.DATASETS.MSD.TRAIN_NUM: cfgs.DATASETS.MSD.VAL_NUM], \
                                                 data_dicts[cfgs.DATASETS.MSD.VAL_NUM:]
        else:
            train_num = int(0.7*len(data_dicts))
            train_files, val_files = data_dicts[:train_num], data_dicts[train_num:]

    train_transforms = transforms.Compose(
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
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            PadWithZeros(keys=["image", "label"],
                         roi_size=(cfgs.INPUT.RAND_CROP.ROI.X,
                                   cfgs.INPUT.RAND_CROP.ROI.Y,
                                   cfgs.INPUT.RAND_CROP.ROI.Z)
                         ),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(cfgs.INPUT.RAND_CROP.ROI.X, cfgs.INPUT.RAND_CROP.ROI.Y, cfgs.INPUT.RAND_CROP.ROI.Z),
                pos=cfgs.INPUT.RAND_CROP.POS,
                neg=cfgs.INPUT.RAND_CROP.NEG,
                num_samples=cfgs.INPUT.RAND_CROP.SAMPLES,
                image_key="image",
                image_threshold=0,
            ),
            transforms.EnsureTyped(keys=["image", "label"]),
        ]
    )
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
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.EnsureTyped(keys=["image", "label"]),
        ]
    )

    if cfgs.MODE == "train":
        train_ds = data.CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cfgs.DATASETS.CACHE.RATE,
            num_workers=cfgs.DATALOADER.TRAIN_WORKERS,
        )
        # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

        # use batch_size=2 to load images and use RandCropByPosNegLabeld
        # to generate 2 x 4 images for network training
        train_sampler = Sampler(train_ds) if get_world_size() > 1 else None
        train_loader = data.DataLoader(train_ds,
                                       batch_size=cfgs.SOLVER.BATCH_SIZE,
                                       shuffle=(train_sampler is None),
                                       sampler=train_sampler,
                                       num_workers=cfgs.DATALOADER.TRAIN_WORKERS,
                                       pin_memory=True,
                                       persistent_workers=True)
        val_ds = data.Dataset(
            data=val_files,
            transform=val_transforms
        )
        val_sampler = Sampler(val_ds, shuffle=False) if get_world_size() > 1 else None
        val_loader = data.DataLoader(val_ds,
                                     batch_size=cfgs.SOLVER.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        return train_loader, val_loader
    elif cfgs.MODE == "test":
        test_ds = data.Dataset(
            data=test_files,
            transform=val_transforms
        )
        test_sampler = Sampler(test_ds, shuffle=False) if get_world_size() > 1 else None
        test_loader = data.DataLoader(test_ds,
                                      batch_size=cfgs.SOLVER.BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                      sampler=test_sampler,
                                      pin_memory=True,
                                      persistent_workers=True)
        return test_loader
    else:
        raise NotImplementedError("This config mode does not support at present!")