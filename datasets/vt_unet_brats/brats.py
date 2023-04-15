import pathlib
import SimpleITK as sitk
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset

from .config import get_brats_folder, get_test_brats_folder
from .image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise
from skimage.transform import resize


class Brats(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, data_aug=False,
                 no_seg=False, normalisation="minmax", cfg=None):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        self.cfg = cfg
        if not no_seg:
            self.patterns += ["_seg"]
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["seg"] is not None:
            et = patient_label == 4
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 4, patient_label == 1)
            wt = np.logical_or(tc, patient_label == 2)
            patient_label = np.stack([et, tc, wt])
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128 64, 64, 64 32, 32, 32
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
            # patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(96, 96, 96))
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]

        if self.cfg.MODEL.NAME == "extending_nnunet" and self.cfg.MODEL.EXTENDING_NN_UNET.DEEP_SUPERVISION and self.training:
            ds_scales = self.cfg.MODEL.EXTENDING_NN_UNET.DEEP_SUPERVISION_VALUES
            order = 0
            cval = 0
            axes = None
            if axes is None:
                axes = list(range(1, len(patient_label.shape)))
            output = []
            for s in ds_scales:
                if all([i == 1 for i in s]):
                    output.append(patient_label)
                else:
                    new_shape = np.array(patient_label.shape).astype(float)
                    for i, a in enumerate(axes):
                        new_shape[a] *= s[i]
                    new_shape = np.round(new_shape).astype(int)
                    out_seg = np.zeros(new_shape, dtype=patient_label.dtype)
                    for b in range(patient_label.shape[0]):
                        out_seg[b] = resize_segmentation(patient_label[b], new_shape[1:], order, cval)
                    output.append(out_seg)
            patient_labels = [torch.from_numpy(p.astype('bool')) for p in output]
            patient_labels = [p.float() for p in patient_labels]
            patient_image = torch.from_numpy(patient_image.astype("float16"))
            return dict(patient_id=_patient["id"],
                        image=patient_image.float(), label=patient_labels,
                        seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                        crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                        et_present=et_present,
                        supervised=True,
                        )
        else:
            patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
            patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
            return dict(patient_id=_patient["id"],
                        image=patient_image.float(), label=patient_label.float(),
                        seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                        crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                        et_present=et_present,
                        supervised=True,
                        )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)


def get_datasets(seed, on="train", fold_number=0, normalisation="minmax"):
    base_folder = pathlib.Path(get_brats_folder(on)).resolve()
    print(base_folder)
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])

    kfold = KFold(3, shuffle=True, random_state=seed)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[fold_number]
    len_val = len(val_idx)
    val_index = val_idx[: len_val//2]
    test_index = val_idx[len_val // 2 :]

    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_index]
    test = [patients_dir[i] for i in test_index]

    # return patients_dir
    train_dataset = Brats(train, training=True,
                          normalisation=normalisation)
    val_dataset = Brats(val, training=False, data_aug=False,
                        normalisation=normalisation)
    bench_dataset = Brats(test, training=False, benchmarking=True,
                          normalisation=normalisation)
    return train_dataset, val_dataset, bench_dataset


def get_test_datasets(seed, on="train", fold_number=0, normalisation="minmax"):
    base_folder = pathlib.Path(get_test_brats_folder()).resolve()
    print(base_folder)
    assert base_folder.exists()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])

    bench_dataset = Brats(patients_dir, training=False, benchmarking=True,
                          normalisation=normalisation)
    return bench_dataset


def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

