import os

import SimpleITK as sitk
import numpy as np
from medpy.metric import binary


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def new_dice(pred, label):
    tp_hard = np.sum((pred == 1).astype(np.float) * (label == 1).astype(np.float))
    fp_hard = np.sum((pred == 1).astype(np.float) * (label != 1).astype(np.float))
    fn_hard = np.sum((pred != 1).astype(np.float) * (label == 1).astype(np.float))
    return 2 * tp_hard / (2 * tp_hard + fp_hard + fn_hard)


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def hd(pred, gt):
    # Modify the HD 95 Calculation from nnUnet
    # https://github.com/MIC-DKFZ/nnUNet/blob/5c18fa32f2b31575aae59d889d196e4c4ba8b844/nnunet/dataset_conversion/Task082_BraTS_2020.py#L330
    # math.sqrt(240**2 + 240**2 + 155**2) = 373.12866
    num_ref = np.sum(pred)
    num_pred = np.sum(gt)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return binary.hd95(pred, gt)


def process_label(label):
    et = label == 4
    ET = et
    TC = np.logical_or(label == 4, label == 1)
    WT = np.logical_or(TC, label == 2)
    return ET, TC, WT


def inference():
    path = '/home/albert_wei/fdisk_c/train_dataset_Medical/BraTS2021/'
    inferts_path = '/home/albert_wei/fdisk_c/HR_TRANS_Results/BraTS_2021/hr_trans_stages_2_epochs_300_vt_unet_preprocess/seg_results/'

    infer_files = os.listdir(inferts_path)
    print(f"Inference files loading success...{infer_files}")
    Dice_et = []
    Dice_tc = []
    Dice_wt = []

    HD_et = []
    HD_tc = []
    HD_wt = []

    file = inferts_path
    fw = open(file + '/dice_pre.txt', 'w')

    print("Loop ahead")
    substring = 'BraTS2021'
    for idx, file_name in enumerate(infer_files):
        if substring in str(file_name):
            print(f"Evaluating the {idx} file, and file name is {file_name}")
            patient_id = file_name.split(".")[0]
            label_path = os.path.join(path, patient_id, f"{patient_id}_seg.nii.gz")
            infer_path = os.path.join(inferts_path, file_name)
            print(label_path)
            print(infer_path)
            label, infer = read_nii(label_path), read_nii(infer_path)
            label_et, label_tc, label_wt = process_label(label)
            infer_et, infer_tc, infer_wt = process_label(infer)
            Dice_et.append(dice(infer_et, label_et))
            Dice_tc.append(dice(infer_tc, label_tc))
            Dice_wt.append(dice(infer_wt, label_wt))

            HD_et.append(hd(infer_et, label_et))
            HD_tc.append(hd(infer_tc, label_tc))
            HD_wt.append(hd(infer_wt, label_wt))

            fw.write('*' * 20 + '\n', )
            fw.write(infer_path.split('/')[-1] + '\n')
            fw.write('hd_et : {:.4f}\n'.format(HD_et[-1]))
            fw.write('hd_tc : {:.4f}\n'.format(HD_tc[-1]))
            fw.write('hd_wt : {:.4f}\n'.format(HD_wt[-1]))
            fw.write('*' * 20 + '\n', )
            fw.write('Dice_et : {:.4f}\n'.format(Dice_et[-1]))
            fw.write('Dice_tc : {:.4f}\n'.format(Dice_tc[-1]))
            fw.write('Dice_wt : {:.4f}\n'.format(Dice_wt[-1]))

        else:
            print(f"Not a valid file, and file name is {file_name}")
    dsc = []
    avg_hd = []
    dsc.append(np.mean(Dice_et))
    dsc.append(np.mean(Dice_tc))
    dsc.append(np.mean(Dice_wt))

    avg_hd.append(np.mean(HD_et))
    avg_hd.append(np.mean(HD_tc))
    avg_hd.append(np.mean(HD_wt))

    fw.write('*' * 20 + '\n', )
    fw.write('*' * 20 + '\n', )

    fw.write('Avg Dice_et: ' + str(np.mean(Dice_et)) + ' ' + '\n')
    fw.write('Avg Dice_tc: ' + str(np.mean(Dice_tc)) + ' ' + '\n')
    fw.write('Avg Dice_wt: ' + str(np.mean(Dice_wt)) + ' ' + '\n')

    fw.write('Avg HD_et: ' + str(np.mean(HD_et)) + ' ' + '\n')
    fw.write('Avg HD_tc: ' + str(np.mean(HD_tc)) + ' ' + '\n')
    fw.write('Avg HD_wt: ' + str(np.mean(HD_wt)) + ' ' + '\n')

    fw.write('Avg Dice: ' + str(np.mean(dsc)) + ' ' + '\n')
    fw.write('Avg HD: ' + str(np.mean(avg_hd)) + ' ' + '\n')


if __name__ == '__main__':
    inference()
