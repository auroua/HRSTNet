import os

import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
from seg.utils.solver_utils import dice


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def new_dice(pred, label):
    tp_hard = np.sum((pred == 1).astype(np.float) * (label == 1).astype(np.float))
    fp_hard = np.sum((pred == 1).astype(np.float) * (label != 1).astype(np.float))
    fn_hard = np.sum((pred != 1).astype(np.float) * (label == 1).astype(np.float))
    return 2 * tp_hard / (2 * tp_hard + fp_hard + fn_hard)


def hd(pred, gt):
    # Modify the HD 95 Calculation from nnUnet
    # https://github.com/MIC-DKFZ/nnUNet/blob/5c18fa32f2b31575aae59d889d196e4c4ba8b844/nnunet/dataset_conversion/Task082_BraTS_2020.py#L330
    # math.sqrt(240**2 + 240**2 + 155**2) = 373.12866
    num_ref = np.sum(pred)
    num_pred = np.sum(gt)
    d, h, w = gt.shape
    max_val = np.sqrt(d*d + h*h + w*w)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return max_val
    elif num_pred == 0 and num_ref != 0:
        return max_val
    else:
        return binary.hd95(pred, gt)


def inference():
    gt_path = '/home/ljm/Fdisk_A/train_datasets/train_datasets_medical/MSD/Task03_Liver/labelsTr/'
    pred_path = '/home/ljm/Fdisk_A/train_outputs/train_output_medical/HR_TRANS_Results/MSD/Liver/msd_liver_hr_trans_monai_stages_4_wo_x0_epochs_300/seg_results/'
    MSD_TYPE = "Liver"    # Liver, Pancreas, Spleen
    CATEGORIES = 3

    MODE = "VALIDATE"   # VALIDATE

    infer_files = os.listdir(pred_path)
    print(f"Inference files loading success...{infer_files}")
    dice_list_organ = []
    dice_list_tumor = []

    hausdorff_95_organ_list = []
    hausdorff_95_tumor_list = []

    file = pred_path
    fw = open(file + '/dice_pred.txt', 'w')

    for idx, file_name in enumerate(infer_files):
        if file_name == "dice_pred.txt":
            continue
        print(f"Evaluating the {idx} file, and file name is {file_name}")
        patient_id = file_name.split(".")[0]
        pred_file_name = f"{patient_id}/{patient_id}_seg.nii.gz"
        label_path = os.path.join(gt_path, f"{patient_id}.nii.gz")
        infer_path = os.path.join(pred_path, pred_file_name)
        print(label_path)
        print(infer_path)
        label, infer = read_nii(label_path), read_nii(infer_path)

        if MODE != "VALIDATE":
            continue

        if MSD_TYPE == "Spleen":
            for i in range(1, CATEGORIES):
                organ_Dice = dice(infer == i, label == i)
                dice_list_organ.append(organ_Dice)
        elif MSD_TYPE == "Liver" or MSD_TYPE == "Pancreas":
            for i in range(1, CATEGORIES):
                if i == 1:
                    organ_Dice = dice(infer == i, label == i)
                    dice_list_organ.append(organ_Dice)
                elif i == 2:
                    tumor_Dice = dice(infer == i, label == i)
                    dice_list_tumor.append(tumor_Dice)
                else:
                    raise ValueError("Illegal input value!")

        if MSD_TYPE == "Spleen":
            organ_pred = infer == 1
            organ_gt = label == 1
            hausdor_95_organ = hd(organ_pred, organ_gt)
            hausdorff_95_organ_list.append(hausdor_95_organ)

        elif MSD_TYPE == "Liver" or MSD_TYPE == "Pancreas":
            organ_pred = infer == 1
            organ_gt = label == 1
            hausdor_95_organ = hd(organ_pred, organ_gt)

            tumor_pred = infer == 2
            tumor_gt = label == 2
            hausdor_95_tumor = hd(tumor_pred, tumor_gt)

            hausdorff_95_organ_list.append(hausdor_95_organ)
            hausdorff_95_tumor_list.append(hausdor_95_tumor)

            fw.write('*' * 20 + '\n', )
            fw.write(infer_path.split('/')[-1] + '\n')
            fw.write('dice_organ : {:.4f}\n'.format(dice_list_organ[-1]))
            if len(dice_list_tumor) > 0:
                fw.write('dice_tumor : {:.4f}\n'.format(dice_list_tumor[-1]))
            fw.write('*' * 20 + '\n', )
            fw.write('HD_organ : {:.4f}\n'.format(hausdorff_95_organ_list[-1]))
            if len(dice_list_tumor) > 0:
                fw.write('HD_tumor : {:.4f}\n'.format(hausdorff_95_tumor_list[-1]))

    dsc = []
    avg_hd = []
    dsc.append(np.mean(dice_list_organ))
    dsc.append(np.mean(dice_list_tumor))

    avg_hd.append(np.mean(hausdorff_95_organ_list))
    avg_hd.append(np.mean(hausdorff_95_tumor_list))

    fw.write('*' * 20 + '\n', )
    fw.write('*' * 20 + '\n', )

    fw.write('Avg Dice_organ: ' + str(np.mean(dice_list_organ)) + ' ' + '\n')
    fw.write('Avg Dice_tumor: ' + str(np.mean(dice_list_tumor)) + ' ' + '\n')

    fw.write('Avg HD_organ: ' + str(np.mean(hausdorff_95_organ_list)) + ' ' + '\n')
    fw.write('Avg HD_tumor: ' + str(np.mean(hausdorff_95_tumor_list)) + ' ' + '\n')

    fw.write('Avg Dice: ' + str(np.mean(dsc)) + ' ' + '\n')
    fw.write('Avg HD: ' + str(np.mean(avg_hd)) + ' ' + '\n')


if __name__ == '__main__':
    inference()
