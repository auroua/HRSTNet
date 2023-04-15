import os

import numpy as np
from seg.utils.solver_utils import dice
from evaluate_msd import read_nii, hd


CATEGORY_LIST = ["BG", "Spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus", "liver", "stomach",
                 "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein", "pancreas", "right_adrenal_gland",
                 "left_adrenal_gland"]


def inference():
    gt_path = '/home/ljm/Fdisk_A/train_datasets/train_datasets_medical/2015_Segmentation_Cranial Vault Challenge/Abdomen/RawData/Training/label/'
    pred_path = '/home/ljm/Fdisk_A/train_outputs/train_output_medical_2022_8/hrstnet/abdomen_seg_hrstnet_stages_4/seg_results/'

    infer_files = os.listdir(pred_path)
    print(f"Inference files loading success...{infer_files}")

    dice_dict = {}
    hausdorff_95_dict = {}

    file = pred_path
    fw = open(file + '/dice_pred.txt', 'w')

    for idx, file_name in enumerate(infer_files):
        if file_name == "dice_pred.txt":
            continue
        print(f"Evaluating the {idx} file, and file name is {file_name}")
        patient_id = file_name.split(".")[0]
        pred_file_name = f"{patient_id}/{patient_id}_seg.nii.gz"
        label_path = os.path.join(gt_path, f"{patient_id.replace('img', 'label')}.nii.gz")
        infer_path = os.path.join(pred_path, pred_file_name)
        print(label_path)
        print(infer_path)
        label, infer = read_nii(label_path), read_nii(infer_path)

        for i in range(1, len(CATEGORY_LIST)):
            if i not in dice_dict:
                dice_dict[i] = [dice(infer == i, label == i)]
            else:
                dice_dict[i].append(dice(infer == i, label == i))
            organ_pred = infer == i
            organ_gt = label == i
            if i not in hausdorff_95_dict:
                hausdorff_95_dict[i] = [hd(organ_pred, organ_gt)]
            else:
                hausdorff_95_dict[i].append(hd(organ_pred, organ_gt))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        for i in range(1, len(CATEGORY_LIST)):
            fw.write('DICE_{} : {:.4f}\n'.format(CATEGORY_LIST[i], dice_dict[i][-1]))
        fw.write('*' * 20 + '\n', )
        for i in range(1, len(CATEGORY_LIST)):
            fw.write('HD_{} : {:.4f}\n'.format(CATEGORY_LIST[i], dice_dict[i][-1]))
        fw.write('HD_organ : {:.4f}\n'.format(hausdorff_95_dict[i][-1]))

    avg_dsc_dict = {}
    avg_hd_dict = {}

    for i in range(1, len(CATEGORY_LIST)):
        avg_dsc_dict[i] = np.mean(dice_dict[i])
        avg_hd_dict[i] = np.mean(hausdorff_95_dict[i])

    fw.write('#' * 20 + '\n', )
    avg_dice_list = []
    avg_hd_list = []
    for i in range(1, len(CATEGORY_LIST)):
        fw.write(f'Avg Dice {CATEGORY_LIST[i]}: ' + str(avg_dsc_dict[i]) + ' ' + '\n')
        avg_dice_list.append(avg_dsc_dict[i])
    fw.write('#' * 20 + '\n', )
    for i in range(1, len(CATEGORY_LIST)):
        fw.write(f'Avg HD {CATEGORY_LIST[i]}: ' + str(avg_hd_dict[i]) + ' ' + '\n')
        avg_hd_list.append(avg_hd_dict[i])
    fw.write('#' * 20 + '\n', )
    fw.write('Avg Dice: ' + str(np.mean(avg_dice_list)) + ' ' + '\n')
    fw.write('Avg HD: ' + str(np.mean(avg_hd_list)) + ' ' + '\n')


if __name__ == '__main__':
    inference()
