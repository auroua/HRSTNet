import nibabel as nib


if __name__ == "__main__":
    """
    The following code is used to fix bugs:
    ITK ERROR: ITK only supports orthonormal direction cosines.  No orthonormal definition found!
    """
    pred_file_path = "/home/ljm/Fdisk_A/train_outputs/train_output_medical/HR_TRANS_Results/MSD/Liver/msd_liver_hr_trans_monai_stages_4_wo_x0_epochs_300/seg_results/liver_93/liver_93_seg.nii.gz"
    original_img_path = "/home/ljm/Fdisk_A/train_datasets/train_datasets_medical/MSD/Task03_Liver/imagesTr/liver_93.nii.gz"

    img = nib.load(original_img_path)
    pred_img = nib.load(pred_file_path)
    qform = img.get_qform()
    pred_img.set_qform(qform)
    sform = img.get_sform()
    pred_img.set_sform(sform)
    nib.save(pred_img, pred_file_path)