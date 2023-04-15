import SimpleITK as sitk
import os

if __name__ == "__main__":
    pred_img_folder = "/home/ljm/Fdisk_A/train_outputs/train_output_medical_2022_8/hrstnet/abdomen_seg_hrstnet_stages_4/seg_results/"
    gt_img_folder = "/home/ljm/Fdisk_A/train_datasets/train_datasets_medical/2015_Segmentation_Cranial Vault Challenge/Abdomen/RawData/Training/img/"

    img_lists = os.listdir(pred_img_folder)

    for img_name in img_lists:
        pred_img_name = os.path.join(pred_img_folder, img_name, f"{img_name}_seg.nii.gz")
        gt_img_name = os.path.join(gt_img_folder, f"{img_name}.nii.gz")
        pred_itkimage = sitk.ReadImage(pred_img_name)
        input_itkimage = sitk.ReadImage(gt_img_name)

        pred_itkimage.SetOrigin(input_itkimage.GetOrigin())
        pred_itkimage.SetDirection(input_itkimage.GetDirection())

        sitk.WriteImage(pred_itkimage, pred_img_name)