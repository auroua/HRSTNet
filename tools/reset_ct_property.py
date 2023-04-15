import os

import SimpleITK as sitk


def reset_properties(original_path, seg_path):
    original_data = sitk.ReadImage(original_path)
    original_spacing = original_data.GetSpacing()
    original_origin = original_data.GetOrigin()
    original_direction = original_data.GetDirection()

    mask_full_path = [os.path.join(seg_path, f) for f in os.listdir(seg_path)]
    for mask in mask_full_path:
        mask_data = sitk.ReadImage(mask)
        mask_spacing = mask_data.GetSpacing()
        mask_data.SetOrigin(original_origin)
        mask_data.SetDirection(original_direction)
        if original_spacing != mask_data.GetSpacing():
            print(f"different spacing: {original_spacing}, {mask_spacing}")
            mask_data.SetSpacing(original_spacing)
        sitk.WriteImage(mask_data, mask)


if __name__ == "__main__":
    original_path = "/home/albert_wei/fdisk_a/datasets_train_medical/2015_Segmentation_Cranial Vault Challenge/Cervix/RawData/Testing/img/2101064-Image.nii.gz"
    # seg_path = "/home/albert_wei/fdisk_c/train_output_medical/abdomen_seg/default0/inference_all_part1/2101064-Image/"
    seg_path = "/home/albert_wei/fdisk_c/train_output_medical/cervix_seg/default0/inference_all_part1/2101064-Image/"
    reset_properties(original_path, seg_path)