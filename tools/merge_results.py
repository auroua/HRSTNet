import os
import SimpleITK as sitk
import numpy as np


if __name__ == "__main__":
    img_name = "img0061"
    # label_part1_dir = f"/home/albert_wei/fdisk_c/train_output_medical/abdomen_seg/default0/inference_all_part1/{img_name}-Image/"
    label_part1_dir = f"/home/albert_wei/fdisk_c/train_output_medical/abdomen_seg/default0/inference/img0061/"
    label_part2_dir = f"/home/albert_wei/fdisk_c/train_output_medical/cervix_seg/default0/inference_all_part1/{img_name}-Image/"
    part1_imgs = [os.path.join(label_part1_dir, img_path) for img_path in os.listdir(label_part1_dir)]
    # part2_imgs = [os.path.join(label_part2_dir, img_path) for img_path in os.listdir(label_part2_dir)]

    # original_data = sitk.ReadImage(os.path.join(label_part1_dir, f"{img_name}-Image_seg_1.nii.gz"))
    original_data = sitk.ReadImage(os.path.join(label_part1_dir, f"img0061_seg_0.nii.gz"))

    data = sitk.GetArrayFromImage(original_data)
    total_data = np.zeros_like(data)

    for idx in range(len(part1_imgs)):
        if idx == 0:
            continue
        # original_data = sitk.ReadImage(os.path.join(label_part1_dir, f"{img_name}-Image_seg_{idx}.nii.gz"))
        original_data = sitk.ReadImage(os.path.join(label_part1_dir, f"img0061_seg_{idx}.nii.gz"))
        data = sitk.GetArrayFromImage(original_data) * idx
        total_data[data != 0] = data[data != 0]

    # base_val = len(part1_imgs)
    # for idx in range(len(part2_imgs)):
    #     if idx == 0:
    #         continue
    #     original_data = sitk.ReadImage(os.path.join(label_part2_dir, f"{img_name}-Image_seg_{idx}.nii.gz"))
    #     data = sitk.GetArrayFromImage(original_data) * base_val
    #     total_data[data != 0] = data[data != 0]
    #     base_val += 1

    out = sitk.GetImageFromArray(total_data)
    print(np.unique(total_data))
    out.SetOrigin(original_data.GetOrigin())
    out.SetSpacing(original_data.GetSpacing())
    out.SetDirection(original_data.GetDirection())

    sitk.WriteImage(out, os.path.join(label_part1_dir, f"{img_name}-Image_seg_total.nii.gz"))
