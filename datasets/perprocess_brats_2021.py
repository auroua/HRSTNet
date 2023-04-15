import os
import pathlib
import SimpleITK as sitk
import numpy as np
import json
from sklearn.model_selection import KFold


def merge_to_single_file(root_path, target_path):
    patterns = ["_t1", "_t1ce", "_t2", "_flair", "_seg"]
    base_folder = pathlib.Path(root_path).resolve()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])

    target_data_folder = os.path.join(target_path, "imagesTr")
    target_label_folder = os.path.join(target_path, "labelsTr")
    target_json_file = os.path.join(target_path, "dataset.json")
    if not os.path.exists(target_data_folder):
        os.mkdir(target_data_folder)
    if not os.path.exists(target_label_folder):
        os.mkdir(target_label_folder)

    file_name_list = []

    for patient_dir in patients_dir:
        patient_id = patient_dir.name
        paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in patterns]
        patient = dict(
            id=patient_id, t1=paths[0], t1ce=paths[1],
            t2=paths[2], flair=paths[3], seg=paths[4]
        )
        patient_image = {key: sitk.GetArrayFromImage(sitk.ReadImage(str(patient[key])))
                         for key in patient if key not in ["id", "seg"]}
        patient_image_sitk = sitk.GetImageFromArray(patient_image["t1"])
        patient_image = np.stack([patient_image[key] for key in patient_image])
        patient_label_sitk = sitk.ReadImage(str(patient["seg"]))
        patient_label = sitk.GetArrayFromImage(patient_label_sitk)
        et = (patient_label == 4).astype(np.uint8)
        tc = np.logical_or(patient_label == 4, patient_label == 1).astype(np.uint8)
        wt = np.logical_or(tc, patient_label == 2).astype(np.uint8)
        patient_label = np.stack([et, tc, wt])

        save_file_name = f"{patient['id']}.nii.gz"
        patient_image_out = sitk.GetImageFromArray(patient_image)
        patient_image_out.SetSpacing(patient_image_sitk.GetSpacing())
        patient_image_out.SetOrigin(patient_image_sitk.GetOrigin())
        patient_image_out.SetDirection(patient_image_sitk.GetDirection())

        patient_label_out = sitk.GetImageFromArray(patient_label)
        patient_label_out.SetSpacing(patient_label_sitk.GetSpacing())
        patient_label_out.SetOrigin(patient_label_sitk.GetOrigin())
        patient_label_out.SetDirection(patient_label_sitk.GetDirection())

        sitk.WriteImage(patient_image_out, os.path.join(target_data_folder, save_file_name))
        sitk.WriteImage(patient_label_out, os.path.join(target_label_folder, save_file_name))

        print(patient_image.shape, patient_label.shape)
        file_name_list.append(save_file_name)


def gen_brats_json_file(root_path, target_path):
    json_information = {}
    json_information["description"] = "BraTS 2021"
    json_information["labels"] = {"0": "background",
                                  "1": "et",
                                  "2": "tc",
                                  "4": "wt"}
    json_information["name"] = "BraTS 2021"
    json_information["numTest"] = 209
    json_information["numTraining"] = 834
    json_information["reference"] = "BraTS 2021"
    json_information["release"] = "2021"
    json_information["tensorImageSize"] = "3D"

    base_folder = pathlib.Path(root_path).resolve()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    kfold = KFold(3, shuffle=True, random_state=234)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[0]
    len_val = len(val_idx)
    val_index = val_idx[: len_val//2]
    test_index = val_idx[len_val // 2 :]

    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_index]
    test = [patients_dir[i] for i in test_index]

    file_types = ["t1", "t1ce", "t2", "flair", "seg"]
    training_list = []
    for t in train:
        patient_dict = {}
        patient_id = str(t).split("/")[-1]
        for f in file_types:
            patient_dict[f] = os.path.join(t, patient_id+f"_{f}.nii.gz")
        patient_dict["id"] = patient_id
        training_list.append(patient_dict)
    json_information["training"] = training_list

    val_list = []
    for t in val:
        patient_dict = {}
        patient_id = str(t).split("/")[-1]
        for f in file_types:
            patient_dict[f] = os.path.join(t, patient_id+f"_{f}.nii.gz")
        patient_dict["id"] = patient_id
        val_list.append(patient_dict)
    json_information["validation"] = val_list


    testing_list = []
    for t in test:
        patient_dict = {}
        patient_id = str(t).split("/")[-1]
        for f in file_types:
            patient_dict[f] = os.path.join(t, patient_id+f"_{f}.nii.gz")
        patient_dict["id"] = patient_id
        testing_list.append(patient_dict)
    json_information["test"] = testing_list

    with open(os.path.join(target_path, "dataset_0.json"), "w") as f:
        json.dump(json_information, f, indent=6)

    # json_str = json.dumps(json_information)
    # print(json_str)


def gen_brats_monai_json_file(root_path, target_path):
    json_information = {}
    json_information["description"] = "BraTS 2021"
    json_information["labels"] = {"0": "background",
                                  "1": "et",
                                  "2": "tc",
                                  "4": "wt"}
    json_information["name"] = "BraTS 2021"
    json_information["numTest"] = 209
    json_information["numTraining"] = 834
    json_information["reference"] = "BraTS 2021"
    json_information["release"] = "2021"
    json_information["tensorImageSize"] = "3D"

    base_folder = pathlib.Path(root_path).resolve()
    patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    kfold = KFold(3, shuffle=True, random_state=234)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[0]
    len_val = len(val_idx)
    val_index = val_idx[: len_val//2]
    test_index = val_idx[len_val // 2 :]

    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_index]
    test = [patients_dir[i] for i in test_index]

    file_types = ["t1", "t1ce", "t2", "flair", "seg"]
    training_list = []
    for t in train:
        patient_dict = {}
        patient_id = str(t).split("/")[-1]
        img_list = []
        for f in file_types:
            if "seg" in f:
                continue
            # img_list.append(os.path.join(t, patient_id+f"_{f}.nii.gz"))
            img_list.append(os.path.join(patient_id, patient_id+f"_{f}.nii.gz"))
        patient_dict["image"] = img_list
        patient_dict["id"] = patient_id
        patient_dict["label"] = os.path.join(patient_id, patient_id+"_seg.nii.gz")
        training_list.append(patient_dict)
    json_information["training"] = training_list

    val_list = []
    for t in val:
        patient_dict = {}
        patient_id = str(t).split("/")[-1]
        img_list = []
        for f in file_types:
            if "seg" in f:
                continue
            img_list.append(os.path.join(patient_id, patient_id+f"_{f}.nii.gz"))
        patient_dict["image"] = img_list
        patient_dict["id"] = patient_id
        patient_dict["label"] = os.path.join(patient_id, patient_id+"_seg.nii.gz")
        val_list.append(patient_dict)
    json_information["validation"] = val_list

    testing_list = []
    for t in test:
        patient_dict = {}
        patient_id = str(t).split("/")[-1]
        img_list = []
        for f in file_types:
            if "seg" in f:
                continue
            img_list.append(os.path.join(patient_id, patient_id+f"_{f}.nii.gz"))
        patient_dict["image"] = img_list
        patient_dict["id"] = patient_id
        patient_dict["label"] = os.path.join(patient_id, patient_id+"_seg.nii.gz")
        testing_list.append(patient_dict)
    json_information["test"] = val_list

    with open(os.path.join(target_path, "dataset_monai_0.json"), "w") as f:
        json.dump(json_information, f, indent=6)


def gen_testing_json_file(root_path, base_folder, target_path):
    files = os.listdir(root_path)
    json_path = "/home/albert_wei/fdisk_c/train_dataset_Medical/BraTS2021/dataset_0.json"
    json_file_list_training = []
    json_file_list_val = []
    with open(json_path, "r") as f:
        json_str = json.load(f)
    training = json_str["training"]
    val = json_str["validation"]
    for tf in training:
        json_file_list_training.append(tf["id"])
    for vf in val:
        json_file_list_val.append(vf["id"])
    total_json_file_brats = json_file_list_val + json_file_list_training
    test_files = []
    for f in files:
        if f not in total_json_file_brats and f.startswith("BraTS"):
            test_files.append(f)
    print(len(test_files))

    json_information = {}
    json_information["description"] = "BraTS 2021"
    json_information["labels"] = {"0": "background",
                                  "1": "et",
                                  "2": "tc",
                                  "4": "wt"}
    json_information["name"] = "BraTS 2021"
    json_information["numTest"] = 209
    json_information["numTraining"] = 834
    json_information["reference"] = "BraTS 2021"
    json_information["release"] = "2021"
    json_information["tensorImageSize"] = "3D"

    file_types = ["t1", "t1ce", "t2", "flair", "seg"]
    testing_list = []
    for t in test_files:
        patient_dict = {}
        patient_id = str(t).split("/")[-1]
        for f in file_types:
            patient_dict[f] = os.path.join(root_path, t, patient_id+f"_{f}.nii.gz")
        patient_dict["id"] = patient_id
        testing_list.append(patient_dict)
    json_information["test"] = testing_list

    with open(os.path.join(target_path, "dataset_0.json"), "w") as f:
        json.dump(json_information, f, indent=6)



def find_missing_file(root_path):
    files = os.listdir(root_path)
    print(files)
    json_path = "/home/albert_wei/fdisk_c/train_dataset_Medical/BraTS2021/dataset_0.json"
    # json_path = "/home/albert_wei/fdisk_c/train_dataset_Medical/BraTS2021_MSD/dataset_0.json"
    json_file_list_training = []
    json_file_list_val = []
    json_file_list_test = []
    with open(json_path, "r") as f:
        json_str = json.load(f)
    training = json_str["training"]
    val = json_str["validation"]
    testing = json_str["test"]
    for tf in training:
        json_file_list_training.append(tf["id"])
    for vf in val:
        json_file_list_val.append(vf["id"])
    for testf in testing:
        json_file_list_test.append(testf["id"])
    total_files = json_file_list_test + json_file_list_training + json_file_list_val
    print(len(json_file_list_training), len(json_file_list_val), len(json_file_list_test), len(files))
    print(len(list(set(json_file_list_training) & set(json_file_list_val))), len(list(set(json_file_list_training) & set(json_file_list_test))),
          len(list(set(json_file_list_val) & set(json_file_list_test))))
    for f_n in files:
        if f_n not in total_files:
            print(f_n)


if __name__ == "__main__":
    root_path = "/home/albert_wei/fdisk_c/train_dataset_Medical/BraTS2021/"
    target_dir = "/home/albert_wei/fdisk_c/train_dataset_Medical/BraTS2021_MSD/"
    # merge_to_single_file(root_path, target_dir)
    # gen_brats_monai_json_file(root_path, target_dir)
    # gen_brats_json_file(root_path, target_dir)

    find_missing_file(root_path)
    # gen_testing_json_file(root_path, root_path, target_dir)