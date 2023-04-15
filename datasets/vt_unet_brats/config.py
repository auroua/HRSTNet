

BRATS_TRAIN_FOLDERS = "/home/albert_wei/fdisk_c/train_dataset_Medical/BraTS2021"
TEST_FOLDER = "test_data"


def get_brats_folder(on="val"):
    if on == "train":
        return BRATS_TRAIN_FOLDERS


def get_test_brats_folder():
    return TEST_FOLDER