from datasets.abdomen import get_loader_abdomen, get_loader_brats, get_loader_brats_monai, \
    get_loader_msd, get_loader_brats_vt_unet


def get_dataloader(cfgs):
    if cfgs.DATASETS.NAME == "ABDOMEN" or cfgs.DATASETS.NAME == "CERVIX" or cfgs.DATASETS.NAME == "TotalSegmentator":
        return get_loader_abdomen(cfgs)
    elif cfgs.DATASETS.NAME == "BraTS_2021":
        return get_loader_brats(cfgs)
    elif cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
        return get_loader_brats_monai(cfgs)
    elif cfgs.DATASETS.NAME == "MSD":
        return get_loader_msd(cfgs)
    elif cfgs.DATASETS.NAME == "BraTS_2021_VT_UNET":
        return get_loader_brats_vt_unet(cfgs)
