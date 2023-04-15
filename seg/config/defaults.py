from .config import CfgNode as CN

_C = CN()

_C.VERSION = 1

_C.TASK = 'SEG'

_C.MODEL = CN()
_C.MODEL.NAME = "unetr"     # unetr swin
_C.MODEL.DEVICE = "cuda"
_C.MODEL.POS_EMBED = "perceptron"
_C.MODEL.NORM = "instance"
_C.MODEL.NUM_HEADS = 12
_C.MODEL.MLP_DIM = 3072
_C.MODEL.HIDDEN_SIZE = 768
_C.MODEL.FEATURE_SIZE = 16
_C.MODEL.IN_CHANNEL = 1
_C.MODEL.OUT_CHANNEL = 14
_C.MODEL.DROPOUT_RATE = 0.0
_C.MODEL.ATTN_DROPOUT_RATE = 0.0
_C.MODEL.DECODER_DEPTHS = [2, 2, 2, 1]
_C.MODEL.PRETRAIN_DIR = ""
_C.MODEL.PRETRAINED_MODEL_NAME = ""     # 'swin_unetr.epoch.b4_5000ep_f48_lr2e-4_pretrained.pt'

# Path (a file path, or URL like detectron2://.., https://..) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""
_C.MODEL.USE_CHECKPOINT = None
_C.MODEL.SSL_PRETRAINED = None

_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.DEPTH = [2, 2, 2, 1]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.SPATIAL_DIMS = 3
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.DROP_PATH_RATE = 0.0
_C.MODEL.SWIN.MLP_RATE = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.PATCH_SIZE = 2

_C.MODEL.VT_UNET = CN()
_C.MODEL.VT_UNET.DEPTHS = [1, 2, 2, 2]
_C.MODEL.VT_UNET.FINAL_UPSAMPLE = "expand_first"
_C.MODEL.VT_UNET.FROZEN_STAGES = -1

_C.MODEL.HR_TRANS = CN()
_C.MODEL.HR_TRANS.USING_DS_STEM = False
_C.MODEL.HR_TRANS.STAGE_NUM = 4
_C.MODEL.HR_TRANS.FUSION_TYPE = "fc"     # fc, res_conv
_C.MODEL.HR_TRANS.DROPOUT_RATE = 0.2


_C.MODEL.DYNUNET = CN()
_C.MODEL.DYNUNET.KERNELS = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
_C.MODEL.DYNUNET.STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
_C.MODEL.DYNUNET.FILTERS = [64, 96, 128, 192, 256, 384, 512]
_C.MODEL.DYNUNET.DIM = 3
_C.MODEL.DYNUNET.DEEP_SUPERVISION = True
_C.MODEL.DYNUNET.DEEP_SUPR_NUM = 2
_C.MODEL.DYNUNET.RES_BLOCK = True
_C.MODEL.DYNUNET.MIN_FMAP = 2
_C.MODEL.DYNUNET.DEPTH = 6

_C.MODEL.EXTENDING_NN_UNET = CN()
_C.MODEL.EXTENDING_NN_UNET.NET_NUM_POOL_OP_KERNEL_SIZES = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
_C.MODEL.EXTENDING_NN_UNET.CONV_PER_STAGE = 2
_C.MODEL.EXTENDING_NN_UNET.NET_CONV_KERNEL_SIZES = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
_C.MODEL.EXTENDING_NN_UNET.BASE_NUM_FEATURES = 32
_C.MODEL.EXTENDING_NN_UNET.DEEP_SUPERVISION = True
_C.MODEL.EXTENDING_NN_UNET.DEEP_SUPERVISION_VALUES = [[1, 1, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.0625, 0.0625, 0.0625]]
_C.MODEL.EXTENDING_NN_UNET.WEIGHT_FACTOR = [0.53333333, 0.26666667, 0.13333333, 0.06666667, 0.]


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.ORIENTATION = "RSA"

_C.INPUT.SPACING = CN()
_C.INPUT.SPACING.X = 1.5
_C.INPUT.SPACING.Y = 1.5
_C.INPUT.SPACING.Z = 2.0
_C.INPUT.SPACING.IMG_MODE = "bilinear"
_C.INPUT.SPACING.LAB_MODEL = "nearest"

# Intensity original range min
_C.INPUT.SCALE_INTENSITY = CN()
_C.INPUT.SCALE_INTENSITY.ORIGINAL_MIN = 175.0
# Intensity original range max
_C.INPUT.SCALE_INTENSITY.ORIGINAL_MAX = 250.0
# Intensity target range min
_C.INPUT.SCALE_INTENSITY.TARGET_MIN = 0.0
# Intensity target range max
_C.INPUT.SCALE_INTENSITY.TARGET_MAX = 1.0

_C.INPUT.RAND_CROP = CN()
_C.INPUT.RAND_CROP.SAMPLES = 4
_C.INPUT.RAND_CROP.POS = 1
_C.INPUT.RAND_CROP.NEG = 1
_C.INPUT.RAND_CROP.ROI = CN()
_C.INPUT.RAND_CROP.ROI.X = 96
_C.INPUT.RAND_CROP.ROI.Y = 96
_C.INPUT.RAND_CROP.ROI.Z = 96

_C.INPUT.RAND = CN()
_C.INPUT.RAND.FLIP_AXIS_PROB = 0.2
_C.INPUT.RAND.ROTATE90_PROB = 0.2
_C.INPUT.RAND.SCALE_INTENSITY_PROB = 0.1
_C.INPUT.RAND.SHIFT_INTENSITY_PROB = 0.1

_C.INPUT.FORMAT = "NII"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
# Samples from these datasets will be merged and used as one dataset.
_C.DATASETS.NAME = ""
_C.DATASETS.JSON = ""
_C.DATASETS.DATA_DIR = ""
_C.DATASETS.TYPE = "CACHE"

_C.DATASETS.DATASET_TYPE = "BraTS_2021"     # "MSD", "BraTS_2021", "ABDOMEN"
_C.DATASETS.DATA_DIR_BRATS_2021 = ""
_C.DATASETS.DATA_DIR_LIVER = ""
_C.DATASETS.DATA_DIR_PANCREAS = ""
_C.DATASETS.DATA_DIR_SPLEEN = ""
_C.DATASETS.DATA_DIR_ABDOMEN = ""
_C.DATASETS.DATA_DIR_TOTAL_SEGMENTATOR = ""

_C.DATASETS.CACHE = CN()
_C.DATASETS.CACHE.NUM = 24
_C.DATASETS.CACHE.RATE = 1.0
_C.DATASETS.TEST_TYPE = "validation"    # training, validation, test

_C.DATASETS.FOLD = 0

_C.DATASETS.MSD_TYPE = "Spleen"     # Liver, Pancreas, Spleen

_C.DATASETS.MSD = CN()
_C.DATASETS.MSD.TRAIN_NUM = 100
_C.DATASETS.MSD.VAL_NUM = 100
_C.DATASETS.MSD.TEST_NUM = 100

_C.DATASETS.TOTAL_SEGMENTATOR = CN()
_C.DATASETS.TOTAL_SEGMENTATOR.ORGAN_TYPE = "car_org_gas"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_WORKERS = 4
_C.DATALOADER.TEST_WORKERS = 2


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 5000
_C.SOLVER.BATCH_SIZE = 1
_C.SOLVER.SW_BATCH_SIZE_TRAIN = 1
_C.SOLVER.SW_BATCH_SIZE_TEST = 1
_C.SOLVER.LR = 1e-4
_C.SOLVER.WEIGHT_DEACY = 1e-5
_C.SOLVER.MOMENTUM = 0.99
_C.SOLVER.OPTIM_ALGO = "adamw"
_C.SOLVER.EVAL_PERIOD = 2
_C.SOLVER.DROPOUT_RATE = 0.0
_C.SOLVER.INFER_OVERLAP = 0.5
_C.SOLVER.LR_SCHEDULE = "warmup_cosine"
_C.SOLVER.WARMUP_EPOCHS = 50
_C.SOLVER.SMOOTH_DR = 1e-6
_C.SOLVER.SMOOTH_NR = 0.0
_C.SOLVER.RESUME_CKPT = False
_C.SOLVER.RESUME_JIT = False
_C.SOLVER.AMP = False
_C.SOLVER.LOSS = "BDICE"

_C.SOLVER.LR_PARAMS = CN()
_C.SOLVER.LR_PARAMS.MILESTONES = [1000, 5000, 8000]
_C.SOLVER.LR_PARAMS.GAMMA = 0.1

# -----------------------------------------------------------------------------
# Dist
# -----------------------------------------------------------------------------
# _C.DIST = CN()
# _C.DIST.URL = "tcp://127.0.0.1:23456"
# _C.DIST.BACKEND = "nccl"
# _C.DIST.WORLD_SIZE = 1
# _C.DIST.RANK = 0
# _C.DIST.FLAG = True


_C.OUTPUT_DIR = ""
_C.SEED = -1
_C.MODE = "train"


def get_cfg():
    return _C.clone()