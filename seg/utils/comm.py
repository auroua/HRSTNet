import random
import os
from seg.utils.file_io import PathManager
from seg.utils.logger import setup_logger
from seg.utils.collect_env import collect_env_info
from seg.config.config import CfgNode
from seg.config.lazy import LazyConfig
from seg.utils.env import seed_all_rng
from seg.utils.dist_utils import is_main_process, get_rank, get_world_size
from omegaconf import OmegaConf
import numpy as np


def random_id(length):
    number = '0123456789'
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    id = ''
    for i in range(0, length, 2):
        id += random.choice(number)
        id += random.choice(alpha)
    return id


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        # OmegaConf.select(default=) is supported only after omegaconf2.1,
        # but some internal users still rely on 2.0
        parts = k.split(".")
        # https://github.com/omry/omegaconf/issues/674
        for p in parts:
            if p not in cfg:
                break
            cfg = OmegaConf.select(cfg, p)
        else:
            return cfg
    return default


def default_setup(cfgs, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfgs, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = get_rank()

    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank, name="seg")

    logger.info("Rank of current process: {}. World size: {}".format(rank, get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                _highlight(PathManager.open(args.config_file, "r").read(), args.config_file),
            )
        )

    if is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, f"config_{cfgs.MODEL.NAME}.yaml")
        if isinstance(cfgs, CfgNode):
            logger.info("Running with full config:\n{}".format(_highlight(cfgs.dump(), ".yaml")))
            with PathManager.open(path, "w") as f:
                f.write(cfgs.dump())
        else:
            LazyConfig.save(cfgs, path)
        logger.info("config file {} saved to {}".format(f"config_{cfgs.MODEL.NAME}.yaml", path))
    seed = _try_get_key(cfgs, "SEED", "train.seed", default=-1)

    seed_all_rng(None if seed < 0 else seed + 0)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def init_data_dir(cfgs):
    if cfgs.DATASETS.DATASET_TYPE == "BraTS_2021":
        cfgs.DATASETS.DATA_DIR = cfgs.DATASETS.DATA_DIR_BRATS_2021
    elif cfgs.DATASETS.DATASET_TYPE == "ABDOMEN":
        cfgs.DATASETS.DATA_DIR = cfgs.DATASETS.DATA_DIR_ABDOMEN
    elif cfgs.DATASETS.DATASET_TYPE == "MSD":
        if cfgs.DATASETS.MSD_TYPE == "Liver":
            cfgs.DATASETS.DATA_DIR = cfgs.DATASETS.DATA_DIR_LIVER
        elif cfgs.DATASETS.MSD_TYPE == "Pancreas":
            cfgs.DATASETS.DATA_DIR = cfgs.DATASETS.DATA_DIR_PANCREAS
        elif cfgs.DATASETS.MSD_TYPE == "Spleen":
            cfgs.DATASETS.DATA_DIR = cfgs.DATASETS.DATA_DIR_SPLEEN
        else:
            raise NotImplementedError(f"The MSD Type {cfgs.DATASETS.MSD_TYPE} does not support at present!")
    elif cfgs.DATASETS.DATASET_TYPE == "TotalSegmentator":
        cfgs.DATASETS.DATA_DIR = cfgs.DATASETS.DATA_DIR_TOTAL_SEGMENTATOR
    else:
        raise NotImplementedError(f"The dataset type {cfgs.DATASETS.DATASET_TYPE} does not support at present!")