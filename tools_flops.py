import os
import sys
import argparse
import torch
from seg.config.defaults import get_cfg
from seg.utils.comm import default_setup
from seg.engine.launch import launch
from seg.utils.comm import init_data_dir
from seg.models.builder import get_model
from mmcv.cnn import get_model_complexity_info


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if len(cfg.MODEL.WEIGHTS) == 0:
        raise ValueError("In inference stage, the model weights should be determined!")
    default_setup(cfg, args)
    init_data_dir(cfg)
    cfg.freeze()
    return cfg


def main(args):
    cfgs = setup(args)
    # input_shape = (4, 96, 96, 96)
    input_shape = (4, 128, 128, 128)
    model = get_model(cfgs)
    if torch.cuda.is_available():
        model.cuda(1)
    model.eval()

    flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30

    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


def get_parse():
    parser = argparse.ArgumentParser(description='Args for abdomen segmentation model.')
    parser.add_argument("--config_file", type=str,
                        # default="./configs/hr_trans_brats_2021_seg.yaml",
                        # default="./configs/hr_trans_liver_seg.yaml",
                        # default="./configs/vt_unet_brats_2021_seg.yaml",
                        # default="./configs/vt_unet_msd_liver_seg.yaml",
                        # default="./configs/swin_unetr_brats_2021_seg.yaml",
                        # default="./configs/swin_unetr_liver_seg.yaml",
                        default="./configs/unetr_brats_2021_seg.yaml",
                        # default="./configs/unetr_liver_seg.yaml",
                        # default="./configs/dynunet_brats_2021_seg.yaml",
                        # default="./configs/extending_nnunet_brats_2021_seg.yaml",
                        help="Configuration files of neural architecture search algorithms.")
    parser.add_argument("--num-gpus", type=int,
                        default=1,
                        help="The number of gpus.")
    parser.add_argument("--num-machines", type=int,
                        default=1,
                        help="The number of machines.")
    parser.add_argument("--machine-rank", type=int,
                        default=0,
                        help="The rank of current machine.")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist_url", type=str,
                        default="tcp://127.0.0.1:{}".format(port),
                        help="initialization URL for pytorch distributed backend.")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
        Modify the MODEL.WEIGHTS in config file.
        Modify the MODE to test in config file
    """
    args = get_parse()
    print("Command Line Args: ", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )