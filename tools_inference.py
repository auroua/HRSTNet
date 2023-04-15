import os
import sys
import argparse
from seg.config.defaults import get_cfg
from seg.utils.comm import default_setup
from seg.engine.launch import launch
from seg.engine.trainer import Trainer
from seg.utils.comm import init_data_dir


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
    trainer = Trainer(cfgs)
    trainer.resume_or_load(resume=True)
    if args.space == "transformed":
        trainer.inference(cfgs)
    elif args.space == "original":
        trainer.inference_on_original_spacing(cfgs)
    elif args.space == "brats_2021_output":
        trainer.gen_inference_img(cfgs)
    elif args.space == "brats_2021_vt_unet":
        trainer.inference_brats_vt_unet(cfgs)
    elif args.space == "original_msd":
        trainer.inference_on_original_spacing_msd(cfgs)
    elif args.space == "original_abdomen":
        trainer.inference_on_original_spacing_abdomen(cfgs)

def get_parse():
    parser = argparse.ArgumentParser(description='Args for abdomen segmentation model.')
    parser.add_argument("--config_file", type=str,
                        default="./configs/hr_trans_brats_2021_seg.yaml",
                        # default="./configs/hr_trans_liver_seg.yaml",
                        # default="./configs/hr_trans_pancreas_seg.yaml",
                        # default="./configs/hr_trans_spleen_seg.yaml",
                        # default="./configs/hr_trans_abdomen_seg.yaml",

                        # default="./configs/vt_unet_brats_2021_seg.yaml",
                        # default="./configs/vt_unet_msd_liver_seg.yaml",
                        # default="./configs/vt_unet_msd_pancreas_seg.yaml",
                        # default="./configs/vt_unet_msd_spleen_seg.yaml",
                        # default="./configs/vt_unet_abdomen_seg.yaml",

                        # default="./configs/swin_unetr_brats_2021_seg.yaml",
                        # default="./configs/swin_unetr_liver_seg.yaml",
                        # default="./configs/swin_unetr_pancreas_seg.yaml",
                        # default="./configs/swin_unetr_spleen_seg.yaml",
                        # default="./configs/swin_unetr_abdomen_seg.yaml",

                        # default="./configs/unetr_abdomen_seg.yaml",
                        # default="./configs/unetr_brats_2021_seg.yaml",
                        # default="./configs/unetr_liver_seg.yaml",
                        # default="./configs/unetr_pancreas_seg.yaml",
                        # default="./configs/unetr_spleen_seg.yaml",
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
    parser.add_argument("--space", type=str, default="brats_2021_vt_unet",
                        choices=["original", "transformed", "original_msd", "brats_2021_output", "brats_2021_vt_unet",
                                 "visualization", "original_abdomen"],
                        help="perform evaluation only")
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