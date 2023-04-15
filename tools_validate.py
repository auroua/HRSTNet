import os
import sys
import argparse
from seg.config.defaults import get_cfg
from seg.utils.comm import default_setup
from seg.engine.launch import launch
from seg.engine.trainer import Trainer
from seg.utils.comm import init_data_dir
from seg.engine.trainer import val_epoch_brats


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    default_setup(cfg, args)
    init_data_dir(cfg)
    return cfg


def main(args):
    cfgs = setup(args)
    trainer = Trainer(cfgs)
    models = os.listdir(args.model_weights_path)
    model_files = [os.path.join(args.model_weights_path, m) for m in models if 'best' not in m and m.endswith('pth')]
    val_acc_dict = {}
    for model_file in model_files:
        model_epoch = model_file.split("/")[-1].split(".")[0].split("_")[-1]
        cfgs.MODEL.WEIGHTS = model_file
        trainer.resume_or_load(resume=args.resume)
        val = trainer.validate()
        val_acc_dict[model_epoch] = val
        print(val)
    print(val_acc_dict)


def get_parse():
    parser = argparse.ArgumentParser(description='Args for segmentation model.')
    parser.add_argument("--config_file", type=str,
                        # default="./configs/hr_trans_brats_2021_seg.yaml",
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

                        # default="./configs/dynunet_brats_2021_seg.yaml",
                        default="./configs/extending_nnunet_brats_2021_seg.yaml",
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
    parser.add_argument("--resume", action="store_true", help="resume from previous ckpt file")
    parser.add_argument("--model_weights_path", type=str, help="resume from previous ckpt file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
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