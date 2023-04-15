import torch
import torch.nn as nn
import copy
from seg.utils.register import MODEL_REGISTRY
from seg.models.vt_unet_utils import SwinTransformerSys3D


class VTUNet(nn.Module):
    def __init__(self, config, num_classes=3, zero_head=False, embed_dim=96, win_size=7):
        super(VTUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.cfg = config
        self.embed_dim = embed_dim
        self.win_size = win_size
        self.win_size = (self.win_size, self.win_size, self.win_size)

        self.swin_unet = SwinTransformerSys3D(img_size=(config.INPUT.RAND_CROP.ROI.X,
                                                        config.INPUT.RAND_CROP.ROI.Y,
                                                        config.INPUT.RAND_CROP.ROI.Z),
                                              patch_size=(config.MODEL.SWIN.PATCH_SIZE,
                                                          config.MODEL.SWIN.PATCH_SIZE,
                                                          config.MODEL.SWIN.PATCH_SIZE),
                                              in_chans=config.MODEL.IN_CHANNEL,
                                              num_classes=self.num_classes,
                                              embed_dim=self.embed_dim,
                                              depths=config.MODEL.SWIN.DEPTH,
                                              depths_decoder=config.MODEL.VT_UNET.DEPTHS,
                                              num_heads=config.MODEL.SWIN.NUM_HEADS,
                                              window_size=self.win_size,
                                              mlp_ratio=config.MODEL.SWIN.MLP_RATE,
                                              qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                              drop_rate=config.MODEL.DROPOUT_RATE,
                                              attn_drop_rate=config.MODEL.ATTN_DROPOUT_RATE,
                                              drop_path_rate=config.MODEL.SWIN.DROP_PATH_RATE,
                                              patch_norm=True,
                                              use_checkpoint=config.MODEL.USE_CHECKPOINT,
                                              frozen_stages=config.MODEL.VT_UNET.FROZEN_STAGES,
                                              final_upsample=config.MODEL.VT_UNET.FINAL_UPSAMPLE)

    def forward(self, x):
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                self.swin_unet.load_state_dict(pretrained_dict, strict=False)

                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")


@MODEL_REGISTRY.register()
def build_vt_unet(cfg):
    return VTUNet(
        config=cfg,
        num_classes=cfg.MODEL.OUT_CHANNEL,
        embed_dim=cfg.MODEL.FEATURE_SIZE,
        win_size=cfg.MODEL.SWIN.WINDOW_SIZE
    )