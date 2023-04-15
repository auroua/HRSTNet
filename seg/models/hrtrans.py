import torch
import torch.nn as nn
from seg.utils.register import MODEL_REGISTRY
from torch.nn import LayerNorm
from monai.networks.blocks import PatchEmbed
from typing import Sequence, Type
from monai.utils import ensure_tuple_rep
from seg.models.hrtrans_utils import FinalPatchExpand, HRTransStages, FinalStage
import torch.nn.functional as F
from einops import rearrange
from monai.networks.blocks import UnetrBasicBlock


class HRTrans(nn.Module):
    def __init__(
            self,
            cfgs,
            in_chans: int,
            embed_dim: int,
            spatial_dims: int,
            patch_size: int,
            num_heads: Sequence[int],
            depths: Sequence[int],
            mlp_ratio: int,
            num_classes: int,
            qkv_bias: bool = True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            window_size: int = 7,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            use_checkpoint: bool = False,
            patch_norm: bool = False
    ):
        super(HRTrans, self).__init__()
        self.cfgs = cfgs
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.window_size = ensure_tuple_rep(window_size, spatial_dims)
        self.patch_embed_dim = embed_dim
        self.stage_num = self.cfgs.MODEL.HR_TRANS.STAGE_NUM

        if self.cfgs.MODEL.HR_TRANS.USING_DS_STEM:
            self.patch_size //= 2
            self.patch_embed_dim //= 2

            self.ds_layer = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=int(embed_dim // 2),
                out_channels=int(embed_dim),
                kernel_size=3,
                stride=2,
                norm_name="instance",
                res_block=True
            )

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=self.patch_embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = HRTransStages(
            cfgs=cfgs,
            stage=self.stage_num,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=self.window_size,
            dpr=dpr,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims
        )

        # final layer
        self.finalStage = FinalStage(
            cfgs=cfgs,
            embed_dim=embed_dim,
            spatial_dims=spatial_dims,
            stage_num=self.stage_num
        )

        # prediction layer
        self.up = FinalPatchExpand(input_resolution=(
                self.cfgs.INPUT.RAND_CROP.ROI.Z // self.patch_size,
                self.cfgs.INPUT.RAND_CROP.ROI.Y // self.patch_size,
                self.cfgs.INPUT.RAND_CROP.ROI.X // self.patch_size),
            dim_scale=self.patch_size,
            dim=self.patch_embed_dim
        )
        self.output = nn.Conv3d(in_channels=self.patch_embed_dim,
                                out_channels=self.num_classes,
                                kernel_size=(1, 1, 1),
                                bias=False)

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)   # n c d h w
        x0 = self.pos_drop(x0)
        if self.cfgs.MODEL.HR_TRANS.USING_DS_STEM:
            x_in = self.ds_layer(x0)
            output = self.stages([x0, x_in])
        else:
            output = self.stages(x0)

        x = self.finalStage(output)

        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.up(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.output(x)
        return x


@MODEL_REGISTRY.register()
def build_hr_trans(cfgs):
    return HRTrans(
        cfgs=cfgs,
        in_chans=cfgs.MODEL.IN_CHANNEL,
        embed_dim=cfgs.MODEL.FEATURE_SIZE,
        spatial_dims=cfgs.MODEL.SWIN.SPATIAL_DIMS,
        patch_size=cfgs.MODEL.SWIN.PATCH_SIZE,
        num_heads=cfgs.MODEL.SWIN.NUM_HEADS,
        mlp_ratio=cfgs.MODEL.SWIN.MLP_RATE,
        depths=cfgs.MODEL.SWIN.DEPTH,
        qkv_bias=cfgs.MODEL.SWIN.QKV_BIAS,
        drop_path_rate=cfgs.MODEL.SWIN.DROP_PATH_RATE,
        use_checkpoint=cfgs.MODEL.USE_CHECKPOINT,
        num_classes=cfgs.MODEL.OUT_CHANNEL
    )