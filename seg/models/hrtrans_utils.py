import torch
import torch.nn as nn
from torch.nn import LayerNorm
from einops import rearrange
from typing import Sequence, Type, Union, Tuple
from seg.models.swin.patch_merging import PatchMerging
from seg.models.swin.swin_blocks import BasicLayer
from monai.networks.blocks import UnetrBasicBlock
from monai.utils import ensure_tuple_rep
from monai.networks.layers import get_act_layer
from monai.utils import look_up_option

SUPPORTED_DROPOUT_MODE = {"vit", "swin"}


class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        D, H, W = self.input_resolution
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape
        # assert L == D * H * W, "input feature has wrong size"

        x = x.view(B, L//(H*W), H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c',
                      p1=self.dim_scale,
                      p2=self.dim_scale,
                      p3=self.dim_scale,
                      c=C // 8)

        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x


class FinalPatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale**3 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        D, H, W = self.input_resolution
        x = x.permute(0, 4, 1, 2, 3)
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale,
                      p3=self.dim_scale,
                      c=C // (self.dim_scale ** 3))
        # x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class HRTransStage(nn.Module):
    def __init__(self,
                 cfgs,
                 stage: int,
                 total_stage_num: int,
                 embed_dim: int,
                 window_size: Sequence[int],
                 depths: Sequence[int],
                 num_heads: Sequence[int],
                 dpr,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 norm_layer: Type[LayerNorm] = nn.LayerNorm,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3
                 ):
        super(HRTransStage, self).__init__()
        self.cfgs = cfgs
        self.stage_val = stage
        self.window_size = window_size
        self.using_ds_stem = self.cfgs.MODEL.HR_TRANS.USING_DS_STEM
        self.total_stage_num = total_stage_num

        self.end_stage_flag = self.total_stage_num-1 == self.stage_val
        self.stages_module = nn.ModuleList()
        self.stage_ds_module = nn.ModuleList()

        for i in range(self.stage_val+1):
            stage = BasicLayer(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                drop_path=dpr[sum(depths[:i]): sum(depths[: i + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.stages_module.append(stage)

            if i != self.total_stage_num-1:
                stage_ds = PatchMerging(dim=int(embed_dim * 2 ** i),
                                        norm_layer=norm_layer,
                                        spatial_dims=len(window_size))
                self.stage_ds_module.append(stage_ds)

        if self.using_ds_stem:
            self.stage_fusion = nn.ModuleList()

            self.x0_trans = BasicLayer(
                dim=int(embed_dim // 2),
                depth=2,
                num_heads=3,
                window_size=window_size,
                drop_path=dpr[sum(depths[:0]): sum(depths[: 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )

            self.x0_ds_1 = PatchMerging(dim=int(embed_dim // 2),
                                        norm_layer=norm_layer,
                                        spatial_dims=len(self.window_size))

            self.x1_up = PatchExpanding(
                input_resolution=(
                    self.cfgs.INPUT.RAND_CROP.ROI.Z//self.cfgs.MODEL.SWIN.PATCH_SIZE,
                    self.cfgs.INPUT.RAND_CROP.ROI.Y//self.cfgs.MODEL.SWIN.PATCH_SIZE,
                    self.cfgs.INPUT.RAND_CROP.ROI.X//self.cfgs.MODEL.SWIN.PATCH_SIZE
                ),
                dim=int(embed_dim)
            )

            for i in range(self.stage_val + 3):
                if i == self.total_stage_num + 1:
                    continue
                if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                    stage_fusion = UnetrBasicBlock(
                        spatial_dims=spatial_dims,
                        in_channels=int((embed_dim // 2) * 2 ** i) * (self.stage_val + 2),
                        out_channels=int((embed_dim // 2) * 2 ** i),
                        kernel_size=3,
                        stride=1,
                        norm_name="instance",
                        res_block=True
                    )
                elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                    stage_fusion = MLPBlock(
                        hidden_size=int((embed_dim // 2) * 2 ** i) * (self.stage_val + 2),
                        mlp_dim=int((embed_dim // 2) * 2 ** i) * (self.stage_val + 2)//2,
                        output_dim=int((embed_dim // 2) * 2 ** i),
                        dropout_rate=self.cfgs.MODEL.HR_TRANS.DROPOUT_RATE
                    )
                else:
                    raise NotImplementedError(f"the fusion model {self.cfgs.MODEL.HR_TRANS.FUSION_TYPE} "
                                              f"has not implemented!")
                self.stage_fusion.append(stage_fusion)

            # for new layer
            if not self.end_stage_flag or self.stage_val > 0:
                self.x0_ds_2 = PatchMerging(dim=int(embed_dim),
                                            norm_layer=norm_layer,
                                            spatial_dims=len(self.window_size))

    def forward(self, x):
        results_x = []
        results_x_ds = []

        if self.using_ds_stem:
            x_embedding, x = x
        for i in range(self.stage_val+1):
            x = self.stages_module[i](x)
            results_x.append(x)

            if i != self.total_stage_num-1:
                x_dc = rearrange(x, "b c d h w -> b d h w c").contiguous()
                x_ds = self.stage_ds_module[i](x_dc)
                x_ds = rearrange(x_ds, "b d h w c -> b c d h w")
                results_x_ds.append(x_ds)

        if self.using_ds_stem:
            x0 = self.x0_trans(x_embedding)
            x0_ds_1 = self.x0_ds_1(rearrange(x0, "b c d h w -> b d h w c"))
            if not self.end_stage_flag:
                x0_ds_2 = self.x0_ds_2(x0_ds_1)
                x0_ds_2 = rearrange(x0_ds_2, "b d h w c -> b c d h w")
            x0_ds_1 = rearrange(x0_ds_1, "b d h w c -> b c d h w")
            x1_up = self.x1_up(results_x[0])

            x0 = torch.cat([x0, x1_up], dim=1)
            x1 = torch.cat([x0_ds_1, results_x[0]], dim=1)

            if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                x0 = self.stage_fusion[0](x0)
                x1 = self.stage_fusion[1](x1)

                if not self.end_stage_flag:
                    x2 = torch.cat([x0_ds_2, results_x_ds[0]], dim=1)
                    x2 = self.stage_fusion[2](x2)
            elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                x0 = self.stage_fusion[0](rearrange(x0, "b c d h w -> b d h w c"))
                x1 = self.stage_fusion[1](rearrange(x1, "b c d h w -> b d h w c"))
                x0 = rearrange(x0, "b d h w c -> b c d h w")
                x1 = rearrange(x1, "b d h w c -> b c d h w")
                if not self.end_stage_flag:
                    x2 = torch.cat([x0_ds_2, results_x_ds[0]], dim=1)
                    x2 = self.stage_fusion[2](rearrange(x2, "b c d h w -> b d h w c"))
                    x2 = rearrange(x2, "b d h w c -> b c d h w")
            else:
                raise NotImplementedError(f"the fusion model {self.cfgs.MODEL.HR_TRANS.FUSION_TYPE} "
                                          f"has not implemented!")
            if not self.end_stage_flag:
                return x0, x1, x2
            else:
                return x0, x1
        else:
            x1 = results_x[0]
            if not self.end_stage_flag:
                x2 = results_x_ds[0]
                return x1, x2
            else:
                return x1


class HRTransStage1(HRTransStage):
    def __init__(self,
                 cfgs,
                 stage: int,
                 total_stage_num: int,
                 embed_dim: int,
                 window_size: Sequence[int],
                 depths: Sequence[int],
                 num_heads: Sequence[int],
                 dpr,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 norm_layer: Type[LayerNorm] = nn.LayerNorm,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3
                 ):
        HRTransStage.__init__(self,
                              cfgs,
                              stage,
                              total_stage_num,
                              embed_dim,
                              window_size,
                              depths,
                              num_heads,
                              dpr,
                              mlp_ratio,
                              qkv_bias,
                              drop_rate,
                              attn_drop_rate,
                              norm_layer,
                              use_checkpoint
                              )
        self.x2_us = PatchExpanding(
            input_resolution=(
                self.cfgs.INPUT.RAND_CROP.ROI.Z//(self.cfgs.MODEL.SWIN.PATCH_SIZE*2),
                self.cfgs.INPUT.RAND_CROP.ROI.Y//(self.cfgs.MODEL.SWIN.PATCH_SIZE*2),
                self.cfgs.INPUT.RAND_CROP.ROI.X//(self.cfgs.MODEL.SWIN.PATCH_SIZE*2)
            ),
            dim=int(embed_dim * 2)
        )

        # for new layer
        if not self.end_stage_flag or self.stage_val > 1:
            self.x1_ds_2 = PatchMerging(dim=int(embed_dim * 2),
                                        norm_layer=norm_layer,
                                        spatial_dims=len(self.window_size))

        if self.using_ds_stem:
            if not self.end_stage_flag or self.stage_val > 1:
                self.x0_ds_3 = PatchMerging(dim=int(embed_dim*2),
                                            norm_layer=norm_layer,
                                            spatial_dims=len(self.window_size))

            self.x2_us_2 = PatchExpanding(
                input_resolution=(
                    self.cfgs.INPUT.RAND_CROP.ROI.Z // self.cfgs.MODEL.SWIN.PATCH_SIZE,
                    self.cfgs.INPUT.RAND_CROP.ROI.Y // self.cfgs.MODEL.SWIN.PATCH_SIZE,
                    self.cfgs.INPUT.RAND_CROP.ROI.X // self.cfgs.MODEL.SWIN.PATCH_SIZE
                ),
                dim=int(embed_dim)
            )

        else:
            self.stage_fusion = torch.nn.ModuleList()
            # fusion model
            for i in range(self.stage_val + 2):
                if i == self.total_stage_num and self.end_stage_flag:
                    continue
                if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                    stage_fusion = UnetrBasicBlock(
                        spatial_dims=spatial_dims,
                        in_channels=int(embed_dim * 2 ** i) * (self.stage_val + 1),
                        out_channels=int(embed_dim * 2 ** i),
                        kernel_size=3,
                        stride=1,
                        norm_name="instance",
                        res_block=True
                    )
                elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                    stage_fusion = MLPBlock(
                        hidden_size=int(embed_dim * 2 ** i) * (self.stage_val + 1),
                        mlp_dim=int(embed_dim * 2 ** i) * (self.stage_val + 1)//2,
                        output_dim=int(embed_dim * 2 ** i),
                        dropout_rate=self.cfgs.MODEL.HR_TRANS.DROPOUT_RATE
                    )
                else:
                    raise NotImplementedError(f"the fusion model {self.cfgs.MODEL.HR_TRANS.FUSION_TYPE} "
                                              f"has not implemented!")
                self.stage_fusion.append(stage_fusion)

    def forward(self, x_in):
        results_x = []
        results_x_ds = []
        if self.using_ds_stem:
            x_embedding = x_in[0]
            x_in = x_in[1:]

        for i in range(self.stage_val+1):
            x = self.stages_module[i](x_in[i])
            results_x.append(x)

            if i != self.total_stage_num-1:
                x_dc = rearrange(x, "b c d h w -> b d h w c").contiguous()
                x_ds = self.stage_ds_module[i](x_dc)
                x_ds = rearrange(x_ds, "b d h w c -> b c d h w")
                results_x_ds.append(x_ds)

        x2_us = self.x2_us(results_x[1])
        if not self.end_stage_flag:
            x1_ds_2 = self.x1_ds_2(rearrange(results_x_ds[0], "b c d h w -> b d h w c").contiguous())
            x1_ds_2 = rearrange(x1_ds_2, "b d h w c-> b c d h w").contiguous()

        if self.using_ds_stem:
            x0 = self.x0_trans(x_embedding)
            x0_ds_1 = self.x0_ds_1(rearrange(x0, "b c d h w -> b d h w c"))
            x0_ds_2 = self.x0_ds_2(x0_ds_1)
            if not self.end_stage_flag:
                x0_ds_3 = self.x0_ds_3(x0_ds_2)
            x0_ds_1 = rearrange(x0_ds_1, "b d h w c -> b c d h w")
            x0_ds_2 = rearrange(x0_ds_2, "b d h w c -> b c d h w")
            if not self.end_stage_flag:
                x0_ds_3 = rearrange(x0_ds_3, "b d h w c -> b c d h w")

            x1_us = self.x1_up(results_x[0])
            x2_us_2 = self.x2_us_2(x2_us)

            x0 = torch.cat([x0, x1_us, x2_us_2], dim=1)
            x1 = torch.cat([x0_ds_1, results_x[0], x2_us], dim=1)
            x2 = torch.cat([x0_ds_2, results_x_ds[0], results_x[1]], dim=1)

            if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                x0 = self.stage_fusion[0](x0)
                x1 = self.stage_fusion[1](x1)
                x2 = self.stage_fusion[2](x2)

                if not self.end_stage_flag:
                    x3 = torch.cat([x0_ds_3, x1_ds_2, results_x_ds[1]], dim=1)
                    x3 = self.stage_fusion[3](x3)
                    return x0, x1, x2, x3
                else:
                    return x0, x1, x2
            elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                x0 = self.stage_fusion[0](rearrange(x0, "b c d h w -> b d h w c"))
                x1 = self.stage_fusion[1](rearrange(x1, "b c d h w -> b d h w c"))
                x2 = self.stage_fusion[2](rearrange(x2, "b c d h w -> b d h w c"))

                x0 = rearrange(x0, "b d h w c -> b c d h w")
                x1 = rearrange(x1, "b d h w c -> b c d h w")
                x2 = rearrange(x2, "b d h w c -> b c d h w")
                if not self.end_stage_flag:
                    x3 = torch.cat([x0_ds_3, x1_ds_2, results_x_ds[1]], dim=1)
                    x3 = self.stage_fusion[3](rearrange(x3, "b c d h w -> b d h w c"))
                    x3 = rearrange(x3, "b d h w c -> b c d h w")
                    return x0, x1, x2, x3
                else:
                    return x0, x1, x2
            else:
                raise NotImplementedError(f"the fusion model {self.cfgs.MODEL.HR_TRANS.FUSION_TYPE} "
                                          f"has not implemented!")
        else:
            x1 = torch.cat([results_x[0], x2_us], dim=1)
            x2 = torch.cat([results_x_ds[0], results_x[1]], dim=1)

            if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                x1 = self.stage_fusion[0](x1)
                x2 = self.stage_fusion[1](x2)

                if not self.end_stage_flag:
                    x3 = torch.cat([x1_ds_2, results_x_ds[1]], dim=1)
                    x3 = self.stage_fusion[2](x3)
                    return x1, x2, x3
                else:
                    return x1, x2
            elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                x1 = self.stage_fusion[0](rearrange(x1, "b c d h w -> b d h w c"))
                x2 = self.stage_fusion[1](rearrange(x2, "b c d h w -> b d h w c"))
                x1 = rearrange(x1, "b d h w c -> b c d h w")
                x2 = rearrange(x2, "b d h w c -> b c d h w")

                if not self.end_stage_flag:
                    x3 = torch.cat([x1_ds_2, results_x_ds[1]], dim=1)
                    x3 = self.stage_fusion[2](rearrange(x3, "b c d h w -> b d h w c"))
                    x3 = rearrange(x3, "b d h w c -> b c d h w")
                    return x1, x2, x3
                else:
                    return x1, x2


class HRTransStage2(HRTransStage1):
    def __init__(self,
                 cfgs,
                 stage: int,
                 total_stage_num: int,
                 embed_dim: int,
                 window_size: Sequence[int],
                 depths: Sequence[int],
                 num_heads: Sequence[int],
                 dpr,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 norm_layer: Type[LayerNorm] = nn.LayerNorm,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3
                 ):
        HRTransStage1.__init__(self,
                               cfgs,
                               stage,
                               total_stage_num,
                               embed_dim,
                               window_size,
                               depths,
                               num_heads,
                               dpr,
                               mlp_ratio,
                               qkv_bias,
                               drop_rate,
                               attn_drop_rate,
                               norm_layer,
                               use_checkpoint,
                               spatial_dims)
        self.x3_us_1 = PatchExpanding(
            input_resolution=(
                self.cfgs.INPUT.RAND_CROP.ROI.Z//(self.cfgs.MODEL.SWIN.PATCH_SIZE*4),
                self.cfgs.INPUT.RAND_CROP.ROI.Y//(self.cfgs.MODEL.SWIN.PATCH_SIZE*4),
                self.cfgs.INPUT.RAND_CROP.ROI.X//(self.cfgs.MODEL.SWIN.PATCH_SIZE*4)
            ),
            dim=int(embed_dim * 2 ** 2)
        )

        self.x3_us_2 = PatchExpanding(
            input_resolution=(
                self.cfgs.INPUT.RAND_CROP.ROI.Z//(self.cfgs.MODEL.SWIN.PATCH_SIZE*2),
                self.cfgs.INPUT.RAND_CROP.ROI.Y//(self.cfgs.MODEL.SWIN.PATCH_SIZE*2),
                self.cfgs.INPUT.RAND_CROP.ROI.X//(self.cfgs.MODEL.SWIN.PATCH_SIZE*2)
            ),
            dim=int(embed_dim * 2 ** 1)
        )

        if not self.end_stage_flag or self.stage_val > 2:
            self.x1_ds_3 = PatchMerging(dim=int(embed_dim * 2 ** 2),
                                        norm_layer=norm_layer,
                                        spatial_dims=len(self.window_size))
            self.x2_ds_2 = PatchMerging(dim=int(embed_dim * 2 ** 2),
                                        norm_layer=norm_layer,
                                        spatial_dims=len(self.window_size))

        if self.using_ds_stem:
            if not self.end_stage_flag or self.stage_val > 2:
                self.x0_ds_4 = PatchMerging(dim=int(embed_dim*4),
                                            norm_layer=norm_layer,
                                            spatial_dims=len(self.window_size))

            self.x3_us_3 = PatchExpanding(
                input_resolution=(
                    self.cfgs.INPUT.RAND_CROP.ROI.Z // self.cfgs.MODEL.SWIN.PATCH_SIZE,
                    self.cfgs.INPUT.RAND_CROP.ROI.Y // self.cfgs.MODEL.SWIN.PATCH_SIZE,
                    self.cfgs.INPUT.RAND_CROP.ROI.X // self.cfgs.MODEL.SWIN.PATCH_SIZE
                ),
                dim=int(embed_dim)
            )

    def forward(self, x_in):
        results_x = []
        results_x_ds = []

        if self.using_ds_stem:
            x_embedding = x_in[0]
            x_in = x_in[1:]

        for i in range(self.stage_val+1):
            x = self.stages_module[i](x_in[i])
            results_x.append(x)

            if i != self.total_stage_num-1:
                x_dc = rearrange(x, "b c d h w -> b d h w c").contiguous()
                x_ds = self.stage_ds_module[i](x_dc)
                x_ds = rearrange(x_ds, "b d h w c -> b c d h w")
                results_x_ds.append(x_ds)

        x1_ds_2 = self.x1_ds_2(rearrange(results_x_ds[0], "b c d h w -> b d h w c").contiguous())
        if not self.end_stage_flag:
            x1_ds_3 = self.x1_ds_3(x1_ds_2)
            x2_ds_2 = self.x2_ds_2(rearrange(results_x_ds[1], "b c d h w -> b d h w c").contiguous())
        x1_ds_2 = rearrange(x1_ds_2, "b d h w c-> b c d h w").contiguous()
        if not self.end_stage_flag:
            x1_ds_3 = rearrange(x1_ds_3, "b d h w c-> b c d h w").contiguous()
            x2_ds_2 = rearrange(x2_ds_2, "b d h w c -> b c d h w").contiguous()

        x2_us = self.x2_us(results_x[1])

        x3_us_1 = self.x3_us_1(results_x[2])
        x3_us_2 = self.x3_us_2(x3_us_1)

        if self.using_ds_stem:
            x0 = self.x0_trans(x_embedding)
            x0_ds_1 = self.x0_ds_1(rearrange(x0, "b c d h w -> b d h w c"))
            x0_ds_2 = self.x0_ds_2(x0_ds_1)
            x0_ds_3 = self.x0_ds_3(x0_ds_2)
            if not self.end_stage_flag:
                x0_ds_4 = self.x0_ds_4(x0_ds_3)
            x0_ds_1 = rearrange(x0_ds_1, "b d h w c -> b c d h w")
            x0_ds_2 = rearrange(x0_ds_2, "b d h w c -> b c d h w")
            x0_ds_3 = rearrange(x0_ds_3, "b d h w c -> b c d h w")
            if not self.end_stage_flag:
                x0_ds_4 = rearrange(x0_ds_4, "b d h w c -> b c d h w")

            x1_us = self.x1_up(results_x[0])
            x2_us_2 = self.x2_us_2(x2_us)

            x3_us_3 = self.x3_us_3(x3_us_2)

            x0 = torch.cat([x0, x1_us, x2_us_2, x3_us_3], dim=1)
            x1 = torch.cat([x0_ds_1, results_x[0], x2_us, x3_us_2], dim=1)
            x2 = torch.cat([x0_ds_2, results_x_ds[0], results_x[1], x3_us_1], dim=1)
            x3 = torch.cat([x0_ds_3, x1_ds_2, results_x_ds[1], results_x[2]], dim=1)

            if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                x0 = self.stage_fusion[0](x0)
                x1 = self.stage_fusion[1](x1)
                x2 = self.stage_fusion[2](x2)
                x3 = self.stage_fusion[3](x3)
            elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                x0 = self.stage_fusion[0](rearrange(x0, "b c d h w -> b d h w c"))
                x1 = self.stage_fusion[1](rearrange(x1, "b c d h w -> b d h w c"))
                x2 = self.stage_fusion[2](rearrange(x2, "b c d h w -> b d h w c"))
                x3 = self.stage_fusion[3](rearrange(x3, "b c d h w -> b d h w c"))
                x0 = rearrange(x0, "b d h w c -> b c d h w")
                x1 = rearrange(x1, "b d h w c -> b c d h w")
                x2 = rearrange(x2, "b d h w c -> b c d h w")
                x3 = rearrange(x3, "b d h w c -> b c d h w")
            else:
                raise NotImplementedError(f"the fusion model {self.cfgs.MODEL.HR_TRANS.FUSION_TYPE} "
                                          f"has not implemented!")
            if not self.end_stage_flag:
                x4 = torch.cat([x0_ds_4, x1_ds_3, x2_ds_2, results_x_ds[2]], dim=1)
                if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                    x4 = self.stage_fusion[4](x4)
                elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                    x4 = self.stage_fusion[4](rearrange(x4, "b c d h w -> b d h w c"))
                    x4 = rearrange(x4, "b d h w c -> b c d h w")
                return x0, x1, x2, x3, x4
            else:
                return x0, x1, x2, x3
        else:
            x1 = torch.cat([results_x[0], x2_us, x3_us_2], dim=1)
            x2 = torch.cat([results_x_ds[0], results_x[1], x3_us_1], dim=1)
            x3 = torch.cat([x1_ds_2, results_x_ds[1], results_x[2]], dim=1)

            if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                x1 = self.stage_fusion[0](x1)
                x2 = self.stage_fusion[1](x2)
                x3 = self.stage_fusion[2](x3)
            elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                x1 = self.stage_fusion[0](rearrange(x1, "b c d h w -> b d h w c"))
                x2 = self.stage_fusion[1](rearrange(x2, "b c d h w -> b d h w c"))
                x3 = self.stage_fusion[2](rearrange(x3, "b c d h w -> b d h w c"))
                x1 = rearrange(x1, "b d h w c -> b c d h w ")
                x2 = rearrange(x2, "b d h w c -> b c d h w ")
                x3 = rearrange(x3, "b d h w c -> b c d h w ")

            if not self.end_stage_flag:
                x4 = torch.cat([x1_ds_3, x2_ds_2, results_x_ds[2]], dim=1)
                if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                    x4 = self.stage_fusion[3](x4)
                elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                    x4 = self.stage_fusion[3](rearrange(x4, "b c d h w -> b d h w c"))
                    x4 = rearrange(x4, "b d h w c -> b c d h w")
                return x1, x2, x3, x4
            else:
                return x1, x2, x3


class HRTransStage3(HRTransStage2):
    def __init__(self,
                 cfgs,
                 stage: int,
                 total_stage_num: int,
                 embed_dim: int,
                 window_size: Sequence[int],
                 depths: Sequence[int],
                 num_heads: Sequence[int],
                 dpr,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 norm_layer: Type[LayerNorm] = nn.LayerNorm,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3
                 ):
        HRTransStage2.__init__(self,
                               cfgs,
                               stage,
                               total_stage_num,
                               embed_dim,
                               window_size,
                               depths,
                               num_heads,
                               dpr,
                               mlp_ratio,
                               qkv_bias,
                               drop_rate,
                               attn_drop_rate,
                               norm_layer,
                               use_checkpoint,
                               spatial_dims)
        self.x4_us_1 = PatchExpanding(
            input_resolution=(
                self.cfgs.INPUT.RAND_CROP.ROI.Z//(self.cfgs.MODEL.SWIN.PATCH_SIZE*8),
                self.cfgs.INPUT.RAND_CROP.ROI.Y//(self.cfgs.MODEL.SWIN.PATCH_SIZE*8),
                self.cfgs.INPUT.RAND_CROP.ROI.X//(self.cfgs.MODEL.SWIN.PATCH_SIZE*8)
            ),
            dim=int(embed_dim * 2 ** 3)
        )

        self.x4_us_2 = PatchExpanding(
            input_resolution=(
                self.cfgs.INPUT.RAND_CROP.ROI.Z//(self.cfgs.MODEL.SWIN.PATCH_SIZE*4),
                self.cfgs.INPUT.RAND_CROP.ROI.Y//(self.cfgs.MODEL.SWIN.PATCH_SIZE*4),
                self.cfgs.INPUT.RAND_CROP.ROI.X//(self.cfgs.MODEL.SWIN.PATCH_SIZE*4)
            ),
            dim=int(embed_dim * 2 ** 2)
        )

        self.x4_us_3 = PatchExpanding(
            input_resolution=(
                self.cfgs.INPUT.RAND_CROP.ROI.Z//(self.cfgs.MODEL.SWIN.PATCH_SIZE*2),
                self.cfgs.INPUT.RAND_CROP.ROI.Y//(self.cfgs.MODEL.SWIN.PATCH_SIZE*2),
                self.cfgs.INPUT.RAND_CROP.ROI.X//(self.cfgs.MODEL.SWIN.PATCH_SIZE*2)
            ),
            dim=int(embed_dim * 2 ** 1)
        )

        if self.using_ds_stem:
            self.x4_us_4 = PatchExpanding(
                input_resolution=(
                    self.cfgs.INPUT.RAND_CROP.ROI.Z // self.cfgs.MODEL.SWIN.PATCH_SIZE,
                    self.cfgs.INPUT.RAND_CROP.ROI.Y // self.cfgs.MODEL.SWIN.PATCH_SIZE,
                    self.cfgs.INPUT.RAND_CROP.ROI.X // self.cfgs.MODEL.SWIN.PATCH_SIZE
                ),
                dim=int(embed_dim)
            )

    def forward(self, x_in):
        results_x = []
        results_x_ds = []
        if self.using_ds_stem:
            x_embedding = x_in[0]
            x_in = x_in[1:]

        for i in range(self.stage_val+1):
            x = self.stages_module[i](x_in[i])
            results_x.append(x)

            if i != self.total_stage_num-1:
                x_dc = rearrange(x, "b c d h w -> b d h w c").contiguous()
                x_ds = self.stage_ds_module[i](x_dc)
                x_ds = rearrange(x_ds, "b d h w c -> b c d h w")
                results_x_ds.append(x_ds)

        x1_ds_2 = self.x1_ds_2(rearrange(results_x_ds[0], "b c d h w -> b d h w c").contiguous())
        x1_ds_3 = self.x1_ds_3(x1_ds_2)
        x1_ds_2 = rearrange(x1_ds_2, "b d h w c-> b c d h w").contiguous()
        x1_ds_3 = rearrange(x1_ds_3, "b d h w c-> b c d h w").contiguous()

        x2_us = self.x2_us(results_x[1])
        x2_ds_2 = self.x2_ds_2(rearrange(results_x_ds[1], "b c d h w -> b d h w c").contiguous())
        x2_ds_2 = rearrange(x2_ds_2, "b d h w c -> b c d h w").contiguous()

        x3_us_1 = self.x3_us_1(results_x[2])
        x3_us_2 = self.x3_us_2(x3_us_1)

        x4_us_1 = self.x4_us_1(results_x[3])
        x4_us_2 = self.x4_us_2(x4_us_1)
        x4_us_3 = self.x4_us_3(x4_us_2)

        if self.using_ds_stem:
            x0 = self.x0_trans(x_embedding)
            x0_ds_1 = self.x0_ds_1(rearrange(x0, "b c d h w -> b d h w c"))
            x0_ds_2 = self.x0_ds_2(x0_ds_1)
            x0_ds_3 = self.x0_ds_3(x0_ds_2)
            x0_ds_4 = self.x0_ds_4(x0_ds_3)
            x0_ds_1 = rearrange(x0_ds_1, "b d h w c -> b c d h w")
            x0_ds_2 = rearrange(x0_ds_2, "b d h w c -> b c d h w")
            x0_ds_3 = rearrange(x0_ds_3, "b d h w c -> b c d h w")
            x0_ds_4 = rearrange(x0_ds_4, "b d h w c -> b c d h w")

            x1_us = self.x1_up(results_x[0])
            x2_us_2 = self.x2_us_2(x2_us)

            x3_us_3 = self.x3_us_3(x3_us_2)

            x4_us_4 = self.x4_us_4(x4_us_3)

            x0 = torch.cat([x0, x1_us, x2_us_2, x3_us_3, x4_us_4], dim=1)
            x1 = torch.cat([x0_ds_1, results_x[0], x2_us, x3_us_2, x4_us_3], dim=1)
            x2 = torch.cat([x0_ds_2, results_x_ds[0], results_x[1], x3_us_1, x4_us_2], dim=1)
            x3 = torch.cat([x0_ds_3, x1_ds_2, results_x_ds[1], results_x[2], x4_us_1], dim=1)
            x4 = torch.cat([x0_ds_4, x1_ds_3, x2_ds_2, results_x_ds[2], results_x[-1]], dim=1)

            if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                x0 = self.stage_fusion[0](x0)
                x1 = self.stage_fusion[1](x1)
                x2 = self.stage_fusion[2](x2)
                x3 = self.stage_fusion[3](x3)
                x4 = self.stage_fusion[4](x4)
            elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                x0 = self.stage_fusion[0](rearrange(x0, "b c d h w -> b d h w c"))
                x1 = self.stage_fusion[1](rearrange(x1, "b c d h w -> b d h w c"))
                x2 = self.stage_fusion[2](rearrange(x2, "b c d h w -> b d h w c"))
                x3 = self.stage_fusion[3](rearrange(x3, "b c d h w -> b d h w c"))
                x4 = self.stage_fusion[4](rearrange(x4, "b c d h w -> b d h w c"))
                x0 = rearrange(x0, "b d h w c -> b c d h w")
                x1 = rearrange(x1, "b d h w c -> b c d h w")
                x2 = rearrange(x2, "b d h w c -> b c d h w")
                x3 = rearrange(x3, "b d h w c -> b c d h w")
                x4 = rearrange(x4, "b d h w c -> b c d h w")
            return x0, x1, x2, x3, x4
        else:
            x1 = torch.cat([results_x[0], x2_us, x3_us_2, x4_us_3], dim=1)
            x2 = torch.cat([results_x_ds[0], results_x[1], x3_us_1, x4_us_2], dim=1)
            x3 = torch.cat([x1_ds_2, results_x_ds[1], results_x[2], x4_us_1], dim=1)
            x4 = torch.cat([x1_ds_3, x2_ds_2, results_x_ds[2], results_x[-1]], dim=1)
            if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
                x1 = self.stage_fusion[0](x1)
                x2 = self.stage_fusion[1](x2)
                x3 = self.stage_fusion[2](x3)
                x4 = self.stage_fusion[3](x4)
            elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
                x1 = self.stage_fusion[0](rearrange(x1, "b c d h w -> b d h w c"))
                x2 = self.stage_fusion[1](rearrange(x2, "b c d h w -> b d h w c"))
                x3 = self.stage_fusion[2](rearrange(x3, "b c d h w -> b d h w c"))
                x4 = self.stage_fusion[3](rearrange(x4, "b c d h w -> b d h w c"))
                x1 = rearrange(x1, "b d h w c -> b c d h w")
                x2 = rearrange(x2, "b d h w c -> b c d h w")
                x3 = rearrange(x3, "b d h w c -> b c d h w")
                x4 = rearrange(x4, "b d h w c -> b c d h w")
            return x1, x2, x3, x4


class HRTransStages(nn.Module):
    def __init__(self,
                 cfgs,
                 stage: int,
                 embed_dim: int,
                 window_size: Sequence[int],
                 depths: Sequence[int],
                 num_heads: Sequence[int],
                 dpr,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 norm_layer: Type[LayerNorm] = nn.LayerNorm,
                 use_checkpoint: bool = False,
                 spatial_dims: int = 3):
        super(HRTransStages, self).__init__()
        self.cfgs = cfgs
        self.using_ds_stem = self.cfgs.MODEL.HR_TRANS.USING_DS_STEM
        self.stage = stage
        self.window_size = ensure_tuple_rep(window_size, spatial_dims)

        stage_cls = [HRTransStage, HRTransStage1, HRTransStage2, HRTransStage3]

        self.total_stages = nn.ModuleList()

        for i in range(stage):
            self.total_stages.append(
                stage_cls[i](
                    cfgs=cfgs,
                    stage=i,
                    total_stage_num=self.stage,
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
            )

    def forward(self, x):
        output = x
        for i in range(self.stage):
            output = self.total_stages[i](output)
        return output


class FinalStage(nn.Module):
    def __init__(self, cfgs, embed_dim, spatial_dims, stage_num):
        super(FinalStage, self).__init__()
        assert stage_num > 1, "Stage Number should larger than 1."

        self.stage_num = stage_num
        self.cfgs = cfgs
        self.stage_ops = torch.nn.ModuleList()

        for i in range(self.stage_num):
            op_lists = []
            if i != 0:
                for j in range(i):
                    patch_expanding = PatchExpanding(
                        input_resolution=(
                            self.cfgs.INPUT.RAND_CROP.ROI.Z // (self.cfgs.MODEL.SWIN.PATCH_SIZE * 2**(i-j)),
                            self.cfgs.INPUT.RAND_CROP.ROI.Y // (self.cfgs.MODEL.SWIN.PATCH_SIZE * 2**(i-j)),
                            self.cfgs.INPUT.RAND_CROP.ROI.X // (self.cfgs.MODEL.SWIN.PATCH_SIZE * 2**(i-j))
                        ),
                        dim=int(embed_dim * 2 ** (i-j))
                    )
                    op_lists.append(patch_expanding)
            if self.cfgs.MODEL.HR_TRANS.USING_DS_STEM:
                op_lists.append(
                    PatchExpanding(
                        input_resolution=(
                            self.cfgs.INPUT.RAND_CROP.ROI.Z // self.cfgs.MODEL.SWIN.PATCH_SIZE,
                            self.cfgs.INPUT.RAND_CROP.ROI.Y // self.cfgs.MODEL.SWIN.PATCH_SIZE,
                            self.cfgs.INPUT.RAND_CROP.ROI.X // self.cfgs.MODEL.SWIN.PATCH_SIZE
                        ),
                        dim=int(embed_dim)
                    )
                )
            if len(op_lists) > 0:
                self.stage_ops.append(nn.Sequential(*op_lists))
        if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
            if self.cfgs.MODEL.HR_TRANS.USING_DS_STEM:
                self.stages_fusion = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=int(embed_dim // 2) * (self.stage_num+1),
                    out_channels=int(embed_dim // 2),
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True
                )
            else:
                self.stages_fusion = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=int(embed_dim) * self.stage_num,
                    out_channels=int(embed_dim),
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True
                )
        elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
            if self.cfgs.MODEL.HR_TRANS.USING_DS_STEM:
                self.stages_fusion = MLPBlock(
                    hidden_size=int(embed_dim // 2) * (self.stage_num+1),
                    mlp_dim=(int(embed_dim // 2) * (self.stage_num+1))//2,
                    output_dim=int(embed_dim // 2),
                    dropout_rate=self.cfgs.MODEL.HR_TRANS.DROPOUT_RATE
                )
            else:
                self.stages_fusion = MLPBlock(
                    hidden_size=int(embed_dim) * self.stage_num,
                    mlp_dim=(int(embed_dim) * self.stage_num)//2,
                    output_dim=int(embed_dim),
                    dropout_rate=self.cfgs.MODEL.HR_TRANS.DROPOUT_RATE
                )
        else:
            raise NotImplementedError(f"the fusion model {self.cfgs.MODEL.HR_TRANS.FUSION_TYPE} "
                                      f"has not implemented!")

    def forward(self, x_in):
        x, x_in = x_in[0], x_in[1:]
        outputs = []
        for input, ops in zip(x_in, self.stage_ops):
            outputs.append(ops(input))
        outputs.insert(0, x)
        x = torch.cat(outputs, dim=1)
        if self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "res_conv":
            x = self.stages_fusion(x)
        elif self.cfgs.MODEL.HR_TRANS.FUSION_TYPE == "fc":
            x = self.stages_fusion(rearrange(x, "b c d h w -> b d h w c"))
            x = rearrange(x, "b d h w c -> b c d h w")
        return x


class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        output_dim: int,
        dropout_rate: float = 0.0,
        act: Union[Tuple, str] = "GELU",
        dropout_mode="vit",
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: faction of the input units to drop.
            act: activation type and arguments. Defaults to GELU.
            dropout_mode: dropout mode, can be "vit" or "swin".
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, output_dim)
        self.fn = get_act_layer(act)
        self.drop1 = nn.Dropout(dropout_rate)
        dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
        if dropout_opt == "vit":
            self.drop2 = nn.Dropout(dropout_rate)
        elif dropout_opt == "swin":
            self.drop2 = self.drop1
        else:
            raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x