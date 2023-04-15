import seg.utils.dist_utils as dist_utils
from torch.nn.parallel import DistributedDataParallel
import torch


def create_ddp_model(model, cfgs, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa
    if dist_utils.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [dist_utils.get_local_rank()]
        kwargs["output_device"] = dist_utils.get_local_rank()
        kwargs["find_unused_parameters"] = True
    if cfgs.MODEL.NORM == "batch":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp = DistributedDataParallel(model, **kwargs)
    return ddp