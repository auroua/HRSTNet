from fvcore.common.checkpoint import Checkpointer
import seg.utils.dist_utils as dist_utils
# from seg.utils.file_io import PathManager
# import logging
import os


class SegCheckpointer(Checkpointer):
    def __init__(self, model, cfg, **checkpointables):
        is_main_process = dist_utils.is_main_process()
        super().__init__(
            model,
            cfg.OUTPUT_DIR,
            save_to_disk=is_main_process,
            **checkpointables
        )
        self.cfg = cfg
        # self.path_manager = PathManager
        # self.logger = logging.getLogger(__name__)

    def load(self, path, checkpointables=None, **kwargs):
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("[Checkpointer] Loading from {} ...".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)
        checkpoint = self._load_file(path)
        # start_dict, best_acc = print(list(checkpoint.keys()))
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in kwargs and "optimizer" in checkpoint:
            kwargs["optimizer"].load_state_dict(checkpoint["optimizer"])
            self.logger.info("[Optimizer] Loading from {} ...".format(path))
        if "scheduler" in kwargs and "scheduler" in checkpoint:
            kwargs["scheduler"].load_state_dict(checkpoint["scheduler"])
            self.logger.info("[Scheduler] Loading from {} ...".format(path))
        epoch, best_acc = checkpoint["epoch"], checkpoint["best_acc"]
        del checkpoint
        return epoch, best_acc

    def save(self, **kwargs):
        super(SegCheckpointer, self).save(self.cfg.MODEL.NAME + "_" + str(kwargs["epoch"]), **kwargs)