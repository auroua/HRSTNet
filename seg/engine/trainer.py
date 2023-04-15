# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import shutil
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from seg.utils.dist_utils import distributed_all_gather
import seg.utils.dist_utils as dist_utils
import torch.utils.data.distributed
from monai.data import decollate_batch
from datasets.builder import get_dataloader
from seg.models.builder import get_model
import logging
from seg.solver.build import build_optimizer, build_lr_schedule, build_loss, build_val_post_processing, build_loss_val
from seg.engine.defaults import create_ddp_model
from seg.checkpoint.seg_checkpoint import SegCheckpointer
from functools import partial
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from monai import transforms, data
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from monai.data import load_decathlon_datalist
from monai.transforms import LoadImage
from seg.utils.solver_utils import dice, AverageMeter
from seg.utils.solver_utils import hd
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
from monai.utils.enums import MetricReduction
import nibabel as nib
from einops import rearrange
import SimpleITK as sitk
from numpy import logical_and as l_and, logical_not as l_not
import pprint
import pandas as pd
import glob


HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
SSIM = "ssim"
# METRICS = [HAUSSDORF, DICE, SENS, SPEC, SSIM]
METRICS = [HAUSSDORF, DICE]


def train_epoch(model,
                loader,
                optimizer,
                epoch,
                loss_func,
                cfgs,
                scaler):
    logger = logging.getLogger(__name__)
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    rank = dist_utils.get_rank()

    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            if cfgs.DATASETS.NAME == "BraTS_2021":
                t1, t1ce, t2, flair = batch_data["t1"], batch_data["t1ce"], batch_data["t2"], batch_data["flair"]
                data = torch.cat([t1, t1ce, t2, flair], dim=1)
                target = batch_data["seg"]
            else:
                data, target = batch_data['image'], batch_data['label']

        if cfgs.MODEL.NAME == "extending_nnunet" and cfgs.MODEL.EXTENDING_NN_UNET.DEEP_SUPERVISION:
            data = data.cuda(rank)
            target = [t.cuda(rank) for t in target]
        else:
            data, target = data.cuda(rank), target.cuda(rank)
        for param in model.parameters(): param.grad = None
        with autocast(enabled=cfgs.SOLVER.AMP):
            logits = model(data)

            if cfgs.MODEL.NAME == "dynunet" and cfgs.MODEL.DYNUNET.DEEP_SUPERVISION:
                loss, weights = 0.0, 0.0
                for i in range(logits.size(1)):
                    loss += loss_func(logits[:, i, ...], target) * 0.5 ** i
                    weights += 0.5 ** i
                loss = loss / weights
            elif cfgs.MODEL.NAME == "extending_nnunet" and cfgs.MODEL.EXTENDING_NN_UNET.DEEP_SUPERVISION:
                loss = loss_func(logits[0], target[0])
                loss = loss * cfgs.MODEL.EXTENDING_NN_UNET.WEIGHT_FACTOR[0]
                for index, w in enumerate(cfgs.MODEL.EXTENDING_NN_UNET.WEIGHT_FACTOR[1:]):
                    if w != 0:
                        loss += w * loss_func(logits[index+1], target[index+1])
            else:
                loss = loss_func(logits, target)
        if cfgs.SOLVER.AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if dist_utils.get_world_size() > 1:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=cfgs.SOLVER.BATCH_SIZE * dist_utils.get_world_size())
        else:
            run_loss.update(loss.item(), n=cfgs.SOLVER.BATCH_SIZE)
        if dist_utils.is_main_process():
            logger.info(f'Epoch {epoch}/{cfgs.SOLVER.EPOCHS} {idx}/{len(loader)}, '
                        f'loss: {run_loss.avg}, time: {time.time() - start_time}, '
                        f"lr: {optimizer.param_groups[0]['lr']}.")
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model,
              loader,
              epoch,
              acc_func,
              cfgs,
              model_inferer=None,
              post_label=None,
              post_pred=None):
    model.eval()
    start_time = time.time()
    rank = dist_utils.get_rank()
    run_acc = AverageMeter()
    logger = logging.getLogger(__name__)
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']

            data, target = data.cuda(rank), target.cuda(rank)
            with autocast(enabled=cfgs.SOLVER.AMP):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(rank)

            if dist_utils.get_world_size() > 1:
                acc_list = distributed_all_gather([acc],
                                                  out_numpy=True,
                                                  is_valid=idx < loader.sampler.valid_length)
                # if dist_utils.is_main_process():
                #     if len(acc_list[0]) > 1:
                #         print(acc_list[0][0], acc_list[0][1], acc_list[0][0].shape, acc_list[0][1].shape)
                if cfgs.DATASETS.NAME == "MSD":
                    if cfgs.DATASETS.MSD_TYPE == "Spleen":
                        avg_acc = np.mean([np.nanmean(l) for l in acc_list[0]])
                        run_acc.update(avg_acc)
                    elif cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas":
                        for v in acc_list[0]:
                            run_acc.update(v[0])
                else:
                    avg_acc = np.mean([np.nanmean(l) for l in acc_list])
                    run_acc.update(avg_acc)
            else:
                if cfgs.DATASETS.NAME == "MSD":
                    if cfgs.DATASETS.MSD_TYPE == "Spleen":
                        acc_list = acc.detach().cpu().numpy()
                        avg_acc = np.mean([np.nanmean(l) for l in acc_list])
                        run_acc.update(avg_acc)
                    elif cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas":
                        acc_list = acc.detach().cpu().numpy()[0]
                        run_acc.update(acc_list)
                else:
                    acc_list = acc.detach().cpu().numpy()
                    avg_acc = np.mean([np.nanmean(l) for l in acc_list])
                    run_acc.update(avg_acc)

            if dist_utils.is_main_process():
                if cfgs.DATASETS.NAME == "MSD":
                    if cfgs.DATASETS.MSD_TYPE == "Spleen":
                        logger.info(f"Val {epoch}/{cfgs.SOLVER.EPOCHS} {idx}/{len(loader)}, "
                                    f"acc: {run_acc.val}, time {time.time() - start_time}")
                    elif cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas":
                        Dice_Orgin = run_acc.val[0]
                        Dice_Tumor = run_acc.val[1]
                        logger_output_str = f"Val {epoch}/{cfgs.SOLVER.EPOCHS} {idx}/{len(loader)}, " \
                                            f"Dice_Orgin: {Dice_Orgin}, Dice_Tumor: {Dice_Tumor}, " \
                                            f"time {time.time() - start_time}s"
                        logger.info(logger_output_str)
                else:
                    logger.info(f"Val {epoch}/{cfgs.SOLVER.EPOCHS} {idx}/{len(loader)}, "
                                f"acc: {run_acc.val}, time {time.time() - start_time}")
            start_time = time.time()
    return run_acc.avg


def val_epoch_brats_monai(model,
                          loader,
                          epoch,
                          acc_func,
                          cfgs,
                          model_inferer=None,
                          post_sigmoid=None,
                          post_pred=None):
    model.eval()
    start_time = time.time()
    rank = dist_utils.get_rank()
    logger = logging.getLogger(__name__)
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data['image'], batch_data['label']
            data, target = data.cuda(rank), target.cuda(rank)
            # if cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
            #     data = rearrange(data, "n c h w d -> n c d h w")
            with autocast(enabled=cfgs.SOLVER.AMP):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            # if cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
            #     logits = rearrange(logits, "n c d h w -> n c h w d")
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=[val > 0.5 for val in val_output_convert], y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(rank)
            if dist_utils.get_world_size() > 1:
                acc_list, not_nans_list = distributed_all_gather([acc, not_nans],
                                                                 out_numpy=True,
                                                                 is_valid=idx < loader.sampler.valid_length)
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            if rank == 0:
                Dice_TC = run_acc.val[0]
                Dice_WT = run_acc.val[1]
                Dice_ET = run_acc.val[2]
                logger_output_str = f"Val {epoch}/{cfgs.SOLVER.EPOCHS} {idx}/{len(loader)}, " \
                                    f"Dice_TC: {Dice_TC}, Dice_WT: {Dice_WT}, Dice_ET: {Dice_ET}, " \
                                    f"time {time.time() - start_time}s"

                logger.info(logger_output_str)
            start_time = time.time()
    return run_acc.avg


def val_epoch_brats(model,
                    loader,
                    epoch,
                    acc_func,
                    cfgs,
                    model_inferer=None,
                    post_pred=None,
                    metric=None,
                    ):
    model.eval()
    start_time = time.time()
    rank = dist_utils.get_rank()
    logger = logging.getLogger(__name__)
    total_metrics = []
    et_metrics = []
    tc_metrics = []
    wt_metrics = []

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                if cfgs.DATASETS.NAME == "BraTS_2021":
                    t1, t1ce, t2, flair = batch_data["t1"], batch_data["t1ce"], batch_data["t2"], batch_data["flair"]
                    data = torch.cat([t1, t1ce, t2, flair], dim=1)
                    target = batch_data["seg"]
                else:
                    data, target = batch_data['image'], batch_data['label']
            data, target = data.cuda(rank), target.cuda(rank)
            with autocast(enabled=cfgs.SOLVER.AMP):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            acc = acc_func(y_pred=val_output_convert, y=target)
            acc = acc.cuda(rank)

            metric_ = metric(logits, target)
            et = metric_[0][0]
            tc = metric_[0][1]
            wt = metric_[0][2]
            et = et.cuda(rank)
            tc = tc.cuda(rank)
            wt = wt.cuda(rank)

            if dist_utils.get_world_size() > 1:
                acc_list = distributed_all_gather([acc],
                                                  out_numpy=True,
                                                  is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

                et_dist_list = distributed_all_gather([et],
                                                      out_numpy=True,
                                                      is_valid=idx < loader.sampler.valid_length)
                # if dist_utils.is_main_process():
                #     print(type(et_dist_list), et_dist_list, et_dist_list[0])
                et_metrics.extend(et_dist_list[0])

                tc_dist_list = distributed_all_gather([tc],
                                                      out_numpy=True,
                                                      is_valid=idx < loader.sampler.valid_length)
                tc_metrics.extend(tc_dist_list[0])

                wt_dist_list = distributed_all_gather([wt],
                                                      out_numpy=True,
                                                      is_valid=idx < loader.sampler.valid_length)
                wt_metrics.extend(wt_dist_list[0])

                # if dist_utils.is_main_process():
                #     print(len(et_dist_list), len(et_metrics), len(tc_dist_list), len(tc_metrics),
                #           len(wt_dist_list), len(wt_metrics))
            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

                total_metrics.extend(metric_)
            if dist_utils.is_main_process():
                logger.info(f"Val {epoch}/{cfgs.SOLVER.EPOCHS} {idx}/{len(loader)}, "
                            f"acc: {avg_acc}, time {time.time() - start_time}")
            start_time = time.time()

    dice_values = acc_func.aggregate().item()
    acc_func.reset()

    if dist_utils.get_world_size() > 1:
        if dist_utils.is_main_process():
            et_mean = np.mean(et_metrics)
            tc_mean = np.mean(tc_metrics)
            wt_mean = np.mean(wt_metrics)
            output_info = f"Epoch {epoch} :{'Val :'}" + f"Val: ET: {et_mean}, TC: {tc_mean}, WT: {wt_mean}"
            logger.info(output_info)
    else:
        metrics = list(zip(*total_metrics))
        # print(metrics)
        # TODO check if doing it directly to numpy work
        metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
        # print(metrics)
        labels = ("ET", "TC", "WT")
        metrics = {key: value for key, value in zip(labels, metrics)}

        output_info = f"Epoch {epoch} :{'Val :'}" + str([f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
        logger.info(output_info)

    return dice_values


def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 acc_func,
                 cfgs,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 post_label=None,
                 post_pred=None,
                 best_acc=0.,
                 checkpointer=None
                 ):
    logger = logging.getLogger(__name__)
    loss_fn_val = build_loss_val()
    metric = loss_fn_val.metric

    if cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
        semantic_classes = ['Dice_Val_TC', 'Dice_Val_WT', 'Dice_Val_ET']
    elif cfgs.DATASETS.NAME == "MSD":
        if cfgs.DATASETS.MSD_TYPE == "Pancreas" or cfgs.DATASETS.MSD_TYPE == "Liver":
            semantic_classes = ['Dice_Val_Organ', 'Dice_Val_Tumor']
    else:
        semantic_classes = None

    writer = None
    if cfgs.OUTPUT_DIR is not None and dist_utils.is_main_process():
        writer = SummaryWriter(log_dir=cfgs.OUTPUT_DIR)
        if dist_utils.is_main_process():
            logger.info(f'Writing Tensorboard logs to {cfgs.OUTPUT_DIR}')
    scaler = None
    if cfgs.SOLVER.AMP:
        scaler = GradScaler()
    val_acc_max = best_acc
    for epoch in range(start_epoch, cfgs.SOLVER.EPOCHS):
        if dist_utils.get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        logger.info(f"{dist_utils.get_rank()}, {time.ctime()}, Epoch:, {epoch}")
        epoch_time = time.time()
        train_loss = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 epoch=epoch,
                                 loss_func=loss_func,
                                 cfgs=cfgs,
                                 scaler=scaler)
        if dist_utils.is_main_process():
            logger.info(f"Final training  {epoch}/{cfgs.SOLVER.EPOCHS - 1}, "
                        f"loss: {train_loss}, "
                        f"time {time.time() - epoch_time}s")
        if dist_utils.is_main_process() and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
        b_new_best = False
        if (epoch+1) % cfgs.SOLVER.EVAL_PERIOD == 0:
            if dist_utils.get_world_size() > 1:
                torch.distributed.barrier()
            epoch_time = time.time()
            if cfgs.DATASETS.NAME == "BraTS_2021" or cfgs.DATASETS.NAME == "BraTS_2021_VT_UNET":
                val_avg_acc = val_epoch_brats(model,
                                              val_loader,
                                              epoch=epoch,
                                              acc_func=acc_func,
                                              model_inferer=model_inferer,
                                              cfgs=cfgs,
                                              post_pred=post_pred,
                                              metric=metric)
            elif cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
                val_avg_acc = val_epoch_brats_monai(model,
                                                    val_loader,
                                                    epoch=epoch,
                                                    acc_func=acc_func,
                                                    model_inferer=model_inferer,
                                                    cfgs=cfgs,
                                                    post_pred=post_pred,
                                                    post_sigmoid=post_label)
            else:
                val_avg_acc = val_epoch(model,
                                        val_loader,
                                        epoch=epoch,
                                        acc_func=acc_func,
                                        model_inferer=model_inferer,
                                        cfgs=cfgs,
                                        post_label=post_label,
                                        post_pred=post_pred)
            if dist_utils.is_main_process() and cfgs.OUTPUT_DIR is not None:
                checkpointer.save(
                    epoch=epoch,
                    best_acc=float(val_acc_max)
                )
            if cfgs.DATASETS.NAME == "BraTS_2021_MONAI" and dist_utils.is_main_process():
                Dice_TC = val_avg_acc[0]
                Dice_WT = val_avg_acc[1]
                Dice_ET = val_avg_acc[2]
                logger_output_str = f"Final validation stats {epoch}/{cfgs.SOLVER.EPOCHS - 1}, " \
                                    f"Dice_TC: {Dice_TC}, Dice_WT: {Dice_WT}, Dice_ET: {Dice_ET}, " \
                                    f"time {time.time() - epoch_time}s"
                logger.info(logger_output_str)
                if writer is not None:
                    writer.add_scalar('Mean_Val_Dice', np.mean(val_avg_acc), epoch)
                    if semantic_classes is not None:
                        for val_channel_ind in range(len(semantic_classes)):
                            if val_channel_ind < val_avg_acc.size:
                                writer.add_scalar(semantic_classes[val_channel_ind], val_avg_acc[val_channel_ind], epoch)
                val_avg_acc = np.mean(val_avg_acc)
                if val_avg_acc > val_acc_max:
                    logger.info(f'new best ({val_acc_max} --> {val_avg_acc}).')
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    # if dist_utils.is_main_process() and cfgs.OUTPUT_DIR is not None:
                    #     checkpointer.save(
                    #         epoch=epoch,
                    #         best_acc=float(val_acc_max)
                    #     )
                if b_new_best:
                    logger.info('Copying to model.pt new best model!!!!')
                    shutil.copyfile(os.path.join(cfgs.OUTPUT_DIR, f'{cfgs.MODEL.NAME}_{epoch}.pth'),
                                    os.path.join(cfgs.OUTPUT_DIR, 'model_best.pth'))
            elif cfgs.DATASETS.NAME == "MSD" and dist_utils.is_main_process():
                if cfgs.DATASETS.MSD_TYPE == "Pancreas" or cfgs.DATASETS.MSD_TYPE == "Liver":
                    Dice_Organ = val_avg_acc[0]
                    Dice_Tumor = val_avg_acc[1]
                    logger_output_str = f"Final validation stats {epoch}/{cfgs.SOLVER.EPOCHS - 1}, " \
                                        f"Dice_Organ: {Dice_Organ}, Dice_Tumor: {Dice_Tumor}, " \
                                        f"time {time.time() - epoch_time}s"
                    logger.info(logger_output_str)
                    if writer is not None:
                        writer.add_scalar('Mean_Val_Dice', np.mean(val_avg_acc), epoch)
                        if semantic_classes is not None:
                            for val_channel_ind in range(len(semantic_classes)):
                                if val_channel_ind < val_avg_acc.size:
                                    writer.add_scalar(semantic_classes[val_channel_ind], val_avg_acc[val_channel_ind],
                                                      epoch)
                    val_avg_acc = np.mean(val_avg_acc)
                    if val_avg_acc > val_acc_max:
                        logger.info(f'new best ({val_acc_max} --> {val_avg_acc}).')
                        val_acc_max = val_avg_acc
                        b_new_best = True
                        # if dist_utils.is_main_process() and cfgs.OUTPUT_DIR is not None:
                        #     checkpointer.save(
                        #         epoch=epoch,
                        #         best_acc=float(val_acc_max)
                        #     )
                    if b_new_best:
                        logger.info('Copying to model.pt new best model!!!!')
                        shutil.copyfile(os.path.join(cfgs.OUTPUT_DIR, f'{cfgs.MODEL.NAME}_{epoch}.pth'),
                                        os.path.join(cfgs.OUTPUT_DIR, 'model_best.pth'))
                else:
                    logger.info(f"Final validation  {epoch}/{cfgs.SOLVER.EPOCHS - 1}, "
                                f"acc, {val_avg_acc}, time {time.time() - epoch_time}s")
                    if writer is not None:
                        writer.add_scalar('val_acc', val_avg_acc, epoch)
                        if val_avg_acc > val_acc_max:
                            logger.info(f'new best ({val_acc_max} --> {val_avg_acc}).')
                            val_acc_max = val_avg_acc
                            b_new_best = True
                        if b_new_best:
                            logger.info('Copying to model.pt new best model!!!!')
                            shutil.copyfile(os.path.join(cfgs.OUTPUT_DIR, f'{cfgs.MODEL.NAME}_{epoch}.pth'),
                                            os.path.join(cfgs.OUTPUT_DIR, 'model_best.pth'))
            else:
                if dist_utils.is_main_process():
                    logger.info(f"Final validation  {epoch}/{cfgs.SOLVER.EPOCHS - 1}, "
                                f"acc, {val_avg_acc}, time {time.time() - epoch_time}s")
                    if writer is not None:
                        writer.add_scalar('val_acc', val_avg_acc, epoch)
                if dist_utils.is_main_process() and cfgs.OUTPUT_DIR is not None:
                    # checkpointer.save(
                    #     epoch=epoch,
                    #     best_acc=float(val_acc_max)
                    # )
                    if val_avg_acc > val_acc_max:
                        logger.info(f'new best ({val_acc_max} --> {val_avg_acc}).')
                        val_acc_max = val_avg_acc
                        b_new_best = True
                    if b_new_best:
                        logger.info('Copying to model.pt new best model!!!!')
                        shutil.copyfile(os.path.join(cfgs.OUTPUT_DIR, f'{cfgs.MODEL.NAME}_{epoch}.pth'),
                                        os.path.join(cfgs.OUTPUT_DIR, 'model_best.pth'))

        if scheduler is not None:
            scheduler.step()

    # if dist_utils.is_main_process() and cfgs.OUTPUT_DIR is not None:
    #     checkpointer.save(
    #         epoch=cfgs.SOLVER.EPOCHS+1,
    #         best_acc=float(val_acc_max)
    #     )
    logger.info(f'Training Finished !, Best Accuracy: {val_acc_max}')

    return val_acc_max


def run_validate(model,
                 val_loader,
                 acc_func,
                 cfgs,
                 model_inferer=None,
                 post_pred=None,
                 ):
    loss_fn_val = build_loss_val()
    metric = loss_fn_val.metric
    val_avg_acc = val_epoch_brats(model,
                                  val_loader,
                                  epoch=300,
                                  acc_func=acc_func,
                                  model_inferer=model_inferer,
                                  cfgs=cfgs,
                                  post_pred=post_pred,
                                  metric=metric)
    return val_avg_acc


class Trainer:
    def __init__(self, cfgs):
        if cfgs.MODE == "train":
            self.train_loader, self.val_loader = get_dataloader(cfgs)
        else:
            self.val_loader = get_dataloader(cfgs)
        self.model = get_model(cfgs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(f"{device}:{dist_utils.get_local_rank()}")
        self.model = create_ddp_model(self.model, cfgs)
        self.start_epochs = 0
        self.best_acc = 0
        self.cfg = cfgs
        if cfgs.MODE == "train":
            self.loss_fn = build_loss(cfgs)
            self.optimizer = build_optimizer(cfg=cfgs, model=self.model)
            self.scheduler = build_lr_schedule(cfgs, self.optimizer)
            self.checkpointer = SegCheckpointer(model=self.model,
                                                cfg=cfgs,
                                                scheduler=self.scheduler,
                                                optimizer=self.optimizer)
        elif cfgs.MODE == "test":
            self.checkpointer = SegCheckpointer(model=self.model,
                                                cfg=cfgs)
        else:
            raise NotImplementedError(f"The mode type {cfgs.MODE} doesn't support at present!")
        self.logger = logging.getLogger(__name__)

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        if len(self.cfg.MODEL.WEIGHTS) == 0:
            return 0, 0
        if self.cfg.MODE == "train" and resume:
            start_epochs, best_acc = self.checkpointer.load(self.cfg.MODEL.WEIGHTS,
                                                            resume=resume,
                                                            optimizer=self.optimizer,
                                                            scheduler=self.scheduler)
            self.start_epochs = start_epochs
            self.best_acc = best_acc
        elif self.cfg.MODE == "test" and resume:
            start_epochs, best_acc = self.checkpointer.load(self.cfg.MODEL.WEIGHTS,
                                                            resume=resume)
            self.start_epochs = start_epochs
            self.best_acc = best_acc
        else:
            raise NotImplementedError(f"The mode type {self.cfg.MODE} doesn't support at present!")

    def train(self):
        # TODO: remove the following post processing and model eval code to a common place
        post_label, post_pred, acc_func = build_val_post_processing(self.cfg)
        model_inferer = partial(sliding_window_inference,
                                roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                                          self.cfg.INPUT.RAND_CROP.ROI.Y,
                                          self.cfg.INPUT.RAND_CROP.ROI.Z),
                                sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TRAIN,
                                predictor=self.model,
                                overlap=self.cfg.SOLVER.INFER_OVERLAP)
        run_training(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            loss_func=self.loss_fn,
            acc_func=acc_func,
            cfgs=self.cfg,
            model_inferer=model_inferer,
            scheduler=self.scheduler,
            start_epoch=self.start_epochs,
            post_label=post_label,
            post_pred=post_pred,
            best_acc=self.best_acc,
            checkpointer=self.checkpointer
        )

    def validate(self):
        # TODO: remove the following post processing and model eval code to a common place
        post_label, post_pred, acc_func = build_val_post_processing(self.cfg)
        model_inferer = partial(sliding_window_inference,
                                roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                                          self.cfg.INPUT.RAND_CROP.ROI.Y,
                                          self.cfg.INPUT.RAND_CROP.ROI.Z),
                                sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TRAIN,
                                predictor=self.model,
                                overlap=self.cfg.SOLVER.INFER_OVERLAP)
        val_result = run_validate(
            model=self.model,
            val_loader=self.val_loader,
            acc_func=acc_func,
            cfgs=self.cfg,
            model_inferer=model_inferer,
            post_pred=post_pred,
        )
        return val_result

    def inference(self, cfgs):
        if cfgs.DATASETS.NAME == "MSD":
            if cfgs.DATASETS.MSD_TYPE == "Spleen":
                dice_list_organ = []

                hausdorff_95_organ_list = []
            elif cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas":
                dice_list_organ = []
                dice_list_tumor = []

                hausdorff_95_organ_list = []
                hausdorff_95_tumor_list = []
        elif cfgs.DATASETS.NAME == "BraTS_2021" or cfgs.DATASETS.NAME == "BraTS_2021_VT_UNET":
            post_pred = Compose(
                [EnsureType(), Activations(sigmoid=True), AsDiscrete(logit_thresh=0.5, threshold_values=True)]
            )
            acc_func = DiceMetric(include_background=True,
                                  reduction=MetricReduction.MEAN_BATCH,
                                  get_not_nans=True)
            run_acc = AverageMeter()
            dice_list_et = []
            dice_list_tc = []
            dice_list_wt = []

            hausdorff_list_et = []
            hausdorff_list_tc = []
            hausdorff_list_wt = []
        elif cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
            post_pred = AsDiscrete(argmax=False, logit_thresh=0.5, threshold_values=True)
            acc_func = DiceMetric(include_background=True,
                                  reduction=MetricReduction.MEAN_BATCH,
                                  get_not_nans=True)
            run_acc = AverageMeter()
            dice_list_et = []
            dice_list_tc = []
            dice_list_wt = []

            hausdorff_list_et = []
            hausdorff_list_tc = []
            hausdorff_list_wt = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if cfgs.DATASETS.NAME == "BraTS_2021" or cfgs.DATASETS.NAME == "BraTS_2021_VT_UNET":
                    t1, t1ce, t2, flair = batch["t1"], batch["t1ce"], batch["t2"], batch["flair"]
                    val_inputs = torch.cat([t1, t1ce, t2, flair], dim=1)
                    val_labels = batch["seg"]
                else:
                    val_inputs, val_labels = (batch["image"], batch["label"])
                if cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
                    val_inputs = rearrange(val_inputs, "n c h w d -> n c d h w")
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                if cfgs.DATASETS.NAME == "BraTS_2021":
                    img_name = batch['t1_meta_dict']['filename_or_obj'][0].split('/')[-1]
                else:
                    img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                self.logger.info("Inference on case {}".format(img_name))
                val_outputs = sliding_window_inference(val_inputs,
                                                       roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                                                                 self.cfg.INPUT.RAND_CROP.ROI.Y,
                                                                 self.cfg.INPUT.RAND_CROP.ROI.Z),
                                                       sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TEST,
                                                       predictor=self.model,
                                                       overlap=self.cfg.SOLVER.INFER_OVERLAP)
                if cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
                    val_outputs = rearrange(val_outputs, "n c d h w -> n c h w d")
                if cfgs.DATASETS.NAME == "MSD":
                    val_outputs_sf = torch.softmax(val_outputs, 1).cpu().numpy()
                    val_outputs = np.argmax(val_outputs_sf, axis=1)
                    val_labels_np = val_labels.cpu().numpy()
                    val_labels = val_labels_np[:, 0, :, :, :]
                elif cfgs.DATASETS.NAME == "BraTS_2021":
                    val_outputs_list = decollate_batch(val_outputs)
                    val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                    val_labels_list = decollate_batch(val_labels)
                    acc_func.reset()
                    acc_func(y_pred=[val > 0.5 for val in val_output_convert], y=val_labels_list)
                    acc, not_nans = acc_func.aggregate()
                    acc = acc.cuda()
                    run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                    Dice_ET = float(run_acc.val[0])
                    Dice_TC = float(run_acc.val[1])
                    Dice_WT = float(run_acc.val[2])
                    dice_list_et.append(Dice_ET)
                    dice_list_tc.append(Dice_TC)
                    dice_list_wt.append(Dice_WT)

                    pred_output = val_output_convert[0].cpu().numpy()
                    target_label = val_labels_list[0].cpu().numpy()
                    hausdor_95_et = hd(pred_output[0, ...], target_label[0, ...])
                    hausdor_95_tc = hd(pred_output[1, ...], target_label[1, ...])
                    hausdor_95_wt = hd(pred_output[2, ...], target_label[2, ...])

                    hausdorff_list_et.append(hausdor_95_et)
                    hausdorff_list_tc.append(hausdor_95_tc)
                    hausdorff_list_wt.append(hausdor_95_wt)
                elif cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
                    val_outputs_list = decollate_batch(val_outputs)
                    val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                    val_labels_list = decollate_batch(val_labels)
                    acc_func.reset()
                    acc_func(y_pred=[val > 0.5 for val in val_output_convert], y=val_labels_list)
                    acc, not_nans = acc_func.aggregate()
                    acc = acc.cuda()
                    run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                    Dice_TC = float(run_acc.val[0])
                    Dice_WT = float(run_acc.val[1])
                    Dice_ET = float(run_acc.val[2])
                    dice_list_et.append(Dice_ET)
                    dice_list_tc.append(Dice_TC)
                    dice_list_wt.append(Dice_WT)

                    pred_output = val_output_convert[0].cpu().numpy()
                    target_label = val_labels_list[0].cpu().numpy()
                    hausdor_95_tc = hd(pred_output[0, ...], target_label[0, ...])
                    hausdor_95_wt = hd(pred_output[1, ...], target_label[1, ...])
                    hausdor_95_et = hd(pred_output[2, ...], target_label[2, ...])

                    hausdorff_list_et.append(hausdor_95_et)
                    hausdorff_list_tc.append(hausdor_95_tc)
                    hausdorff_list_wt.append(hausdor_95_wt)

                if cfgs.DATASETS.NAME == "MSD":
                    if cfgs.DATASETS.MSD_TYPE == "Spleen":
                        for i in range(1, cfgs.MODEL.OUT_CHANNEL):
                            organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                            dice_list_organ.append(organ_Dice)
                    elif cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas":
                        for i in range(1, cfgs.MODEL.OUT_CHANNEL):
                            if i == 1:
                                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                                dice_list_organ.append(organ_Dice)
                            elif i == 2:
                                tumor_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                                dice_list_tumor.append(tumor_Dice)
                            else:
                                raise ValueError("Illegal input value!")

                if cfgs.DATASETS.NAME == "MSD":
                    pred = val_outputs[0]
                    gt = val_labels[0]
                    if cfgs.DATASETS.MSD_TYPE == "Spleen":
                        organ_pred = pred == 1
                        organ_gt = gt == 1
                        hausdor_95_organ = hd(organ_pred, organ_gt)
                        hausdorff_95_organ_list.append(hausdor_95_organ)

                    elif cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas":
                        organ_pred = pred == 1
                        organ_gt = gt == 1
                        hausdor_95_organ = hd(organ_pred, organ_gt)

                        tumor_pred = pred == 2
                        tumor_gt = gt == 2
                        hausdor_95_tumor = hd(tumor_pred, tumor_gt)

                        hausdorff_95_organ_list.append(hausdor_95_organ)
                        hausdorff_95_tumor_list.append(hausdor_95_tumor)

                if cfgs.DATASETS.NAME == "MSD":
                    if cfgs.DATASETS.MSD_TYPE == "Spleen":
                        self.logger.info("Mean Organ Dice: {}, Mean organ Hausdorff_95: {}".format(organ_Dice, hausdor_95_organ))

                    elif cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas":
                        self.logger.info("Mean Organ Dice: {} Tumor Dice: {}, Mean organ Hausdorff_95: {} Mean tumor Hausdorff_95: {}".format(organ_Dice, tumor_Dice, hausdor_95_organ, hausdor_95_tumor))
                elif cfgs.DATASETS.NAME == "BraTS_2021" or cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
                    self.logger.info(
                        "ET Dice: {} TC Dice: {}, WT Dice: {}, ET Hausdorff_95: {} TC Hausdorff_95: {} WT Hausdorff_95: {}".format(
                            Dice_ET, Dice_TC, Dice_WT, hausdor_95_et, hausdor_95_tc, hausdor_95_wt))

            if cfgs.DATASETS.NAME == "MSD":
                if cfgs.DATASETS.MSD_TYPE == "Spleen":
                    overall_organ_dice = np.mean(dice_list_organ)
                    overall_organ_hausdorff = np.mean(hausdorff_95_organ_list)
                    self.logger.info(
                        "Overall Organ Dice: {}, Overall organ Hausdorff_95: {}".format(overall_organ_dice, overall_organ_hausdorff))

                elif cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas":
                    overall_organ_dice = np.mean(dice_list_organ)
                    overall_organ_hausdorff = np.mean(hausdorff_95_organ_list)
                    overall_tumor_hausdorff = np.mean(hausdorff_95_tumor_list)
                    overall_tumor_dice = np.mean(dice_list_tumor)

                    overall_dice = (overall_organ_dice+overall_tumor_dice)/2.0
                    overall_hausdorff = (overall_organ_hausdorff+overall_tumor_hausdorff)/2.0
                    self.logger.info(
                        "Overall Organ Dice: {} Overall Tumor Dice: {}, Overall Dice: {}, "
                        "Overall organ Hausdorff_95: {} "
                        "Overall tumor Hausdorff_95: {}, Overall Hausdorff_95: {},".format(
                            overall_organ_dice, overall_tumor_dice, overall_dice,
                            overall_organ_hausdorff, overall_tumor_hausdorff, overall_hausdorff))
            elif cfgs.DATASETS.NAME == "BraTS_2021" or cfgs.DATASETS.NAME == "BraTS_2021_MONAI":
                dice_et_mean = np.mean(dice_list_et)
                dice_tc_mean = np.mean(dice_list_tc)
                dice_wt_mean = np.mean(dice_list_wt)

                hausdorff_et_mean = np.mean(hausdorff_list_et)
                hausdorff_tc_mean = np.mean(hausdorff_list_tc)
                hausdorff_wt_mean = np.mean(hausdorff_list_wt)

                self.logger.info(
                    "Overall ET Dice: {} Overall TC Dice: {}, Overall WT Dice: {}, "
                    "Overall ET Hausdorff_95: {} Overall TC Hausdorff_95: {} Overall WT Hausdorff_95: {}".format(
                        dice_et_mean, dice_tc_mean, dice_wt_mean, hausdorff_et_mean, hausdorff_tc_mean, hausdorff_wt_mean))

    def inference_on_original_spacing(self, cfgs):
        train_images = sorted(
            glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "imagesTr", "*.nii.gz")))
        train_labels = sorted(
            glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "labelsTr", "*.nii.gz")))
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(cfgs.INPUT.SPACING.X, cfgs.INPUT.SPACING.Y, cfgs.INPUT.SPACING.Z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MIN,
                    a_max=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MAX,
                    b_min=cfgs.INPUT.SCALE_INTENSITY.TARGET_MIN,
                    b_max=cfgs.INPUT.SCALE_INTENSITY.TARGET_MAX,
                    clip=True,
                ),
                transforms.EnsureTyped(keys=["image", "label"]),
            ]
        )

        if cfgs.DATASETS.NAME == "MSD":
            if cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas" \
                    or cfgs.DATASETS.MSD_TYPE == "Spleen":
                train_files, val_files, test_files = data_dicts[:cfgs.DATASETS.MSD.TRAIN_NUM], \
                                                     data_dicts[cfgs.DATASETS.MSD.TRAIN_NUM: cfgs.DATASETS.MSD.VAL_NUM], \
                                                     data_dicts[cfgs.DATASETS.MSD.VAL_NUM:]

        # val_transform = transforms.Compose(
        #     [
        #         transforms.LoadImaged(keys=["image", "label"]),
        #         transforms.AddChanneld(keys=["image", "label"]),
        #         transforms.Orientationd(keys=["image"],
        #                                 axcodes=cfgs.INPUT.ORIENTATION),
        #         transforms.Spacingd(keys="image",
        #                             pixdim=(cfgs.INPUT.SPACING.X, cfgs.INPUT.SPACING.Y, cfgs.INPUT.SPACING.Z),
        #                             mode="bilinear"),
        #         transforms.ScaleIntensityRanged(keys=["image"],
        #                                         a_min=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MIN,
        #                                         a_max=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MAX,
        #                                         b_min=cfgs.INPUT.SCALE_INTENSITY.TARGET_MIN,
        #                                         b_max=cfgs.INPUT.SCALE_INTENSITY.TARGET_MAX,
        #                                         clip=True),
        #         transforms.CropForegroundd(keys=["image"], source_key="image"),
        #         transforms.ToTensord(keys=["image", "label"]),
        #     ]
        # )
        test_ds = data.Dataset(data=test_files, transform=val_transforms)
        test_loader = data.DataLoader(test_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                      sampler=None,
                                      pin_memory=True,
                                      persistent_workers=True)

        post_transforms = transforms.Compose([
            transforms.EnsureTyped(keys="pred"),
            transforms.Invertd(
                keys="pred",
                transform=val_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=14),
            transforms.AsDiscreted(keys="label", to_onehot=14),
        ])

        dice_metric = DiceMetric(include_background=False, reduction="mean")

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                self.logger.info("Inference on case {}".format(img_name))
                batch["pred"] = sliding_window_inference(val_inputs,
                                                         roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                                                                   self.cfg.INPUT.RAND_CROP.ROI.Y,
                                                                   self.cfg.INPUT.RAND_CROP.ROI.Z),
                                                         sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TEST,
                                                         predictor=self.model,
                                                         overlap=self.cfg.SOLVER.INFER_OVERLAP)
                batch = [post_transforms(i) for i in decollate_batch(batch)]
                val_outputs, val_labels = from_engine(["pred", "label"])(batch)

                print(type(val_outputs), type(val_labels), val_outputs[0].size(), val_labels[0].size())

                dice_metric(y_pred=val_outputs, y=val_labels)

            metric_org = dice_metric.aggregate().item()
            dice_metric.reset()

        print("Metric on original image spacing: ", metric_org)

    def gen_inference_img(self, cfgs):
        seg_output = os.path.join(cfgs.OUTPUT_DIR, "seg_results")
        if not os.path.exists(seg_output):
            os.mkdir(seg_output)
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                val_inputs, val_labels = (batch["image"], batch["label"])
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                affine = batch['image_meta_dict']['original_affine'][0].numpy()
                self.logger.info("Inference on case {}".format(img_name))
                val_outputs = sliding_window_inference(val_inputs,
                                                       roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                                                                 self.cfg.INPUT.RAND_CROP.ROI.Y,
                                                                 self.cfg.INPUT.RAND_CROP.ROI.Z),
                                                       sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TEST,
                                                       predictor=self.model,
                                                       overlap=self.cfg.SOLVER.INFER_OVERLAP)
                prob = torch.sigmoid(val_outputs)
                seg = prob[0].detach().cpu().numpy()
                seg = (seg > 0.5).astype(np.int8)
                seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
                seg_out[seg[1] == 1] = 2
                seg_out[seg[0] == 1] = 1
                seg_out[seg[2] == 1] = 4
                nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine),
                         os.path.join(seg_output, img_name))

    def inference_brats_vt_unet(self, cfgs):
        post_trans = Compose(
            [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
        )
        # metrics_list = []
        seg_output = os.path.join(cfgs.OUTPUT_DIR, "seg_results")
        self.model.eval()
        if not os.path.exists(seg_output):
            os.mkdir(seg_output)
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                print(f"Validating case {i}")
                patient_id = batch["patient_id"][0]
                ref_path = batch["seg_path"][0]
                crops_idx = batch["crop_indexes"]

                ref_seg_img = sitk.ReadImage(ref_path)
                ref_seg = sitk.GetArrayFromImage(ref_seg_img)

                val_inputs, val_labels = (batch["image"], batch["label"])
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                self.logger.info("Inference on case {}".format(patient_id))
                val_outputs = sliding_window_inference(val_inputs,
                                                       roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                                                                 self.cfg.INPUT.RAND_CROP.ROI.Y,
                                                                 self.cfg.INPUT.RAND_CROP.ROI.Z),
                                                       sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TEST,
                                                       predictor=self.model,
                                                       overlap=self.cfg.SOLVER.INFER_OVERLAP)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                segs = torch.zeros((1, 3, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
                segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs[0]
                segs = segs[0].numpy() > 0.5

                et = segs[0]
                net = np.logical_and(segs[1], np.logical_not(et))
                ed = np.logical_and(segs[2], np.logical_not(segs[1]))
                labelmap = np.zeros(segs[0].shape)
                labelmap[et] = 4
                labelmap[net] = 1
                labelmap[ed] = 2
                labelmap = sitk.GetImageFromArray(labelmap)

                # refmap_et = ref_seg == 4
                # refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
                # refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
                # refmap = np.stack([refmap_et, refmap_tc, refmap_wt])

                # patient_metric_list = calculate_metrics(segs, refmap, patient_id)
                # metrics_list.append(patient_metric_list)
                labelmap.CopyInformation(ref_seg_img)

                print(f"Writing {seg_output}/{patient_id}.nii.gz")
                sitk.WriteImage(labelmap, f"{seg_output}/{patient_id}.nii.gz")

    def visualzation(self, cfgs):
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if cfgs.DATASETS.NAME == "BraTS_2021":
                    t1, t1ce, t2, flair = batch["t1"], batch["t1ce"], batch["t2"], batch["flair"]
                    val_inputs = torch.cat([t1, t1ce, t2, flair], dim=1)
                    val_labels = batch["seg"]
                else:
                    val_inputs, val_labels = (batch["image"], batch["label"])
                val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()
                if cfgs.DATASETS.NAME == "BraTS_2021":
                    img_name = batch['t1_meta_dict']['filename_or_obj'][0].split('/')[-1]
                else:
                    img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                self.logger.info("Inference on case {}".format(img_name))
                # val_outputs = sliding_window_inference(val_inputs,
                #                                        roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                #                                                  self.cfg.INPUT.RAND_CROP.ROI.Y,
                #                                                  self.cfg.INPUT.RAND_CROP.ROI.Z),
                #                                        sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TEST,
                #                                        predictor=self.model,
                #                                        overlap=self.cfg.SOLVER.INFER_OVERLAP)
                # prob = torch.sigmoid(val_outputs)
                # seg = prob[0].detach().cpu().numpy()
                # seg = (seg > 0.5).astype(np.int8)
                # seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
                # seg_out[seg[1] == 1] = 2
                # seg_out[seg[0] == 1] = 1
                # seg_out[seg[2] == 1] = 4

                val_data_example = val_inputs[0]
                print(f"image shape: {val_data_example.shape}")
                plt.figure("image", (24, 6))
                for i in range(4):
                    plt.subplot(1, 4, i + 1)
                    plt.title(f"image channel {i}")
                    plt.imshow(val_data_example[i, :, :, 97].detach().cpu(), cmap="gray")
                plt.show()
                # also visualize the 3 channels label corresponding to this image
                print(f"label shape: {val_labels[0].shape}")
                plt.figure("label", (18, 6))
                for i in range(3):
                    plt.subplot(1, 3, i + 1)
                    plt.title(f"label channel {i}")
                    plt.imshow(val_labels[0][i, :, :, 97].detach().cpu())
                plt.show()

    def visualize_2d(self):
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                self.logger.info("Inference on case {}".format(img_name))
                val_outputs = sliding_window_inference(val_inputs,
                                                       roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                                                                 self.cfg.INPUT.RAND_CROP.ROI.Y,
                                                                 self.cfg.INPUT.RAND_CROP.ROI.Z),
                                                       sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TEST,
                                                       predictor=self.model,
                                                       overlap=self.cfg.SOLVER.INFER_OVERLAP)

                plt.figure("check", (18, 6))
                plt.subplot(1, 3, 1)
                plt.title(f"image {i}")
                plt.imshow(val_inputs.cpu()[0, 0, :, 150, :], cmap="gray")
                plt.subplot(1, 3, 2)
                plt.title(f"label {i}")
                plt.imshow(val_labels.cpu()[0, 0, :, 150, :])
                plt.subplot(1, 3, 3)
                plt.title(f"output {i}")
                plt.imshow(torch.argmax(
                    val_outputs, dim=1).detach().cpu()[0, :, 150, :])
                plt.show()
                if i == 2:
                    break

    def visualize_3d(self, cfgs):
        datalist_json = os.path.join(cfgs.DATASETS.DATA_DIR, cfgs.DATASETS.JSON)

        test_org_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys="image"),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.Orientationd(keys=["image"],
                                        axcodes=cfgs.INPUT.ORIENTATION),
                transforms.Spacingd(keys=["image"],
                                    pixdim=(cfgs.INPUT.SPACING.X, cfgs.INPUT.SPACING.Y, cfgs.INPUT.SPACING.Z),
                                    mode="bilinear"),
                transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MIN,
                                                a_max=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MAX,
                                                b_min=cfgs.INPUT.SCALE_INTENSITY.TARGET_MIN,
                                                b_max=cfgs.INPUT.SCALE_INTENSITY.TARGET_MAX,
                                                clip=True),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
                transforms.EnsureTyped(keys="image"),
            ]
        )
        test_files = load_decathlon_datalist(datalist_json,
                                             True,
                                             cfgs.DATASETS.TEST_TYPE,
                                             base_dir=cfgs.DATASETS.DATA_DIR)
        test_ds = data.Dataset(data=test_files, transform=test_org_transforms)
        test_loader = data.DataLoader(test_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                      sampler=None,
                                      pin_memory=True,
                                      persistent_workers=True)

        post_transforms = transforms.Compose([
            transforms.EnsureTyped(keys="pred"),
            transforms.Invertd(
                keys="pred",
                transform=test_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=5),
            transforms.SplitChanneld(keys="pred", channel_dim=0),
            transforms.SaveImaged(keys="pred_0",
                                  meta_keys="pred_meta_dict",
                                  output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
                                  output_postfix="seg_0",
                                  resample=False),
            transforms.SaveImaged(keys="pred_1",
                                  meta_keys="pred_meta_dict",
                                  output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
                                  output_postfix="seg_1",
                                  resample=False),
            transforms.SaveImaged(keys="pred_2",
                                  meta_keys="pred_meta_dict",
                                  output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
                                  output_postfix="seg_2",
                                  resample=False),
            transforms.SaveImaged(keys="pred_3",
                                  meta_keys="pred_meta_dict",
                                  output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
                                  output_postfix="seg_3",
                                  resample=False),
            transforms.SaveImaged(keys="pred_4",
                                  meta_keys="pred_meta_dict",
                                  output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
                                  output_postfix="seg_4",
                                  resample=False)
            # transforms.SaveImaged(keys="pred_5",
            #                       meta_keys="pred_meta_dict",
            #                       output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
            #                       output_postfix="seg_5",
            #                       resample=False),
            #
            # transforms.SaveImaged(keys="pred_6",
            #                       meta_keys="pred_meta_dict",
            #                       output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
            #                       output_postfix="seg_6",
            #                       resample=False),
            # transforms.SaveImaged(keys="pred_7",
            #                       meta_keys="pred_meta_dict",
            #                       output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
            #                       output_postfix="seg_7",
            #                       resample=False),
            # transforms.SaveImaged(keys="pred_8",
            #                       meta_keys="pred_meta_dict",
            #                       output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
            #                       output_postfix="seg_8",
            #                       resample=False),
            # transforms.SaveImaged(keys="pred_9",
            #                       meta_keys="pred_meta_dict",
            #                       output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
            #                       output_postfix="seg_9",
            #                       resample=False),
            # transforms.SaveImaged(keys="pred_10",
            #                       meta_keys="pred_meta_dict",
            #                       output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
            #                       output_postfix="seg_10",
            #                       resample=False),
            #
            # transforms.SaveImaged(keys="pred_11",
            #                       meta_keys="pred_meta_dict",
            #                       output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
            #                       output_postfix="seg_11",
            #                       resample=False),
            # transforms.SaveImaged(keys="pred_12",
            #                       meta_keys="pred_meta_dict",
            #                       output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
            #                       output_postfix="seg_12",
            #                       resample=False),
            # transforms.SaveImaged(keys="pred_13",
            #                       meta_keys="pred_meta_dict",
            #                       output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
            #                       output_postfix="seg_13",
            #                       resample=False)
        ])

        loader = LoadImage()

        with torch.no_grad():
            for test_data in test_loader:
                test_data["image"] = test_data["image"].cuda()
                img_name = test_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                self.logger.info("Inference on case {}".format(img_name))
                test_data["pred"] = sliding_window_inference(test_data["image"],
                                                             roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                                                                       self.cfg.INPUT.RAND_CROP.ROI.Y,
                                                                       self.cfg.INPUT.RAND_CROP.ROI.Z),
                                                             sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TEST,
                                                             predictor=self.model,
                                                             overlap=self.cfg.SOLVER.INFER_OVERLAP)
                test_data = [post_transforms(i) for i in decollate_batch(test_data)]

                test_data[0]["pred_meta_dict"]["affine"] = test_data[0]["image_meta_dict"]["affine"]
                test_output = from_engine(["pred"])(test_data)[0]
                original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])[0]
                print(test_output.shape, original_image.shape)
                plt.figure("check", (18, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(original_image[:, :, int(test_output.shape[-1]/2)], cmap="gray")
                plt.subplot(1, 2, 2)
                plt.imshow(test_output.detach().cpu()[6, :, :, int(test_output.shape[-1]/2)])
                plt.show()

    def visualization_3d_msd_liver(self, cfgs):
        train_images = sorted(
            glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "imagesTr", "*.nii.gz")))
        train_labels = sorted(
            glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "labelsTr", "*.nii.gz")))
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(cfgs.INPUT.SPACING.X, cfgs.INPUT.SPACING.Y, cfgs.INPUT.SPACING.Z),
                                    mode=("bilinear", "nearest")),
                transforms.ScaleIntensityRanged(
                    keys=["image"],
                    a_min=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MIN,
                    a_max=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MAX,
                    b_min=cfgs.INPUT.SCALE_INTENSITY.TARGET_MIN,
                    b_max=cfgs.INPUT.SCALE_INTENSITY.TARGET_MAX,
                    clip=True,
                ),
                transforms.EnsureTyped(keys=["image", "label"]),
            ]
        )

        if cfgs.DATASETS.NAME == "MSD":
            if cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas" \
                    or cfgs.DATASETS.MSD_TYPE == "Spleen":
                train_files, val_files, test_files = data_dicts[:cfgs.DATASETS.MSD.TRAIN_NUM], \
                                                     data_dicts[cfgs.DATASETS.MSD.TRAIN_NUM: cfgs.DATASETS.MSD.VAL_NUM], \
                                                     data_dicts[cfgs.DATASETS.MSD.VAL_NUM:]
        test_ds = data.Dataset(data=test_files, transform=val_transforms)
        test_loader = data.DataLoader(test_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                      sampler=None,
                                      pin_memory=True,
                                      persistent_workers=True)
        post_transforms = transforms.Compose([
            transforms.EnsureTyped(keys="pred"),
            transforms.Invertd(
                keys="pred",
                transform=val_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=5),
            transforms.SplitChanneld(keys="pred", channel_dim=0),
            transforms.SaveImaged(keys="pred_0",
                                  meta_keys="pred_meta_dict",
                                  output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
                                  output_postfix="seg_0",
                                  resample=False),
            transforms.SaveImaged(keys="pred_1",
                                  meta_keys="pred_meta_dict",
                                  output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
                                  output_postfix="seg_1",
                                  resample=False),
            transforms.SaveImaged(keys="pred_2",
                                  meta_keys="pred_meta_dict",
                                  output_dir=os.path.join(cfgs.OUTPUT_DIR, "inference_all_part1"),
                                  output_postfix="seg_2",
                                  resample=False)
        ])
        loader = LoadImage()

        with torch.no_grad():
            for test_data in test_loader:
                test_data["image"] = test_data["image"].cuda()
                img_name = test_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                print("Inference on case {}".format(img_name))
                test_data["pred"] = sliding_window_inference(test_data["image"],
                                                             roi_size=(cfgs.INPUT.RAND_CROP.ROI.X,
                                                                       cfgs.INPUT.RAND_CROP.ROI.Y,
                                                                       cfgs.INPUT.RAND_CROP.ROI.Z),
                                                             sw_batch_size=cfgs.SOLVER.SW_BATCH_SIZE_TEST,
                                                             predictor=self.model,
                                                             overlap=self.cfg.SOLVER.INFER_OVERLAP)
                test_data = [post_transforms(i) for i in decollate_batch(test_data)]


                test_data[0]["pred_meta_dict"]["affine"] = test_data[0]["image_meta_dict"]["affine"]
                test_output = from_engine(["pred"])(test_data)[0]
                original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])[0]
                print(test_output.shape, original_image.shape)
                plt.figure("check", (18, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(original_image[:, :, int(test_output.shape[-1]/2)], cmap="gray")
                plt.subplot(1, 2, 2)
                plt.imshow(test_output.detach().cpu()[1, :, :, int(test_output.shape[-1]/2)])
                plt.show()

    def inference_on_original_spacing_msd(self, cfgs):
        train_images = sorted(
            glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "imagesTr", "*.nii.gz")))
        train_labels = sorted(
            glob.glob(os.path.join(cfgs.DATASETS.DATA_DIR, "labelsTr", "*.nii.gz")))
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys="image"),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.Orientationd(keys="image", axcodes=cfgs.INPUT.ORIENTATION),
                transforms.Spacingd(keys="image",
                                    pixdim=(cfgs.INPUT.SPACING.X, cfgs.INPUT.SPACING.Y, cfgs.INPUT.SPACING.Z),
                                    mode="bilinear"),
                transforms.ScaleIntensityRanged(
                    keys="image",
                    a_min=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MIN,
                    a_max=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MAX,
                    b_min=cfgs.INPUT.SCALE_INTENSITY.TARGET_MIN,
                    b_max=cfgs.INPUT.SCALE_INTENSITY.TARGET_MAX,
                    clip=True,
                ),
                transforms.CropForegroundd(keys="image", source_key="image"),
                transforms.EnsureType(),
            ]
        )

        output_dir = os.path.join(cfgs.OUTPUT_DIR, "seg_results")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if cfgs.DATASETS.NAME == "MSD":
            if cfgs.DATASETS.MSD_TYPE == "Liver" or cfgs.DATASETS.MSD_TYPE == "Pancreas" \
                    or cfgs.DATASETS.MSD_TYPE == "Spleen":
                train_files, val_files, test_files = data_dicts[:cfgs.DATASETS.MSD.TRAIN_NUM], \
                                                     data_dicts[cfgs.DATASETS.MSD.TRAIN_NUM: cfgs.DATASETS.MSD.VAL_NUM], \
                                                     data_dicts[cfgs.DATASETS.MSD.VAL_NUM:]
        test_ds = data.Dataset(data=test_files, transform=val_transforms)
        test_loader = data.DataLoader(test_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                      sampler=None,
                                      pin_memory=True,
                                      persistent_workers=True)

        post_transforms = transforms.Compose([
            transforms.EnsureTyped(keys="pred"),
            transforms.Invertd(
                keys="pred",
                transform=val_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            transforms.Activationsd(keys="pred", softmax=True),
            transforms.AsDiscreted(keys="pred", argmax=True),
            # transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=cfgs.MODEL.OUT_CHANNEL),
            transforms.SaveImaged(keys="pred", output_dir=output_dir, output_postfix="seg", resample=False)
            # transforms.AsDiscreted(keys="label", to_onehot=cfgs.MODEL.OUT_CHANNEL),
        ])

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                val_inputs = batch["image"].cuda()
                img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                if img_name == "liver_87.nii.gz" or img_name == "liver_95.nii.gz":
                    continue
                self.logger.info("Inference on case {}".format(img_name))
                batch["pred"] = sliding_window_inference(val_inputs,
                                                         roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                                                                   self.cfg.INPUT.RAND_CROP.ROI.Y,
                                                                   self.cfg.INPUT.RAND_CROP.ROI.Z),
                                                         sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TEST,
                                                         predictor=self.model,
                                                         overlap=self.cfg.SOLVER.INFER_OVERLAP)
                batch = [post_transforms(i) for i in decollate_batch(batch)]

    def inference_on_original_spacing_abdomen(self, cfgs):
        data_dir = cfgs.DATASETS.DATA_DIR
        datalist_json = os.path.join(data_dir, cfgs.DATASETS.JSON)

        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys="image"),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.Orientationd(keys="image",
                                        axcodes=cfgs.INPUT.ORIENTATION),
                transforms.Spacingd(keys="image",
                                    pixdim=(cfgs.INPUT.SPACING.X, cfgs.INPUT.SPACING.Y, cfgs.INPUT.SPACING.Z),
                                    mode="bilinear"),
                transforms.ScaleIntensityRanged(
                    keys="image",
                    a_min=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MIN,
                    a_max=cfgs.INPUT.SCALE_INTENSITY.ORIGINAL_MAX,
                    b_min=cfgs.INPUT.SCALE_INTENSITY.TARGET_MIN,
                    b_max=cfgs.INPUT.SCALE_INTENSITY.TARGET_MAX,
                    clip=True,
                ),
                # transforms.CropForegroundd(keys="image", source_key="image"),
            ]
        )

        test_files = load_decathlon_datalist(datalist_json,
                                             True,
                                             cfgs.DATASETS.TEST_TYPE,
                                             base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_loader = data.DataLoader(test_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=cfgs.DATALOADER.TEST_WORKERS,
                                      sampler=None,
                                      pin_memory=True,
                                      persistent_workers=True)

        output_dir = os.path.join(cfgs.OUTPUT_DIR, "seg_results")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        post_transforms = transforms.Compose([
            transforms.EnsureTyped(keys="pred"),
            transforms.Invertd(
                keys="pred",
                transform=val_transform,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            transforms.Activationsd(keys="pred", softmax=True),
            transforms.AsDiscreted(keys="pred", argmax=True),
            # transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=cfgs.MODEL.OUT_CHANNEL),
            transforms.SaveImaged(keys="pred", output_dir=output_dir, output_postfix="seg", resample=False)
            # transforms.AsDiscreted(keys="label", to_onehot=cfgs.MODEL.OUT_CHANNEL),
        ])

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                val_inputs = batch["image"].cuda()
                img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                self.logger.info("Inference on case {}".format(img_name))
                batch["pred"] = sliding_window_inference(val_inputs,
                                                         roi_size=(self.cfg.INPUT.RAND_CROP.ROI.X,
                                                                   self.cfg.INPUT.RAND_CROP.ROI.Y,
                                                                   self.cfg.INPUT.RAND_CROP.ROI.Z),
                                                         sw_batch_size=self.cfg.SOLVER.SW_BATCH_SIZE_TEST,
                                                         predictor=self.model,
                                                         overlap=self.cfg.SOLVER.INFER_OVERLAP)
                batch = [post_transforms(i) for i in decollate_batch(batch)]


def calculate_metrics(preds, targets, patient, tta=False):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            dice = 1 if np.sum(preds[i]) == 0 else 0

        else:
            tp = np.sum(l_and(preds[i], targets[i]))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[DICE] = dice
        pp.pprint(metrics)
        metrics_list.append(metrics)

    return metrics_list
