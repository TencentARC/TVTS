import numpy as np
import torch
from torch import nn
from torchvision.utils import make_grid
from base import Multi_BaseTrainer_dist, BaseTrainer
from utils import inf_loop
from model.model_dist_TVTSv2_ViT_B_16 import sim_matrix
from itertools import cycle
import sys
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat
import itertools
from torch import nn, einsum
from functools import reduce
import os
import json
from itertools import cycle
from torch.cuda.amp import autocast, GradScaler


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(n_gpu)]
        dist.all_gather(output, tensor)
        ctx.local_rank = args.local_rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.local_rank: ctx.batch_size * (ctx.local_rank + 1)],
            None, None,
        )


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )


class Trainer_TVTSv2_B_32(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            target_dset = 'YT'
            for x in data_loader:
                if x.dataset_name.startswith(target_dset):
                    self.len_epoch = len(x)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.loss_func = nn.CrossEntropyLoss()
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        self.num_clips = 4
        self.n_trans = 4

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr_rate, global_lr_rate = 1., 1.
        for milestone in args.schedule:
            if epoch == milestone:
                lr_rate = 0.1
            if epoch >= milestone:
                global_lr_rate *= 0.1

        for group in optimizer.param_groups:
            group['lr'] = group['lr'] * lr_rate

        if self.args.rank == 0:
            print('adjusting learning rate..')
            print('base lr: {}'.format(self.base_lr))
            print('lr_rate: {}'.format(lr_rate))
            print('global_lr_rate: {}'.format(global_lr_rate))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)

        # choose target dataloader to enumerate
        # for other dataloaders we use iterator instead of cycle(dl) to avoid memory leak
        iter_dl = [None] * len(self.data_loader)
        for dl_idx, dl in enumerate(self.data_loader):
            if len(dl) == self.len_epoch:
                loop_dl = dl
                loop_dl_idx = dl_idx
            else:
                iter_dl[dl_idx] = iter(dl)

        for batch_idx, loop_dl_data in enumerate(loop_dl):
            # collect data from all dataloaders
            data_li = [None] * len(self.data_loader)
            for dl_idx in range(len(iter_dl)):
                if dl_idx != loop_dl_idx:
                    try:
                        data_li[dl_idx] = next(iter_dl[dl_idx])
                    except StopIteration:
                        iter_dl[dl_idx] = iter(self.data_loader[dl_idx])
                        data_li[dl_idx] = next(iter_dl[dl_idx])
            data_li[loop_dl_idx] = loop_dl_data

            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    num_clips = len(data['text'])
                    for i in range(num_clips):
                        if i == 0:
                            text_all = data['text'][0]
                        else:
                            text_all = text_all + data['text'][i]
                    data['text'] = text_all
                    data['text'] = self.tokenizer(data['text'], truncate=True)
                data['text'] = data['text'].to(self.device)
                data['video'] = data['video'].to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    text_embeds, video_embeds, pred_order = self.model(data)

                    video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
                    text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)

                    output = sim_matrix(video_embeds, text_embeds)
                    loss1 = self.loss(output)

                    if pred_order is not None:
                        # B*n_trans
                        labels = data['label'].to(self.device).reshape(-1)
                        # B*n_trans x n_trans
                        pred_order = pred_order.reshape(-1, pred_order.shape[-1])
                        loss2 = self.loss_func(pred_order, labels) * 2
                    else:
                        loss2 = torch.Tensor([0]).to(self.device)

                loss = loss1 + loss2
                loss.backward()

                self.optimizer.step()
                if self.writer is not None and self.args.rank == 0:
                    self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                    self.logger.debug('Train Epoch: {} dl{} {} Loss_ct: {:.6f} Loss_ce: {:.6f} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss1.detach().item(),
                        loss2.detach().item(),
                        loss.detach().item()))

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_num = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(dl):
                    # meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        num_clips = len(data['text'])
                        for i in range(num_clips):
                            if i == 0:
                                text_all = data['text'][0]
                            else:
                                text_all = text_all + data['text'][i]
                        data['text'] = text_all
                        data['text'] = self.tokenizer(data['text'], truncate=True)
                    data['text'] = data['text'].to(self.device)
                    data['video'] = data['video'].to(self.device)

                    text_embed, vid_embed, preds = self.model(data, return_embeds=True)

                    vid_embed_all = [torch.zeros_like(vid_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(vid_embed_all, vid_embed)
                    vid_embed_all = torch.cat(vid_embed_all, dim=0)

                    text_embed_all = [torch.zeros_like(text_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(text_embed_all, text_embed)
                    text_embed_all = torch.cat(text_embed_all, dim=0)

                    if preds is not None:
                        # B x n_trans
                        labels = data['label'].to(self.device)
                        labels_all = [torch.zeros_like(labels) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(labels_all, labels)
                        labels_all = torch.cat(labels_all, dim=0).cpu().numpy()

                        # B x n_trans x n_trans
                        preds = torch.argmax(preds, axis=-1)
                        preds_all = [torch.zeros_like(preds) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(preds_all, preds)
                        preds_all = torch.cat(preds_all, dim=0).cpu().numpy()

                        acc = len(np.where(np.all(preds_all == labels_all, axis=1) == 1)[0])

                        total_val_loss[dl_idx] += acc
                        total_val_num[dl_idx] += preds_all.shape[0]
                    else:
                        total_val_loss[dl_idx] = 1
                        total_val_num[dl_idx] = 1

                    text_embed_arr[dl_idx].append(text_embed_all.cpu())
                    vid_embed_arr[dl_idx].append(vid_embed_all.cpu())

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None and self.args.rank == 0:
                self.writer.log_scalar(f'loss_val_{dl_idx}',
                                       total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            text_embeds = torch.cat(text_embed_arr[dl_idx])
            vid_embeds = torch.cat(vid_embed_arr[dl_idx])
            sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims)
                if self.args.rank == 0:
                    verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                            mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        self.writer.log_scalar(key, val)

                if self.visualizer is not None and self.args.rank == 0:
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta in meta_arr:
                        for key, val in meta.items():
                            meta_arr_cat[key] += val
                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / total_val_num[dl_idx]
                        for dl_idx in range(len(self.valid_data_loader))}
        if self.args.rank == 0:
            print("Top-1 Accuracy for Frame Prediction:", total_val_loss[0] / total_val_num[0])

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class Trainer_TVTSv2_B_16(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            target_dset = 'YT'
            for x in data_loader:
                if x.dataset_name.startswith(target_dset):
                    self.len_epoch = len(x)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.loss_func = nn.CrossEntropyLoss()
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        self.num_clips = 4
        self.n_trans = 4

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr_rate, global_lr_rate = 1., 1.
        for milestone in args.schedule:
            if epoch == milestone:
                lr_rate = 0.1
            if epoch >= milestone:
                global_lr_rate *= 0.1

        for group in optimizer.param_groups:
            group['lr'] = group['lr'] * lr_rate

        if self.args.rank == 0:
            print('adjusting learning rate..')
            print('base lr: {}'.format(self.base_lr))
            print('lr_rate: {}'.format(lr_rate))
            print('global_lr_rate: {}'.format(global_lr_rate))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)

        # choose target dataloader to enumerate
        # for other dataloaders we use iterator instead of cycle(dl) to avoid memory leak
        iter_dl = [None] * len(self.data_loader)
        for dl_idx, dl in enumerate(self.data_loader):
            if len(dl) == self.len_epoch:
                loop_dl = dl
                loop_dl_idx = dl_idx
            else:
                iter_dl[dl_idx] = iter(dl)

        for batch_idx, loop_dl_data in enumerate(loop_dl):
            # collect data from all dataloaders
            data_li = [None] * len(self.data_loader)
            for dl_idx in range(len(iter_dl)):
                if dl_idx != loop_dl_idx:
                    try:
                        data_li[dl_idx] = next(iter_dl[dl_idx])
                    except StopIteration:
                        iter_dl[dl_idx] = iter(self.data_loader[dl_idx])
                        data_li[dl_idx] = next(iter_dl[dl_idx])
            data_li[loop_dl_idx] = loop_dl_data

            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    num_clips = len(data['text'])
                    for i in range(num_clips):
                        if i == 0:
                            text_all = data['text'][0]
                        else:
                            text_all = text_all + data['text'][i]
                    data['text'] = text_all
                    data['text'] = self.tokenizer(data['text'], truncate=True)
                data['text'] = data['text'].to(self.device)
                data['video'] = data['video'].to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    text_embeds, video_embeds, pred_order = self.model(data)

                    video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
                    text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)

                    output = sim_matrix(video_embeds, text_embeds)
                    loss1 = self.loss(output)

                    if pred_order is not None:
                        # B*n_trans
                        labels = data['label'].to(self.device).reshape(-1)
                        # B*n_trans x n_trans
                        pred_order = pred_order.reshape(-1, pred_order.shape[-1])
                        loss2 = self.loss_func(pred_order, labels) * 2
                    else:
                        loss2 = torch.Tensor([0]).to(self.device)

                loss = loss1 + loss2
                loss.backward()

                self.optimizer.step()
                if self.writer is not None and self.args.rank == 0:
                    self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                    self.logger.debug('Train Epoch: {} dl{} {} Loss_ct: {:.6f} Loss_ce: {:.6f} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss1.detach().item(),
                        loss2.detach().item(),
                        loss.detach().item()))

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_num = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(dl):
                    # meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        num_clips = len(data['text'])
                        for i in range(num_clips):
                            if i == 0:
                                text_all = data['text'][0]
                            else:
                                text_all = text_all + data['text'][i]
                        data['text'] = text_all
                        data['text'] = self.tokenizer(data['text'], truncate=True)
                    data['text'] = data['text'].to(self.device)
                    data['video'] = data['video'].to(self.device)

                    text_embed, vid_embed, preds = self.model(data, return_embeds=True)

                    vid_embed_all = [torch.zeros_like(vid_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(vid_embed_all, vid_embed)
                    vid_embed_all = torch.cat(vid_embed_all, dim=0)

                    text_embed_all = [torch.zeros_like(text_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(text_embed_all, text_embed)
                    text_embed_all = torch.cat(text_embed_all, dim=0)

                    if preds is not None:
                        # B x n_trans
                        labels = data['label'].to(self.device)
                        labels_all = [torch.zeros_like(labels) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(labels_all, labels)
                        labels_all = torch.cat(labels_all, dim=0).cpu().numpy()

                        # B x n_trans x n_trans
                        preds = torch.argmax(preds, axis=-1)
                        preds_all = [torch.zeros_like(preds) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(preds_all, preds)
                        preds_all = torch.cat(preds_all, dim=0).cpu().numpy()

                        acc = len(np.where(np.all(preds_all == labels_all, axis=1) == 1)[0])

                        total_val_loss[dl_idx] += acc
                        total_val_num[dl_idx] += preds_all.shape[0]
                    else:
                        total_val_loss[dl_idx] = 1
                        total_val_num[dl_idx] = 1

                    text_embed_arr[dl_idx].append(text_embed_all.cpu())
                    vid_embed_arr[dl_idx].append(vid_embed_all.cpu())

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None and self.args.rank == 0:
                self.writer.log_scalar(f'loss_val_{dl_idx}',
                                       total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            text_embeds = torch.cat(text_embed_arr[dl_idx])
            vid_embeds = torch.cat(vid_embed_arr[dl_idx])
            sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims)
                if self.args.rank == 0:
                    verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                            mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        self.writer.log_scalar(key, val)

                if self.visualizer is not None and self.args.rank == 0:
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta in meta_arr:
                        for key, val in meta.items():
                            meta_arr_cat[key] += val
                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / total_val_num[dl_idx]
                        for dl_idx in range(len(self.valid_data_loader))}
        if self.args.rank == 0:
            print("Top-1 Accuracy for Frame Prediction:", total_val_loss[0] / total_val_num[0])

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class Trainer_TVTSv2_H_14(Multi_BaseTrainer_dist):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, args, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, writer=None,
                 visualizer=None, tokenizer=None, max_samples_per_epoch=50000):
        super().__init__(args, model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.args = args
        self.data_loader = data_loader
        if len_epoch is None:
            target_dset = 'YT'
            for x in data_loader:
                if x.dataset_name.startswith(target_dset):
                    self.len_epoch = len(x)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.visualizer = visualizer
        self.val_chunking = True
        self.batch_size = self.data_loader[0].batch_size
        self.log_step = int(np.sqrt(self.batch_size))
        self.total_batch_sum = sum([x.batch_size for x in self.data_loader])
        self.tokenizer = tokenizer
        self.max_samples_per_epoch = max_samples_per_epoch
        self.loss_func = nn.CrossEntropyLoss()
        self.n_gpu = self.args.world_size
        self.allgather = AllGather_multi.apply
        self.num_clips = 4
        self.n_trans = 4

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _adjust_learning_rate(self, optimizer, epoch, args):
        lr_rate, global_lr_rate = 1., 1.
        for milestone in args.schedule:
            if epoch == milestone:
                lr_rate = 0.1
            if epoch >= milestone:
                global_lr_rate *= 0.1

        for group in optimizer.param_groups:
            group['lr'] = group['lr'] * lr_rate

        if self.args.rank == 0:
            print('adjusting learning rate..')
            print('base lr: {}'.format(self.base_lr))
            print('lr_rate: {}'.format(lr_rate))
            print('global_lr_rate: {}'.format(global_lr_rate))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        total_metrics = np.zeros(len(self.metrics))
        for loader in self.data_loader:
            loader.train_sampler.set_epoch(epoch)

        # choose target dataloader to enumerate
        # for other dataloaders we use iterator instead of cycle(dl) to avoid memory leak
        iter_dl = [None] * len(self.data_loader)
        for dl_idx, dl in enumerate(self.data_loader):
            if len(dl) == self.len_epoch:
                loop_dl = dl
                loop_dl_idx = dl_idx
            else:
                iter_dl[dl_idx] = iter(dl)

        for batch_idx, loop_dl_data in enumerate(loop_dl):
            # collect data from all dataloaders
            data_li = [None] * len(self.data_loader)
            for dl_idx in range(len(iter_dl)):
                if dl_idx != loop_dl_idx:
                    try:
                        data_li[dl_idx] = next(iter_dl[dl_idx])
                    except StopIteration:
                        iter_dl[dl_idx] = iter(self.data_loader[dl_idx])
                        data_li[dl_idx] = next(iter_dl[dl_idx])
            data_li[loop_dl_idx] = loop_dl_data

            for dl_idx, data in enumerate(data_li):
                # then assume we must tokenize the input, e.g. its a string
                if self.tokenizer is not None:
                    num_clips = len(data['text'])
                    for i in range(num_clips):
                        if i == 0:
                            text_all = data['text'][0]
                        else:
                            text_all = text_all + data['text'][i]
                    data['text'] = text_all
                    data['text'] = self.tokenizer(data['text'])
                data['text'] = data['text'].to(self.device)
                data['video'] = data['video'].to(self.device)

                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    text_embeds, video_embeds, pred_order = self.model(data)

                    video_embeds = self.allgather(video_embeds, self.n_gpu, self.args)
                    text_embeds = self.allgather(text_embeds, self.n_gpu, self.args)

                    output = sim_matrix(video_embeds, text_embeds)
                    loss1 = self.loss(output)

                    if pred_order is not None:
                        # B*n_trans
                        labels = data['label'].to(self.device).reshape(-1)
                        # B*n_trans x n_trans
                        pred_order = pred_order.reshape(-1, pred_order.shape[-1])
                        loss2 = self.loss_func(pred_order, labels) * 2
                    else:
                        loss2 = torch.Tensor([0]).to(self.device)

                loss = loss1 + loss2
                loss.backward()

                self.optimizer.step()
                if self.writer is not None and self.args.rank == 0:
                    self.writer.log_scalar(f'loss_train_{dl_idx}', loss.detach().item())

                total_loss[dl_idx] += loss.detach().item()

                if batch_idx % self.log_step == 0 and self.args.local_rank == 0:
                    self.logger.debug('Train Epoch: {} dl{} {} Loss_ct: {:.6f} Loss_ce: {:.6f} Loss: {:.6f}'.format(
                        epoch,
                        dl_idx,
                        self._progress(batch_idx, dl_idx),
                        loss1.detach().item(),
                        loss2.detach().item(),
                        loss.detach().item()))

        log = {
            f'loss_{dl_idx}': total_loss[dl_idx] / self.len_epoch for dl_idx in range(len(self.data_loader))
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.args.rank == 0:
                log.update(val_log)

        self._adjust_learning_rate(self.optimizer, epoch, self.args)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        total_val_num = [0] * len(self.valid_data_loader)
        total_val_metrics = [np.zeros(len(self.metrics))] * len(self.valid_data_loader)
        meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        text_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}
        vid_embed_arr = {x: [] for x in range(len(self.valid_data_loader))}

        with torch.no_grad():
            # for validation we switch the nested loop order, because alternate batches not needed...
            # ... and dataloaders can be of different length
            for dl_idx, dl in enumerate(self.valid_data_loader):
                for batch_idx, data in enumerate(dl):
                    # meta_arr[dl_idx].append(data['meta'])
                    if self.tokenizer is not None:
                        num_clips = len(data['text'])
                        for i in range(num_clips):
                            if i == 0:
                                text_all = data['text'][0]
                            else:
                                text_all = text_all + data['text'][i]
                        data['text'] = text_all
                        data['text'] = self.tokenizer(data['text'])
                    data['text'] = data['text'].to(self.device)
                    data['video'] = data['video'].to(self.device)

                    text_embed, vid_embed, preds = self.model(data, return_embeds=True)

                    vid_embed_all = [torch.zeros_like(vid_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(vid_embed_all, vid_embed)
                    vid_embed_all = torch.cat(vid_embed_all, dim=0)

                    text_embed_all = [torch.zeros_like(text_embed) for _ in range(self.n_gpu)]
                    torch.distributed.all_gather(text_embed_all, text_embed)
                    text_embed_all = torch.cat(text_embed_all, dim=0)

                    if preds is not None:
                        # B x n_trans
                        labels = data['label'].to(self.device)
                        labels_all = [torch.zeros_like(labels) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(labels_all, labels)
                        labels_all = torch.cat(labels_all, dim=0).cpu().numpy()

                        # B x n_trans x n_trans
                        preds = torch.argmax(preds, axis=-1)
                        preds_all = [torch.zeros_like(preds) for _ in range(self.n_gpu)]
                        torch.distributed.all_gather(preds_all, preds)
                        preds_all = torch.cat(preds_all, dim=0).cpu().numpy()

                        acc = len(np.where(np.all(preds_all == labels_all, axis=1) == 1)[0])

                        total_val_loss[dl_idx] += acc
                        total_val_num[dl_idx] += preds_all.shape[0]
                    else:
                        total_val_loss[dl_idx] = 1
                        total_val_num[dl_idx] = 1

                    text_embed_arr[dl_idx].append(text_embed_all.cpu())
                    vid_embed_arr[dl_idx].append(vid_embed_all.cpu())

        for dl_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None and self.args.rank == 0:
                self.writer.log_scalar(f'loss_val_{dl_idx}',
                                       total_val_loss[dl_idx] / len(self.valid_data_loader[dl_idx]))
            nested_metrics = {x: {} for x in range(len(self.valid_data_loader))}

            text_embeds = torch.cat(text_embed_arr[dl_idx])
            vid_embeds = torch.cat(vid_embed_arr[dl_idx])
            sims = sim_matrix(text_embeds, vid_embeds).detach().cpu().numpy()

            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims)
                if self.args.rank == 0:
                    verbose(epoch=epoch, metrics=res, name=self.valid_data_loader[dl_idx].dataset_name,
                            mode=metric_name)
                nested_metrics[dl_idx][metric_name] = res

                if self.writer is not None and self.args.rank == 0:
                    to_write = format_nested_metrics_for_writer(res, mode=metric_name,
                                                                name=self.valid_data_loader[dl_idx].dataset_name)
                    for key, val in to_write.items():
                        self.writer.log_scalar(key, val)

                if self.visualizer is not None and self.args.rank == 0:
                    meta_arr_cat = {key: [] for key in meta_arr[0]}
                    for meta in meta_arr:
                        for key, val in meta.items():
                            meta_arr_cat[key] += val
                    self.visualizer.visualize_ranking(sims, epoch, meta_arr_cat, nested_metrics)

        res_dict = {}
        if self.args.rank == 0:
            res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / total_val_num[dl_idx]
                        for dl_idx in range(len(self.valid_data_loader))}
        if self.args.rank == 0:
            print("Top-1 Accuracy for Frame Prediction:", total_val_loss[0] / total_val_num[0])

        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = int(self.data_loader[dl_idx].n_samples / self.n_gpu)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def verbose(epoch, metrics, mode, name="TEST"):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f", R@5: {r5:.1f}, R@10: {r10:.1f}, R@50: {r50:.1f}"
    msg += f", MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    print(msg)


def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res
