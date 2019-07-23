from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from utils.util import AverageMeter
from utils.lr_scheduler import CosineLr
from models.yolo_loss import yoloLoss,yoloLoss_GIOU
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.checkpoint import save_checkpoint,load_checkpoint
import os
import numpy as np

class ModleWithLoss(nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return loss, loss_stats

class BaseTrainer(object):

    def __init__(self,config ,model, optimizer=None,loader=None):
        self.config=config
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(self.config)
        self.model_with_loss = ModleWithLoss(model, self.loss)
        if loader is None or len(loader) !=2 :
            raise Exception("not two loader")
        self.train_loader = loader[0]
        self.val_loader = loader[1]
        self.num_train_iter=len(self.train_loader)
        self.num_val_iter=len(self.val_loader)
        self.lr_just = CosineLr(self.optimizer, self.config.OPTIM,self.num_train_iter)
        self.writer=SummaryWriter(logdir=self.config.TRAIN['LOGS'])
        self.loss_log=[]

    def set_device(self, device):

        self.model_with_loss = self.model_with_loss.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def train_epoch(self,epoch):
        model_with_loss = self.model_with_loss
        model_with_loss.train()
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        end = time.time()
        for iter_id, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            for k in batch:
                batch[k] = batch[k].to(device=self.config.TRAIN['DEVICE'], non_blocking=True)
            loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if torch.isinf(loss) or torch.isnan(loss):
                raise Exception("nan/inf in sum loss")
            self.optimizer.zero_grad()
            lr = self.lr_just.step(1e-3)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('train/lr',lr,self.lr_just.global_step)
            batch_time.update(time.time() - end)
            end = time.time()
            show_str = '[%d/%d/%d] ' % (epoch + 1,iter_id,self.num_train_iter)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                self.writer.add_scalar('train/'+l,avg_loss_stats[l].avg,epoch*self.num_train_iter+iter_id)
                show_str+='{}:{:0.4} '.format(l, avg_loss_stats[l].avg)
            print(show_str)


    def val_epoch(self,epoch):
        model_with_loss = self.model_with_loss
        model_with_loss.eval()
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        end = time.time()
        for iter_id, batch in enumerate(self.val_loader):
            show_str = '[%d/%d/%d] ' % (epoch+1,iter_id+1, self.num_val_iter)
            data_time.update(time.time() - end)
            with torch.no_grad():
                for k in batch:
                    batch[k] = batch[k].to(device=self.config.TRAIN['DEVICE'], non_blocking=True)
                loss, loss_stats = model_with_loss(batch)
            batch_time.update(time.time() - end)
            end = time.time()
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                self.writer.add_scalar('val/'+l, avg_loss_stats[l].avg,epoch*self.num_val_iter+iter_id)
                show_str+=' {}:{:0.4}   '.format(l, avg_loss_stats[l].avg)
            print(show_str)
        save_checkpoint(model_with_loss.model,self.config.TRAIN['CHECKPOINT']+'/model_%d.pth'%epoch)


    def run(self):

        for epoch in range(self.config.OPTIM['EPOCH']):
            self.train_epoch(epoch)
            self.val_epoch(epoch)


    def _get_losses(self, config):
        raise NotImplementedError


class YolodetTrainer(BaseTrainer):
    def __init__(self, config, model, optimizer=None, loader=None):
        super(YolodetTrainer,self).__init__(config, model, optimizer, loader)

    def _get_losses(self, config):
        loss_states = ['loss', 'xywh_loss', 'conf_loss', 'prob_loss']
        self.use_giou = config.DATASET.USE_GIOU
        if self.use_giou:
            loss = yoloLoss_GIOU()
        else:
            loss = yoloLoss()
        return loss_states, loss