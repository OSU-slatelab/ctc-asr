from models import *
from util import *
from data import *
from tqdm import tqdm
from copy import deepcopy
from sklearn.metrics import f1_score, accuracy_score
from contextlib import nullcontext
import torch.distributed as dist
import numpy as np
import copy
import pdb
import random
import math
import torch
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, cfg, tokenizer, model, data, device, optimizer, sampler=None, rank=0, checkpoint=None, data_val=None, logger=None, writer=None):
        if sampler is not None:
            self.shuffle = False
        else:
            self.shuffle = True
        self.cfg = cfg
        self.model = model
        self.logger = logger
        self.writer = writer
        self.sampler = sampler
        self.rank = rank
        self.data = data
        self.data_val = data_val
        self.device = device
        self.optimizer = optimizer
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)

        eff_bsz = cfg.trainer.batch_size / cfg.distributed.world_size
        self.update_after = math.ceil(eff_bsz / cfg.trainer.bsz_small)

        collator = CollatorCTC(cfg, tokenizer)
        self.loader = torch.utils.data.DataLoader(data, batch_size=cfg.trainer.bsz_small, shuffle=self.shuffle, num_workers=4, collate_fn=collator, pin_memory=True, sampler=sampler)
        self.statsE = statRecorder('loss_ctc', ddp=cfg.distributed.ddp)
        self.statsI = statRecorder('loss_ctc', ddp=cfg.distributed.ddp)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.trainer.lr, epochs=cfg.trainer.nepochs, steps_per_epoch=math.ceil(1. * len(self.loader) / self.update_after), anneal_strategy=cfg.trainer.anneal_strategy, pct_start=cfg.trainer.pct_start, div_factor=cfg.trainer.div_factor, final_div_factor=cfg.trainer.final_div_factor)
        if checkpoint is not None and cfg.trainer.load_sch:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def tensorboard_log(self, lr, count):
        dct = self.statsI.get()
        for lossName, lossVal in dct.items():
            self.writer.add_scalar(lossName, lossVal, count)
        self.writer.add_scalar('learning rate', lr, count)

    def forward(self, batch):
        pred, _ = self.model(batch)
        self.loss_ctc = self.ctc_loss(pred, batch.target, batch.logitLens.int(), batch.targetLens.int()) / self.update_after
        loss = self.loss_ctc
        return loss

    def train(self):
        training_steps = self.cfg.trainer.nepochs * math.ceil(1. * len(self.loader) / self.update_after)
        print(f'This model is being trained for {self.cfg.trainer.nepochs} epochs = {training_steps} steps as per the batch size = {self.cfg.trainer.batch_size}')
        iterations = self.cfg.trainer.iterations_done
        for epoch in range(self.cfg.trainer.epochs_done+1, self.cfg.trainer.nepochs+1):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            print(f'Running epoch {epoch}.')
            step = 0
            self.statsE.reset()
            self.statsI.reset()
            self.optimizer.zero_grad()
            for batch in tqdm(self.loader):
                if not batch:
                    continue
                self.model.train()
                step += 1

                batch.load2gpu(self.device)

                if step % self.update_after != 0 and step != len(self.loader):
                    with self.model.no_sync() if self.cfg.distributed.ddp else nullcontext():
                        loss = self.forward(batch)
                        #####
                        if torch.isnan(loss):
                            print(f'encountered NaN loss!! please double check')
                            with open("nan_batch_instance.pkl", "wb") as f:
                                pickle.dump(batch, f)
                            torch.save(batch, "nan_batch_instance.pt")
                            if self.rank == 0 or not self.cfg.distributed.ddp:
                                checkpoint = {'state_dict':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'scheduler':self.scheduler.state_dict(), 'epochs_done':epoch, 'iterations_done':iterations}
                                save_checkpoint(checkpoint, f'nan_model.pth.tar')
                            continue
                        ####
                        loss.backward()
                        for loss_name in self.statsE.losses:
                            self.statsE.backward(loss_name, getattr(self, loss_name).detach())
                            self.statsI.backward(loss_name, getattr(self, loss_name).detach())
                else:
                    loss = self.forward(batch)
                    ###
                    if torch.isnan(loss):
                        print(f'encountered NaN loss!! please double check')
                        with open("nan_batch_instance.pkl", "wb") as f:
                            pickle.dump(batch, f)
                        torch.save(batch, "nan_batch_instance.pt")
                        if self.rank == 0 or not self.cfg.distributed.ddp:
                            checkpoint = {'state_dict':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'scheduler':self.scheduler.state_dict(), 'epochs_done':epoch, 'iterations_done':iterations}
                            save_checkpoint(checkpoint, f'nan_model.pth.tar')
                        continue
                    ###
                    loss.backward()
                    for loss_name in self.statsE.losses:
                        self.statsE.backward(loss_name, getattr(self, loss_name).detach())
                        self.statsI.backward(loss_name, getattr(self, loss_name).detach())
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.statsE.accumulate()
                    self.statsI.accumulate()
                    self.statsE.reset()
                    self.statsI.reset()
                    iterations += 1
                    if iterations % self.cfg.trainer.log_after == 0 and (self.rank == 0 or not self.cfg.distributed.ddp):
                        self.tensorboard_log(self.scheduler.get_last_lr()[0], iterations)
                        self.statsI.empty()

            if self.rank == 0 or not self.cfg.distributed.ddp:
                log = f'| {epoch} |{self.statsE.display()}| lr = {self.scheduler.get_last_lr()} |'
                print(log)
                self.logger.info(log)
            self.statsE.empty()
            if self.rank == 0 or not self.cfg.distributed.ddp:
                checkpoint = {'state_dict':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'scheduler':self.scheduler.state_dict(), 'epochs_done':epoch, 'iterations_done':iterations}
                save_checkpoint(checkpoint, f'{self.cfg.paths.save_path}')

class TrainerSC(Trainer):
    def __init__(self, cfg, tokenizer, model, data, device, optimizer, sampler=None, rank=0, checkpoint=None, data_val=None, logger=None, writer=None):
        super(TrainerSC, self).__init__(cfg, tokenizer, model, data, device, optimizer, sampler=sampler, rank=rank, checkpoint=checkpoint, data_val=data_val, logger=logger, writer=writer)
        collator = CollatorSCCTC(cfg, tokenizer)
        self.loader = torch.utils.data.DataLoader(data, batch_size=cfg.trainer.bsz_small, shuffle=self.shuffle, num_workers=4, collate_fn=collator, pin_memory=True, sampler=sampler)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.trainer.lr, epochs=cfg.trainer.nepochs, steps_per_epoch=math.ceil(1. * len(self.loader) / self.update_after), anneal_strategy=cfg.trainer.anneal_strategy, pct_start=cfg.trainer.pct_start, div_factor=cfg.trainer.div_factor, final_div_factor=cfg.trainer.final_div_factor)
        if checkpoint is not None and cfg.trainer.load_sch:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.statsE = statRecorder('loss_last', 'loss_interim', ddp=cfg.distributed.ddp)
        self.statsI = statRecorder('loss_last', 'loss_interim', ddp=cfg.distributed.ddp)

    def forward(self, batch):
        pred, pred_inter = self.model(batch)
        self.loss_last = self.ctc_loss(pred, batch.target, batch.logitLens.int(), batch.targetLens.int()) / self.update_after
        self.loss_interim = self.ctc_loss(pred_inter, batch.targetInter, batch.logitLensInter.int(), batch.targetLensInter.int()) / self.update_after
        loss = (self.loss_last + self.loss_interim) / self.cfg.model.num_ctc
        return loss

class TrainerHC(Trainer):
    def __init__(self, cfg, tokenizer, inter_tokenizers, model, data, device, optimizer, sampler=None, rank=0, checkpoint=None, data_val=None, logger=None, writer=None):
        super(TrainerHC, self).__init__(cfg, tokenizer, model, data, device, optimizer, sampler=sampler, rank=rank, checkpoint=checkpoint, data_val=data_val, logger=logger, writer=writer)

        collator = CollatorHCCTC(cfg, tokenizer, inter_tokenizers)
        self.loader = torch.utils.data.DataLoader(data, batch_size=cfg.trainer.bsz_small, shuffle=self.shuffle, num_workers=4, collate_fn=collator, pin_memory=True, sampler=sampler)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.trainer.lr, epochs=cfg.trainer.nepochs, steps_per_epoch=math.ceil(1. * len(self.loader) / self.update_after), anneal_strategy=cfg.trainer.anneal_strategy, pct_start=cfg.trainer.pct_start, div_factor=cfg.trainer.div_factor, final_div_factor=cfg.trainer.final_div_factor)
        if checkpoint is not None and cfg.trainer.load_sch:
            print(f'Loading scheduler.')
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.statsE = statRecorder('loss_last', ddp=cfg.distributed.ddp)
        self.statsI = statRecorder('loss_last', ddp=cfg.distributed.ddp)
        for i in range(cfg.model.num_ctc-1):
            self.statsE.add_loss(f'loss_ctc{i}')
            self.statsI.add_loss(f'loss_ctc{i}')

    def forward(self, batch):
        pred, pred_inter = self.model(batch)
        self.loss_last = self.ctc_loss(pred, batch.target, batch.logitLens.int(), batch.targetLens.int()) / self.update_after
        loss = self.loss_last
        for i in range(self.cfg.model.num_ctc-1):
            loss_i = self.ctc_loss(pred_inter[i], batch.targetInter[i], batch.logitLens.int(), batch.targetLensInter[i].int()) / self.update_after
            setattr(self, f'loss_ctc{i}', loss_i)
            loss = loss + loss_i
        loss = loss / self.cfg.model.num_ctc
        return loss

