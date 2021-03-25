from torch.utils.data import DataLoader
import torch
import numpy as np
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model import FewShotGen, GPPatchMcResDis
from data_utils import get_data_loader
from utils import *
import copy
from functools import reduce
from collections import defaultdict
from scipy.io.wavfile import write
from scipy.io.wavfile import read
from torch import nn
from math import ceil


class Solver(nn.Module):
    def __init__(self, config, args):
        super(Solver, self).__init__()
        # config store the value of hyperparameters, turn to attr by cak
        self.config = config
        # print(config)

        # args store other information
        self.args = args
        # print(self.args))

        # init the model with config
        self._build_model()
        # Normailze
        # self.mean = torch.from_numpy(np.load(os.path.join(self.args.data_dir, 'mean.npy'))).float().cuda().view(1, 80,
        #                                                                                                         1)
        # self.std = torch.from_numpy(np.load(os.path.join(self.args.data_dir, 'std.npy'))).float().cuda().view(1, 80, 1)

    def _build_model(self):
        self.gen = FewShotGen(self.config)
        self.dis = GPPatchMcResDis(self.config)
        self.gen_test = copy.deepcopy(self.gen)
        return

    def forward(self, co_data, cl_data, mode):
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()


        if mode == 'step_gen':
            c_xa = self.gen.content_encoder(xa)
            s_xa, s_xa_ori = self.gen.speaker_encoder(xa, c_xa)
            s_xb, s_xb_ori = self.gen.speaker_encoder(xb, c_xa)


            xt = self.gen.decoder(c_xa, s_xb, s_xb_ori)  # translation
            xr = self.gen.decoder(c_xa, s_xa, s_xa_ori)  # reconstruction

            l_adv_t = self.dis.calc_gen_loss(xt)
            l_adv_r = self.dis.calc_gen_loss(xr)

            c_xt = self.gen.content_encoder(xt)
            s_xt, _ = self.gen.speaker_encoder(xt, c_xt)
            

            # reconstruction encoder loss
            l_content = calc_content_loss(c_xt, c_xa) 

            l_speaker = calc_content_loss(s_xt, s_xb) 

            l_x_rec = recon_criterion(xr, xa) 
            l_adv = 0.5 * (l_adv_t + l_adv_r)
            l_total = (self.config['lambda']['gan_w'] * l_adv + self.config['lambda']['r_w'] * l_x_rec +
                       self.config['lambda']['content_w'] * l_content + self.config['lambda'][
                           'speaker_w'] * l_speaker)
            l_total.backward()
            grad_clip([self.gen], self.config['lambda']['max_grad_norm'])
            return l_total, l_adv, l_x_rec
        elif mode == 'step_dis':
            xb.requires_grad_()
            with torch.no_grad():
                c_xa = self.gen.content_encoder(xa)
                s_xb, s_xb_ori = self.gen.speaker_encoder(xb, c_xa)
                xt = self.gen.decoder(c_xa, s_xb, s_xb_ori)
            w_dis, gp = self.dis.cal_dis_loss(xb, xt)
            l_total = - self.config['lambda']['gan_w'] * w_dis + 10 * gp
            l_total.backward()
            grad_clip([self.dis], self.config['lambda']['max_grad_norm'])
            return l_total
        else:
            assert 0, 'Not support operation'

    def inference_one_utterance(self, xa, xb):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()
        pad_len = 0
        pad = False
        while ceil(xa.shape[-1] / 8) % 2 != 0 or ceil(xa.shape[-1] / 4) % 2 != 0 or ceil(xa.shape[-1] / 2) % 2 != 0:
            xa = F.pad(xa, [0, 1], 'reflect')
            pad_len += 1
            pad = True
        c_xa_current = self.gen.content_encoder(xa)
        s_xa_current, s_xa_current_ori = self.gen.speaker_encoder(xa, c_xa_current)
        s_xb_current, s_xb_current_ori = self.gen.speaker_encoder(xb, c_xa_current)
        xt_current = self.gen.decoder(c_xa_current, s_xb_current, s_xb_current_ori)
        xr_current = self.gen.decoder(c_xa_current, s_xa_current, s_xa_current_ori)
        c_xa = self.gen_test.content_encoder(xa)
        s_xa, s_xa_ori = self.gen_test.speaker_encoder(xa, c_xa)
        s_xb, s_xb_ori = self.gen_test.speaker_encoder(xb, c_xa)
        xt = self.gen_test.decoder(c_xa, s_xb, s_xb_ori)
        xr = self.gen_test.decoder(c_xa, s_xa, s_xa_ori)
        if pad is True:
            xt = xt[:, :, :-pad_len]
            xr = xr[:, :, :-pad_len]
            xt_current = xt_current[:, :, :-pad_len]
            xr_current = xr_current[:, :, :-pad_len]
        self.train()
        return xr_current, xt_current, xr, xt


def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


def recon_criterion_l2(predict, target):
    return torch.mean((predict - target) ** 2)


def calc_style_loss(predict, target):
    loss = 0
    for i in range(len(predict)):
        input_mean, input_std = calc_mean_std(predict[i])
        target_mean, target_std = calc_mean_std(target[i])
        loss += recon_criterion_l2(input_mean, target_mean) + recon_criterion_l2(input_std, target_std)
    return loss


def calc_content_loss(predict, target):
    loss = 0
    for i in range(len(predict)):
        loss += recon_criterion_l2(predict[i], predict[i])
    return loss


def cal_acc(logits, y_true):
    _, ind = torch.max(logits, dim=1)
    acc = torch.sum((ind == y_true).type(torch.FloatTensor)) / y_true.size(0)
    return acc


def grad_clip(net_list, max_grad_norm):
    for net in net_list:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
