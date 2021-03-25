from torch.utils.data import DataLoader
import sys
import os
import math
import torch
import torch.nn as nn
from solver import Solver
from scipy.io.wavfile import write
from scipy.io.wavfile import read
from data_utils import get_data_loader
from data_utils import TrainData, TestData
import torch.nn.functional as F
import yaml
import pickle
import copy
from utils import *


class Trainer(nn.Module):
    def __init__(self, config, args):
        super(Trainer, self).__init__()
        self.config = config
        self.args = args
        self.logger = Logger(self.args.logdir)
        # get dataloader
        self.get_data_loaders()
        # init the model with config
        self._build_model()
        self.save_config()
        self.iteration = 0
        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')


    def _build_model(self):
        self.model = Solver(self.config, self.args)
        # print('Generator: \n', self.gen)
        # print('Discriminator: \n', self.dis)
        optimizer = self.config['optimizer']
        dis_params = list(self.model.dis.parameters())
        gen_params = list(self.model.gen.parameters())
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=optimizer['lr_dis'], betas=(optimizer['beta1'], optimizer['beta2']),
            amsgrad=optimizer['amsgrad'], weight_decay=optimizer['weight_decay'])
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=optimizer['lr_gen'], betas=(optimizer['beta1'], optimizer['beta2']),
            amsgrad=optimizer['amsgrad'], weight_decay=optimizer['weight_decay'])
        # print('Generator Optimizer: \n', self.gen_opt)
        # print('Discriminator Optimizer: \n', self.dis_opt)
        self.model.gen_test = copy.deepcopy(self.model.gen)
        return

    def train_model(self):
        max_iter = self.iteration + self.args.iters
        while True:
            co_data = next(self.train_content_iter)
            cl_data = next(self.train_class_iter)


            l_total = self.dis_update(co_data, cl_data)
            meta_dis = {'dis_loss_total': torch.mean(l_total)}

            l_total, l_adv, l_x_rec = self.gen_update(co_data, cl_data, self.args.multigpus)
            meta_gen = {'gen_loss_total': torch.mean(l_total),
                        'gen_loss_gan': torch.mean(l_adv),
                        'gen_loss_x_rec': torch.mean(l_x_rec)}

            dis_loss_total = meta_dis['dis_loss_total'].item()
            gen_loss_gan = meta_gen['gen_loss_gan'].item()
            gen_loss_rec = meta_gen['gen_loss_x_rec'].item()

            print(
                f'[{self.iteration + 1}/{max_iter}]: \t d_loss={dis_loss_total:.2f}  |\t'
                f'g_loss={gen_loss_gan:.2f}, gen_loss_rec={gen_loss_rec:.2f}    ', end='\n')
            # add to logger
            if self.iteration % self.args.summary_steps == 0:
                print('=======> Testing Sampling')
                self.logger.scalars_summary(f'{self.args.tag}/gen_train', meta_gen, self.iteration)
                self.logger.scalars_summary(f'{self.args.tag}/dis_train', meta_dis, self.iteration)
                with torch.no_grad():
                    for i in range(self.args.test_batch_size):
                        co_feature, co_path = next(self.in_test_content_iter)
                        cl_feature, cl_path = next(self.in_test_class_iter)
                        # for i_ in range(len(in_co_path)):
                        self.inference_from_path(co_feature, co_path[0], cl_feature, cl_path[0], self.iteration,
                                                 'in_' + str(i), self.args.multigpus)

                    for i in range(self.args.test_batch_size):
                        co_feature, co_path = next(self.out_test_content_iter)
                        cl_feature, cl_path = next(self.out_test_class_iter)
                        # for i_ in range(len(in_co_path)):
                        self.inference_from_path(co_feature, co_path[0], cl_feature, cl_path[0], self.iteration,
                                                 'out' + str(i), self.args.multigpus)

            if (self.iteration + 1) % self.args.save_steps == 0 or self.iteration + 1 == max_iter:
                self.save_model(iteration=self.iteration, multigpus=self.args.multigpus)
                print('Saved model at iteration %d' % (self.iteration + 1))

            self.iteration += 1
            if self.iteration >= max_iter:
                print("Finish Training")
                sys.exit(0)

    def gen_update(self, co_data, cl_data, multigpus):
        self.gen_opt.zero_grad()
        meta_gan = self.model(co_data, cl_data, 'step_gen')
        self.gen_opt.step()
        this_model = self.model.module if multigpus else self.model
        update_average(this_model.gen_test, this_model.gen)
        return meta_gan

    def dis_update(self, co_data, cl_data):
        self.dis_opt.zero_grad()
        meta_dis = self.model(co_data, cl_data, 'step_dis')
        self.dis_opt.step()
        return meta_dis

    def get_data_loaders(self):
        data_dir = self.args.data_dir
        self.train_dataset = TrainData(os.path.join(data_dir),
                                       os.path.join(data_dir, self.args.train_index_file),
                                       segment_size=self.config['data_loader']['segment_size'])

        self.train_content_loader = get_data_loader(self.train_dataset,
                                                    frame_size=self.config['data_loader']['frame_size'],
                                                    batch_size=self.config['data_loader']['batch_size'],
                                                    shuffle=self.config['data_loader']['shuffle'],
                                                    num_workers=1, drop_last=False)
        self.train_class_loader = get_data_loader(self.train_dataset,
                                                  frame_size=self.config['data_loader']['frame_size'],
                                                  batch_size=self.config['data_loader']['batch_size'],
                                                  shuffle=self.config['data_loader']['shuffle'],
                                                  num_workers=1, drop_last=False)


        self.in_test_dataset = TestData(os.path.join(data_dir), os.path.join(data_dir, 'in_test_samples_100.json'))
        self.in_test_content_loader = DataLoader(self.in_test_dataset, batch_size=1, shuffle=True, num_workers=1,
                                                 pin_memory=True)
        self.in_test_class_loader = DataLoader(self.in_test_dataset, batch_size=1, shuffle=True, num_workers=1,
                                               pin_memory=True)

        self.out_test_dataset = TestData(os.path.join(data_dir), os.path.join(data_dir, 'out_test_samples_100.json'))
        self.out_test_content_loader = DataLoader(self.out_test_dataset, batch_size=1, shuffle=True, num_workers=1,
                                                  pin_memory=True)
        self.out_test_class_loader = DataLoader(self.out_test_dataset, batch_size=1, shuffle=True, num_workers=1,
                                                pin_memory=True)

        self.train_content_iter = infinite_iter(self.train_content_loader)
        self.in_test_content_iter = infinite_iter(self.in_test_content_loader)
        self.out_test_content_iter = infinite_iter(self.out_test_content_loader)
        self.train_class_iter = infinite_iter(self.train_class_loader)
        self.in_test_class_iter = infinite_iter(self.in_test_class_loader)
        self.out_test_class_iter = infinite_iter(self.out_test_class_loader)

        return

    def load_model(self, multigpus):
        this_model = self.model.module if multigpus else self.model
        print(f'Load model from {self.args.load_model_path}')
        last_model_name = get_model_list(self.args.load_model_path, "gen")
        state_dict = torch.load(last_model_name)
        this_model.gen.load_state_dict(state_dict['gen'])
        this_model.gen_test.load_state_dict(state_dict['gen_test'])
        iterations = int(last_model_name[-11:-3])

        last_model_name = get_model_list(self.args.load_model_path, "dis")
        state_dict = torch.load(last_model_name)
        this_model.dis.load_state_dict(state_dict['dis'])


        state_dict = torch.load(os.path.join(self.args.load_model_path, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        print('Resume from iteration %d' % iterations)
        return iterations

    def save_model(self, iteration, multigpus):
        this_model = self.model.module if multigpus else self.model
        # save model and discriminator and their optimizer
        gen_name = os.path.join(self.args.store_model_path, 'gen_%08d.pt' % (iteration + 1))
        dis_name = os.path.join(self.args.store_model_path, 'dis_%08d.pt' % (iteration + 1))
        opt_name = os.path.join(self.args.store_model_path, 'optimizer.pt')
        torch.save({'gen': this_model.gen.state_dict(),
                    'gen_test': this_model.gen_test.state_dict()}, gen_name)
        torch.save({'dis': this_model.dis.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)

    def save_config(self):
        with open(f'{self.args.store_model_path}.config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        with open(f'{self.args.store_model_path}.args.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    # inference

    def write_wav_to_file(self, wav_data, iteration, sub, file_name, sample_rate):
        if not os.path.exists(os.path.join(self.args.store_wav_path, str(iteration), sub)):
            os.makedirs(os.path.join(self.args.store_wav_path, str(iteration), sub))
        write(os.path.join(self.args.store_wav_path, str(iteration), sub, file_name), rate=sample_rate,
              data=wav_data)
        self.logger.audio_summary(f'{sub}/{file_name}', wav_data, iteration, sample_rate)
        return

    def inference_from_path(self, source_mel, source_path, target_mel, target_path, iteration, sub, multigpus):
        this_model = self.model.module if multigpus else self.model
        source_path = source_path.replace('npy', 'wav')
        target_path = target_path.replace('npy', 'wav')
        src_mel = source_mel.cuda()
        tar_mel = target_mel.cuda()
        # print(source_mel.shape)
        xr_current, xt_current, xr, xt = this_model.inference_one_utterance(src_mel, tar_mel)
        src_mel = self.vocoder.inverse(src_mel)
        src_mel = src_mel.detach().cpu().numpy()
        tar_mel = self.vocoder.inverse(tar_mel)
        tar_mel = tar_mel.detach().cpu().numpy()
        xr_current = self.vocoder.inverse(xr_current)
        xr_current = xr_current.detach().cpu().numpy()
        xt_current = self.vocoder.inverse(xt_current)
        xt_current = xt_current.detach().cpu().numpy()
        xr = self.vocoder.inverse(xr)
        xr = xr.detach().cpu().numpy()
        xt = self.vocoder.inverse(xt)
        xt = xt.detach().cpu().numpy()
        self.write_wav_to_file(src_mel[0], iteration, sub, 'source_' + os.path.basename(source_path),
                               self.args.sample_rate)
        self.write_wav_to_file(tar_mel[0], iteration, sub, 'target_' + os.path.basename(target_path),
                               self.args.sample_rate)
        self.write_wav_to_file(xr_current[0], iteration, sub, 'xr_current.wav', self.args.sample_rate)
        self.write_wav_to_file(xt_current[0], iteration, sub, 'xt_current.wav', self.args.sample_rate)
        self.write_wav_to_file(xr[0], iteration, sub, 'xr.wav', self.args.sample_rate)
        self.write_wav_to_file(xt[0], iteration, sub, 'xt.wav', self.args.sample_rate)
        return


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun
