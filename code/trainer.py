import torch.nn as nn
from utils.common_util import Logger, save_config, get_data_loaders, update_average, get_model_list, write_waveform
from vc_model import VCModel
import torch
import copy
import os
import sys


class Trainer(nn.Module):
    def __init__(self, config, args):
        super(Trainer, self).__init__()
        self.config = config
        self.args = args
        self.logger = Logger(self.args.logdir)

        self.train_iter, self.in_test_iter, self.out_test_iter = get_data_loaders(config, args)
        self.__bulid_model()
        save_config(self.config, self.args)
        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

    def __bulid_model(self):
        self.model = VCModel(self.config)
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
        self.model.gen_test = copy.deepcopy(self.model.gen)

        return

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

    def gen_update(self, co_data, cl_data, multigpus):
        self.gen_opt.zero_grad()
        l_gen, l_rec, l_content, l_speaker, l_adv = self.model(co_data, cl_data, 'step_gen')
        self.gen_opt.step()
        this_model = self.model.module if multigpus else self.model
        update_average(this_model.gen_test, this_model.gen)
        return torch.mean(l_gen), torch.mean(l_rec), torch.mean(l_content), torch.mean(l_speaker), torch.mean(l_adv)

    def dis_update(self, co_data, cl_data):
        self.dis_opt.zero_grad()
        l_dis = self.model(co_data, cl_data, 'step_dis')
        self.dis_opt.step()
        return torch.mean(l_dis)

    def vocoder_inverse(self, inp):
        return self.vocoder.inverse(inp).detach().cpu().numpy()[0]

    def inference(self, co_data, cl_data, src_name, tgt_name, iteration, mode, multigpus):
        this_model = self.model.module if multigpus else self.model
        co_data = co_data.cuda()
        cl_data = cl_data.cuda()
        xr_current, xt_current, xr, xt = this_model.inference(co_data, cl_data)
        src = self.vocoder_inverse(co_data)
        tgt = self.vocoder_inverse(cl_data)
        xr_current = self.vocoder_inverse(xr_current)
        xt_current = self.vocoder_inverse(xt_current)
        xr = self.vocoder_inverse(xr)
        xt = self.vocoder_inverse(xt)

        write_waveform(src, iteration, mode, self.args.store_wav_path, 'source_' + src_name + '.wav',
                       self.args.sample_rate, self.logger)
        write_waveform(tgt, iteration, mode, self.args.store_wav_path, 'target_' + tgt_name + '.wav',
                       self.args.sample_rate, self.logger)
        write_waveform(xr_current, iteration, mode, self.args.store_wav_path, 'xr_current.wav', self.args.sample_rate,
                       self.logger)
        write_waveform(xr, iteration, mode, self.args.store_wav_path, 'xr.wav', self.args.sample_rate, self.logger)
        write_waveform(xt_current, iteration, mode, self.args.store_wav_path, 'xt_current.wav', self.args.sample_rate,
                       self.logger)
        write_waveform(xt, iteration, mode, self.args.store_wav_path, 'xt.wav', self.args.sample_rate, self.logger)
        return

    def train_model(self, iteration=0):
        max_iteration = iteration + self.args.iters
        while True:
            co_data, cl_data = next(self.train_iter)

            l_dis = self.dis_update(co_data, cl_data)
            meta_dis = {'dis_loss_total': torch.mean(l_dis)}

            l_gen, l_rec, l_content, l_speaker, l_adv = self.gen_update(co_data, cl_data, self.args.multigpus)
            meta_gen = {'gen_loss_total': l_gen,
                        'gen_loss_rec': l_rec,
                        'gen_loss_content': l_content,
                        'gen_loss_style': l_speaker,
                        'gen_loss_l_adv': l_adv}

            print(f'[{iteration + 1}/{max_iteration}], loss_rec={l_rec.item():.2f}, '
                  f'loss_content={l_content.item():.2f}, loss_style={l_speaker.item():.2f}, loss_adv={l_adv.item():.2f} '
                  f'| loss_dis={l_dis.item():.2f} ',
                  end='\r')

            if (iteration + 1) % self.args.summary_steps == 0:
                print('\n======= Testing and Logging =======\n')
                self.logger.scalars_summary(f'{self.args.tag}/gen_train', meta_gen, iteration + 1)
                self.logger.scalars_summary(f'{self.args.tag}/dis_train', meta_dis, iteration + 1)
                with torch.no_grad():
                    for i in range(self.args.test_batch_size):
                        co_data, cl_data, src_name, tgt_name = next(self.out_test_iter)
                        self.inference(co_data, cl_data, src_name[0], tgt_name[0], iteration + 1, 'out_' + str(i),
                                       self.args.multigpus)

                        co_data, cl_data, src_name, tgt_name = next(self.in_test_iter)
                        self.inference(co_data, cl_data, src_name[0], tgt_name[0], iteration + 1, 'in_' + str(i),
                                       self.args.multigpus)



            if (iteration + 1) % self.args.save_steps == 0 or (iteration + 1) == max_iteration:
                self.save_model(iteration, self.args.multigpus)
                print('Saved model at iteration %d' % (iteration + 1))

            iteration += 1
            if iteration >= max_iteration:
                print("Finish Training")
                sys.exit(0)
