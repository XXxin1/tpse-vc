from network import Generator, Discriminator
from utils.common_util import *
import copy
from torch import nn


class VCModel(nn.Module):
    def __init__(self, config):
        super(VCModel, self).__init__()
        self.config = config
        self.__build_model()

    def __build_model(self):
        self.gen = Generator(self.config)
        self.dis = Discriminator(self.config)
        self.gen_test = copy.deepcopy(self.gen)
        return

    def forward(self, co_data, cl_data, mode):
        xa = co_data.cuda()
        xb = cl_data.cuda()
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
            l_content = self.config['lambda']['content_w'] * multi_recon_criterion_l2(c_xt, c_xa)
            l_speaker = self.config['lambda'][
                            'speaker_w'] * multi_recon_criterion_l2(s_xt, s_xb)

            l_x_rec = self.config['lambda']['r_w'] * recon_criterion(xr, xa)
            l_adv = self.config['lambda']['gan_w'] * 0.5 * (l_adv_t + l_adv_r)
            l_total = (l_adv + l_x_rec + l_content + l_speaker)
            l_total.backward()
            grad_clip([self.gen], self.config['lambda']['max_grad_norm'])
            return l_total, l_x_rec, l_content, l_speaker, l_adv
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

    def inference(self, xa, xb):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()
        xa, pad_len = padding_for_inference(xa)
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
        if pad_len != 0:
            xt = xt[:, :, :-pad_len]
            xr = xr[:, :, :-pad_len]
            xt_current = xt_current[:, :, :-pad_len]
            xr_current = xr_current[:, :, :-pad_len]
        self.train()
        return xr_current, xt_current, xr, xt
