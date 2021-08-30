import yaml
from utils.operation_util import *
from math import ceil


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        ns = 0.01
        n_class = 89

        self.ns = ns
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=2)
        self.conv6 = nn.Conv2d(512, 32, kernel_size=1)
        self.conv7 = nn.Conv2d(32, 1, kernel_size=(3, 4))
        self.conv_classify = nn.Conv2d(32, n_class, kernel_size=(3, 2))
        self.ins_norm1 = nn.InstanceNorm2d(self.conv1.out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(self.conv2.out_channels)
        self.ins_norm3 = nn.InstanceNorm2d(self.conv3.out_channels)
        self.ins_norm4 = nn.InstanceNorm2d(self.conv4.out_channels)
        self.ins_norm5 = nn.InstanceNorm2d(self.conv5.out_channels)
        self.ins_norm6 = nn.InstanceNorm2d(self.conv6.out_channels)

    def conv_block(self, x, conv_layer, after_layers):
        out = pad_layer_dis(x, conv_layer, is_2d=True)
        out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        return out

    def forward(self, x, classify=False):
        x = torch.unsqueeze(x, dim=1)
        out = self.conv_block(x, self.conv1, [self.ins_norm1])
        out = self.conv_block(out, self.conv2, [self.ins_norm2])
        out = self.conv_block(out, self.conv3, [self.ins_norm3])
        out = self.conv_block(out, self.conv4, [self.ins_norm4])
        out = self.conv_block(out, self.conv5, [self.ins_norm5])
        out = self.conv_block(out, self.conv6, [self.ins_norm6])
        # GAN output value
        val = self.conv7(out)
        val = val.view(val.size(0), -1)
        mean_val = torch.mean(val, dim=1)
        if classify:
            # classify
            logits = self.conv_classify(out)
            logits = logits.view(logits.size(0), -1)
            # print logits shape
            return mean_val, logits
        else:
            return mean_val

    def calc_dis_fake_loss(self, input_fake):
        D_fake = self.forward(input_fake, classify=False)
        return D_fake

    def calc_dis_real_loss(self, input_real):
        D_real = self.forward(input_real, classify=False)
        return D_real

    def calc_gen_loss(self, input_fake):
        D_fake = self.forward(input_fake, classify=False)
        loss = -torch.mean(D_fake)
        return loss

    def calculate_gradients_penalty(self, x_real, x_fake):
        alpha = torch.rand(x_real.size(0))
        alpha = alpha.view(x_real.size(0), 1, 1)
        alpha = alpha.cuda() if torch.cuda.is_available() else alpha
        alpha = torch.autograd.Variable(alpha)
        interpolates = alpha * x_real + (1 - alpha) * x_fake
        disc_interpolates = self.forward(interpolates)

        use_cuda = torch.cuda.is_available()
        grad_outputs = torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(disc_interpolates.size())

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_penalty = (1. - torch.sqrt(1e-12 + torch.sum(gradients.view(gradients.size(0), -1) ** 2, dim=1))) ** 2
        gradients_penalty = torch.mean(gradients_penalty)
        return gradients_penalty

    def cal_dis_loss(self, x_real, x_fake):
        D_real = self.calc_dis_real_loss(x_real)
        D_fake = self.calc_dis_fake_loss(x_fake)
        w_dis = torch.mean(D_real - D_fake)
        gp = self.calculate_gradients_penalty(x_real, x_fake)
        return w_dis, gp


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.speaker_encoder = SpeakerEncoder(**config['SpeakerEncoder'])
        self.content_encoder = ContentEncoder(**config['ContentEncoder'])
        self.decoder = Decoder(**config['Decoder'])

    def forward(self, one_image, model_set):
        content, speaker_code = self.encode(one_image, model_set)
        dec, merge_feature = self.decode(content, speaker_code)
        return dec, merge_feature

    def encode(self, source, target):
        content = self.content_encoder(source)
        speaker_code = self.speaker_encoder(target)
        return content, speaker_code


class SPAttention(nn.Module):
    def __init__(self, c_h):
        super(SPAttention, self).__init__()
        self.f = nn.Conv1d(c_h, c_h, kernel_size=1)
        self.g = nn.Conv1d(c_h, c_h, kernel_size=1)
        self.h = nn.Conv1d(c_h, c_h, kernel_size=1)
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv1d(c_h, c_h, kernel_size=1)

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        F = F.permute(0, 2, 1)
        S = torch.bmm(F, G)
        S = self.sm(S)
        O = torch.bmm(H, S.permute(0, 2, 1))
        O = self.out_conv(O)
        return O


class SpeakerEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size,
                 bank_size, bank_scale, c_bank,
                 n_conv_blocks,
                 subsample, act, dropout_rate):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
            [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                                                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                                                 for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.output_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.spattention_layers = nn.ModuleList([SPAttention(c_h) for _ in range(3)])

    def conv_blocks(self, inp, content):
        out = inp
        outputs = []
        a = 0
        # conv blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
                speaker_code = self.spattention_layers[a](content[a], y)
                outputs.append(speaker_code)
                a += 1
            out = y + out
        return out, outputs

    def forward(self, x, content):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
      
        out, outputs = self.conv_blocks(out, content)
        output = pad_layer(out, self.output_layer)
        outputs[-1] = self.spattention_layers[-1](content[-1], output)
        return outputs, output


class ContentEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size,
                 bank_size, bank_scale, c_bank,
                 n_conv_blocks, subsample,
                 act, dropout_rate):
        super(ContentEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
            [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                                                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                                                 for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.output_layer = nn.Conv1d(c_h, c_out, kernel_size=1)

    def conv_block(self, inp):
        out = inp
        outputs = []
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
                outputs.append(y)
            out = y + out
        return out, outputs

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)

        out, outputs = self.conv_block(out)
        output = pad_layer(out, self.output_layer)
        outputs[-1] = output
        return outputs


class Decoder(nn.Module):
    def __init__(self,
                 c_in, c_h, c_out,
                 kernel_size,
                 n_conv_blocks, upsample, act, dropout_rate):
        super(Decoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act(act)
        self.in_conv_layer = (nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList([(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) for _ \
                                                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList( \
            [(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size)) \
             for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.cond_affine_layers = nn.ModuleList(
            [(nn.Conv1d(c_h, c_h, kernel_size=1)) for _ in range(n_conv_blocks * 2)])
        self.z_affine_layers = nn.ModuleList(
            [(nn.Conv1d(c_h, c_h, kernel_size=1)) for _ in range(n_conv_blocks // 2)])
        self.out_conv_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.out_layer_1 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.out_layer_2 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(4)])
        self.adain_affine_layers = nn.ModuleList([(nn.Linear(c_h, c_h * 2)) for _ in range(2)])

    def linear_block(self, inp):
        spk_emd = inp
        # process spk_emd
        spk_emd = self.pooling_layer(spk_emd).squeeze(2)
        spk_emdding = self.dense_layers[0](spk_emd)
        spk_emdding = self.act(spk_emdding)
        spk_emdding = self.dense_layers[1](spk_emdding)
        spk_emdding = self.act(spk_emdding)

        spk_emd = spk_emdding + spk_emd
        spk_emdding = self.dense_layers[2](spk_emd)
        spk_emdding = self.act(spk_emdding)
        spk_emdding = self.dense_layers[3](spk_emdding)
        spk_emdding = self.act(spk_emdding)

        spk_emd = spk_emdding + spk_emd
        return spk_emd

    def unet_block(self, inp, cond, z):
        out = inp
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = out
            a = l
            if self.upsample[l] <= 1 and l != 0:
                y = y + self.norm_layer(pad_layer((z[- (a // 2 + 1)]), self.z_affine_layers[l // 2]))

            y = pad_layer(y, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = y + pad_layer((cond[- (a // 2 + 1)]), self.cond_affine_layers[2 * l])

            y = pad_layer(y, self.second_conv_layers[l])

            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
                a += 2
            if l != (self.n_conv_blocks - 1):
                y = self.norm_layer(y)
                y = self.act(y)
                y = y + pad_layer((cond[- (a // 2 + 1)]), self.cond_affine_layers[2 * l + 1])
            else:
                y = self.act(y)

            if self.upsample[l] > 1:
                out = y + upsample(out, scale_factor=self.upsample[l])
            else:
                out = y + out
        return out

    def adain_block(self, inp, spk_emd):
        out = inp
        # adain
        y = pad_layer(out, self.out_layer_1)
        y = self.norm_layer(y)
        y = append_cond(y, self.adain_affine_layers[0](spk_emd))
        y = self.act(y)
        y = pad_layer(y, self.out_layer_2)
        y = self.norm_layer(y)
        
        y = append_cond(y, self.adain_affine_layers[1](spk_emd))
        y = self.act(y)
        out = y + out
        return out

    def forward(self, z, cond, spk_emd):

        spk_emd = self.linear_block(spk_emd)
        out = z[-1]
        out = pad_layer(out, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.unet_block(out, cond, z)
        out = self.adain_block(out, spk_emd)
        out = pad_layer(out, self.out_conv_layer)
        return out


if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.load(f)
    t = torch.rand([1, 80, 128])
    pad_len = 0
    while ceil(t.shape[-1] / 4) % 2 != 0 or ceil(t.shape[-1] / 2) % 2 != 0:
        t = F.pad(t, [0, 1], 'reflect')
        pad_len += 1

    print(t.shape)
    t1 = torch.rand([1, 80, 567])

    l = 0
    dis = Discriminator(config)
    print(dis(t, classify=True)[1].shape, '---')
    gen = Generator(config)
    co_code = gen.content_encoder(t)
    print(len(co_code))
    print(co_code[0].shape)
    print(co_code[1].shape)
    cl_code = gen.speaker_encoder(t1, co_code)
    print(cl_code[0].shape)
    print(cl_code[1].shape)
    xt = gen.decoder(co_code, cl_code)
    xt = xt[:, :, :-pad_len]
    print(xt.shape)
