import json
from argparse import ArgumentParser
import librosa
import sys
from tqdm import tqdm
from librosa.filters import mel as librosa_mel_fn
import soundfile as sf
import os
import torch.nn as nn
import torch
import numpy as np
import yaml
from torch.nn import functional as F

sys.path.append("..")
from vc_model import VCModel
from utils.common_util import get_model_list

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class Audio2Mel(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            sampling_rate=22050,
            n_mel_channels=240,
            mel_fmin=0.0,
            mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))

        return log_mel_spec


n_fft = 1024
hop_length = 256
win_length = 1024
sampling_rate = 22050
n_mel_channels = 80
extract_func = Audio2Mel(n_fft, hop_length, win_length, sampling_rate, n_mel_channels)
sr = 22050


def convert_file(path, trim=False):
    y, _ = librosa.load(path, sr=22050)
    if trim:
        y, index = librosa.effects.trim(y, top_db=20)

    y = torch.from_numpy(y)

    y = y[None, None]
    mel = extract_func(y)
    mel = mel.numpy()
    mel = mel[0]

    return mel.astype(np.float32)


class Inferencer(nn.Module):
    def __init__(self, config, args):
        super(Inferencer, self).__init__()
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)
        # args store other information
        self.args = args
        print(self.args)

        # init the model with config
        self.build_model()
        self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        # load model
        self.load_model()

    def load_model(self):
        this_model = self.model
        print(f'Load model from {self.args.load_model_path}')
        last_model_name = get_model_list(self.args.load_model_path, "gen")
        # last_model_name = os.path.join(self.args.load_model_path, 'gen_00200000.pt')
        # last_model_name = os.path.join(self.args.load_model_path, 'gen_00300000.pt')
        state_dict = torch.load(last_model_name)
        this_model.gen.load_state_dict(state_dict['gen'])
        this_model.gen_test.load_state_dict(state_dict['gen_test'])
        iterations = int(last_model_name[-11:-3])

        print('Resume from iteration %d' % iterations)
        return iterations

    def build_model(self):
        self.model = VCModel(self.config)
        self.model.eval()
        return

    def vocoder_inverse(self, inp):
        return self.vocoder.inverse(inp).detach().cpu().numpy()[0]

    def write_wav_to_file(self, wav_data, sub, file_name, sample_rate):
        os.makedirs(os.path.join(self.args.store_wav_path, sub), exist_ok=True)

        sf.write(os.path.join(self.args.store_wav_path, sub, file_name), wav_data, sample_rate)
        return

    def inference_from_mel(self, source_mel, source_path, target_mel, target_path):
        source_id = source_path.split('_')[0]
        target_id = target_path.split('_')[0]

        source_ = source_path.split('.')[0]
        target_ = target_path.split('.')[0]

        origin_source_path = os.path.join(args.origin_data_dir, source_id, source_file)
        origin_target_path = os.path.join(args.origin_data_dir, target_id, target_file)

        sub = source_ + '_' + target_
        this_model = self.model
        src_mel = source_mel.cuda()
        tgt_mel = target_mel.cuda()
        xr_current, xt_current, xr, xt = this_model.inference(src_mel, tgt_mel)
        xt = self.vocoder_inverse(xt)
        src = self.vocoder_inverse(src_mel)
        tgt = self.vocoder_inverse(tgt_mel)
        self.write_wav_to_file(src, sub,
                               'vocoder_source_' + os.path.basename(source_path),
                               self.args.sample_rate)
        self.write_wav_to_file(tgt, sub,
                               'vocoder_target_' + os.path.basename(target_path),
                               self.args.sample_rate)
        self.write_wav_to_file(xt, sub, 'converted_proposed.wav', self.args.sample_rate)

        origin_source_audio, index = librosa.effects.trim(
            librosa.load(os.path.join(origin_source_path), sr=48000)[0], top_db=20)
        origin_target_audio, index = librosa.effects.trim(
            librosa.load(os.path.join(origin_target_path), sr=48000)[0], top_db=20)

        sf.write(os.path.join(self.args.store_wav_path, sub, 'origin_source_' + source_file), origin_source_audio,
                 48000)
        sf.write(os.path.join(self.args.store_wav_path, sub, 'origin_target_' + target_file), origin_target_audio,
                 48000)

        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='./config.yaml')
    parser.add_argument('-load_model_path',
                        default='./output/model')
    parser.add_argument('-test_file', '-s',
                        default='/files/xxx/VC/speaker-attention-vc-all/speechmetrics/objective_evaluation_male2male.json')
    parser.add_argument('-origin_data_dir', '-o', default='/files/xxx/VC/VCTK/VCTK-Corpus/wav48/')
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=22050, type=int)
    parser.add_argument('-store_wav_path', default='inference/m2m')
    args = parser.parse_args()
    # load config file
    with open(args.config) as f:
        config = yaml.load(f)
    inferencer = Inferencer(config=config, args=args).cuda()

    with open(os.path.join(args.test_file), 'r') as f:
        test = json.load(f)

    for test_ in tqdm(test):
        source_file, target_file = test_[0], test_[1]
        source_id = source_file.split('_')[0]
        target_id = target_file.split('_')[0]
        source_wav_path = os.path.join(args.origin_data_dir, source_id, source_file)
        target_wav_path = os.path.join(args.origin_data_dir, target_id, target_file)

        source_mel = torch.from_numpy(convert_file(source_wav_path, trim=True))
        source_mel = source_mel.unsqueeze(0)
        target_mel = torch.from_numpy(convert_file(target_wav_path, trim=True))
        target_mel = target_mel.unsqueeze(0)
        inferencer.inference_from_mel(source_mel, source_file, target_mel, target_file)
