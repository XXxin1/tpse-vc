import torch.nn.functional as F
from math import ceil
from tensorboardX import SummaryWriter
import torch
import yaml
from data.vctk_dataset import *
import soundfile as sf


def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


def recon_criterion_l2(predict, target):
    target = target.detach()
    return torch.mean((predict - target) ** 2)


def multi_recon_criterion_l2(predict, target):
    loss = 0.0
    for i in range(0, len(predict)):
        loss += recon_criterion_l2(predict[i], target[i])
    return loss / len(predict)


def padding_for_inference(inp):
    pad_len = 0
    while ceil(inp.shape[-1] / 8) % 2 != 0 or ceil(inp.shape[-1] / 4) % 2 != 0 or ceil(inp.shape[-1] / 2) % 2 != 0:
        inp = F.pad(inp, [0, 1], 'reflect')
        pad_len += 1
    return inp, pad_len


def grad_clip(net_list, max_grad_norm):
    for net in net_list:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)


def save_config(config, args):
    with open(f'{args.store_model_path}.config.yaml', 'w') as f:
        yaml.dump(config, f)
    with open(f'{args.store_model_path}.args.yaml', 'w') as f:
        yaml.dump(vars(args), f)
    return


def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)


def get_data_loaders(config, args):
    data_dir = args.data_dir
    train_dataset = VCTKDateset(os.path.join(data_dir, args.train_set))
    in_test_dataset = VCTKDateset_name(os.path.join(data_dir, args.seen_set))
    out_test_dataset = VCTKDateset_name(os.path.join(data_dir, args.unseen_set))

    train_dataloader = DataLoader(train_dataset, batch_size=config['data_loader']['batch_size'], shuffle=True,
                                  num_workers=4, drop_last=True, pin_memory=True,
                                  collate_fn=VCTK_collate)
    in_test_dataloader = DataLoader(in_test_dataset, batch_size=1, shuffle=True,
                                    num_workers=1)
    out_test_dataloader = DataLoader(out_test_dataset, batch_size=1, shuffle=True,
                                     num_workers=1)

    return infinite_iter(train_dataloader), infinite_iter(in_test_dataloader), infinite_iter(out_test_dataloader)


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


def write_waveform(wav_data, iteration, mode, store_wav_path, file_name, sample_rate, logger):
    os.makedirs(os.path.join(store_wav_path, str(iteration), mode), exist_ok=True)
    sf.write(os.path.join(store_wav_path, str(iteration), mode, file_name), wav_data, sample_rate)
    logger.audio_summary(f'{mode}/{file_name}', wav_data, iteration, sample_rate)
    return


class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def scalars_summary(self, tag, dictionary, step):
        self.writer.add_scalars(tag, dictionary, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

    def audio_summary(self, tag, value, step, sr):
        self.writer.add_audio(tag, value, step, sample_rate=sr)
