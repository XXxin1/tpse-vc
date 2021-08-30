from trainer import Trainer
from argparse import ArgumentParser
import torch
import yaml
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d',
                        default='/files/xxx/VC/VCTK/VCTK-Corpus/melspectrogram_vctk/')
    parser.add_argument('-train_set', default='train')
    parser.add_argument('-seen_set', default='test_seen')
    parser.add_argument('-unseen_set', default='test_unseen')
    parser.add_argument('-logdir', default='./output/log/')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_opt', action='store_true')
    parser.add_argument('--multigpus', action='store_true')
    parser.add_argument('-store_model_path', default='./output/model')
    parser.add_argument('-load_model_path', default='./output/model')
    parser.add_argument('-summary_steps', default=1000, type=int)
    parser.add_argument('-save_steps', default=10000, type=int)
    parser.add_argument('-iters', default=200000, type=int)
    parser.add_argument('-tag', '-t', default='init')

    # inference
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=22050, type=int)
    parser.add_argument('-test_batch_size', help='test audio size', default=4, type=int)
    parser.add_argument('-store_wav_path', default='./output/sample')
    args = parser.parse_args()

    # load config file
    with open(args.config) as f:
        config = yaml.load(f)

    print("Creating directory: {}".format(args.store_model_path))
    os.makedirs(args.store_model_path, exist_ok=True)
    print("Creating directory: {}".format(args.store_wav_path))
    os.makedirs(args.store_wav_path, exist_ok=True)

    trainer = Trainer(config=config, args=args)
    trainer.cuda()
    iteration = 0
    if args.multigpus:
        ngpus = torch.cuda.device_count()
        print("Number of GPUs: %d" % ngpus)
        trainer.model = torch.nn.DataParallel(
            trainer.model, device_ids=range(ngpus))
    if args.load_model:
        iteration = trainer.load_model(args.multigpus)
    
    trainer.train_model(iteration)
