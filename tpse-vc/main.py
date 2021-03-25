from trainer import Trainer
from argparse import ArgumentParser, Namespace
import torch
from solver import Solver
import yaml
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d',
                        default='/files/xxx/VC/VCTK/VCTK-Corpus/mel.melgan/')
    parser.add_argument('-parallel_data_dir', '-pd',
                        default='/files/xxx/VC/data/vcc2016_training_align_melgan_npy/')
    parser.add_argument('-train_set', default='train')
    parser.add_argument('-train_index_file', default='train_samples_100.json')
    parser.add_argument('-logdir', default='./output/log/')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_opt', action='store_true')
    parser.add_argument('--multigpus', action='store_true')
    parser.add_argument('-store_model_path', default='./output/model')
    parser.add_argument('-load_model_path', default='./output/model')
    parser.add_argument('-summary_steps', default=1000, type=int)
    parser.add_argument('-save_steps', default=10000, type=int)
    parser.add_argument('-tag', '-t', default='init')
    parser.add_argument('-iters', default=2000000, type=int)

    # inference
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=22050, type=int)
    parser.add_argument('-test_batch_size', help='test audio size', default=4, type=int)
    parser.add_argument('-store_wav_path', default='./output/sample')
    args = parser.parse_args()

    # load config file
    with open(args.config) as f:
        config = yaml.load(f)

    if not os.path.exists(args.store_model_path):
        print("Creating directory: {}".format(args.store_model_path))
        os.makedirs(args.store_model_path)
    if not os.path.exists(args.store_wav_path):
        print("Creating directory: {}".format(args.store_wav_path))
        os.makedirs(args.store_wav_path)

    trainer = Trainer(config=config, args=args)
    trainer.cuda()
    if args.multigpus:
        ngpus = torch.cuda.device_count()
        print("Number of GPUs: %d" % ngpus)
        trainer.model = torch.nn.DataParallel(
            trainer.model, device_ids=range(ngpus))

    if args.load_model:
        trainer.iteration = trainer.load_model(args.multigpus)
    trainer.train_model()
