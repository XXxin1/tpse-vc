import yaml
from torch.utils.data import Dataset
import os
import json
from torch.utils.data import DataLoader
from utils import *
import random


def get_data_loader(dataset, batch_size, frame_size, shuffle=True, num_workers=4, drop_last=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True)
    return dataloader


class TrainData(Dataset):
    def __init__(self, npy_dir, sample_index_path, segment_size):
        with open(sample_index_path, 'r') as f:
            self.indexes = json.load(f)

        self.root = npy_dir
        self.segment_size = segment_size
        self.classes = sorted(list(set([path.split('_')[1] for path, _ in self.indexes])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        print('Data loader')
        print("\tNumber of classes: %d" % (len(self.classes)))

    def __getitem__(self, ind):
        utt_id, t = self.indexes[ind]
        label = self.class_to_idx[utt_id.split('_')[1]]
        data = np.load(os.path.join(self.root, utt_id))
        segment = data[:, t:t + self.segment_size]

        return segment, label

    def __len__(self):
        return len(self.indexes)


class TestData(Dataset):
    def __init__(self, npy_dir, sample_index_path):
        with open(sample_index_path, 'r') as f:
            self.indexes = json.load(f)

        self.root = npy_dir

    def __getitem__(self, ind):
        utt_id, t = self.indexes[ind]
        data = np.load(os.path.join(self.root, utt_id))

        return data, utt_id

    def __len__(self):
        return len(self.indexes)


if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.load(f)
    data_dir = '/files/xxx/VC/data/vcc2016_training_align_melgan_npy/'
    train_index_file = 'train_samples_100.json'
    train_parallel_dataset = TrainParallelData(os.path.join(data_dir),
                                                    os.path.join(data_dir,
                                                                 train_index_file),
                                                    segment_size=config['data_loader']['segment_size'])

    train_content_loader = get_data_loader(train_parallel_dataset,
                                           frame_size=config['data_loader']['frame_size'],
                                           batch_size=config['data_loader']['batch_size'],
                                           shuffle=config['data_loader']['shuffle'],
                                           num_workers=4, drop_last=False)
    train_content_iter = iter(train_content_loader)
    co_data = next(train_content_iter)
    # cl_data = next(train_content_iter)

    print(co_data[0].shape)
    print(co_data[1].shape)
    print(co_data[2].shape)
