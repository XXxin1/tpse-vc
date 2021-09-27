import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import copy


def segmenting_or_padding(audios):
    maxn = 128
    audio = []
    for item in audios:
        item_len = int(item.shape[1])
        assert item_len >= maxn
        if item_len > maxn:
            rand = np.random.randint(item_len - maxn)
            item_ = item[:, rand:rand + maxn]
        else:
            item_ = item
   
        audio += [item_]


    return torch.tensor((np.array(audio)))


def VCTK_collate(batch):
    src, tgt = zip(*batch)
    return segmenting_or_padding(src), segmenting_or_padding(tgt)


class VCTKDateset(Dataset):
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audios_source = os.listdir(self.audio_path)
        self.audios_target = copy.deepcopy(self.audios_source)
        random.shuffle(self.audios_source)
        random.shuffle(self.audios_target)

    def __getitem__(self, index):
        src = np.load(os.path.join(self.audio_path, self.audios_source[index]))
        tgt = np.load(os.path.join(self.audio_path, self.audios_target[index]))
        return src, tgt

    def __len__(self):
        return len(self.audios_source)


class VCTKDateset_name(Dataset):
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.audios_source = os.listdir(self.audio_path)
        self.audios_target = copy.deepcopy(self.audios_source)
        random.shuffle(self.audios_source)
        random.shuffle(self.audios_target)

    def __getitem__(self, index):
        src = np.load(os.path.join(self.audio_path, self.audios_source[index]))
        tgt = np.load(os.path.join(self.audio_path, self.audios_target[index]))
        return src, tgt, self.audios_source[index][:-4], self.audios_target[index][:-4]

    def __len__(self):
        return len(self.audios_source)


if __name__ == '__main__':
    # testing dataset
    dataset = VCTKDateset_name('/files/xxx/VC/VCTK/VCTK-Corpus/melspectrogram_vctk/test_unseen')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1,  pin_memory=True)
    data_iter = iter(dataloader)
    print('========')
    source, target, src_name, tgt_name = next(data_iter)
    print('========')
    print(source)
    print(source.shape)
    print(target.shape)
    print(src_name, tgt_name)
