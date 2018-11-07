"""
This module is served as torchvision.datasets to load SUN360 Half & Half Benchmark.
"""

import os
import sys
import time
import csv
import json
from random import randint
from PIL import Image
import torch
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
from torch.utils.data import Dataset, DataLoader


class Sun360Dataset(Dataset):
    def __init__(self, root, train=True, dataset='train', flip=False, version=0, cut=None, opt='pt'):
        super(Sun360Dataset, self).__init__()
        self.root = root
        self.train = train
        self.dataset = dataset  # 'train', 'test'
        self.flip = flip
        self.ver = version
        self.opt = opt

        self._param_check()
        self.tf = Compose([Resize(227), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.pt_path = '{:s}/IMGs_{:s}'.format(self.root, self.opt)
        self.data_path = '{:s}/{:s}_v{:d}'.format(self.pt_path, self.dataset, self.ver)
        self.data, self.gt = self._load_data()
        if cut:
            self.data = self.data[cut[0]: cut[1]]
            self.gt = self.gt[cut[0]: cut[1]]
        self.len = len(self.data)

        if self.opt == 'pt' and not self._file_check():
            self._file_create()

    def __getitem__(self, idx):
        file_path = '{:s}/{:s}.pt'.format(self.data_path, self.data[idx])
        tensor = torch.load(file_path)
        if self.train:
            return tensor[[0, 1, randint(2, 10)]]
        else:
            return tensor

    def __len__(self):
        return self.len

    def _param_check(self):
        if self.opt not in ['pt', 'fc7']:
            raise ValueError('Unavailable dataset option!')
        if self.dataset not in ['train', 'test']:
            raise ValueError('Unavailable dataset part!')
        if self.ver not in [0, 1, 2]:
            raise ValueError('Unavailable dataset version!')

    def _load_data(self):
        gt_file = 'gt_' + self.dataset
        if self.ver == 0:
            gt_file += '.csv'
        else:
            gt_file += '_v{:d}.csv'.format(self.ver)
        with open(self.root + gt_file, 'r') as csv_file:
            gt_list = list(csv.reader(csv_file, delimiter=','))
        data, gt = [], []
        for i, j in gt_list:
            data.append(i)
            gt.append(int(j))
        return data, gt

    def _file_check(self):
        if not os.path.exists(self.pt_path):
            os.mkdir(self.pt_path)
            os.mkdir(self.data_path)
            return False
        elif not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
            return False
        else:
            return True

    def _file_create(self):
        img_path = self.root + 'IMGs/'
        task_path = 'task_' + self.dataset

        if self.ver == 0:
            task_path += '/'
        else:
            task_path += '_v{:d}/'.format(self.ver)

        for i in range(0, self.len):
            tensor = torch.zeros(11, 3, 227, 227)
            file_path = '{:s}/{:s}.pt'.format(self.data_path, self.data[i])
            with open(self.root + task_path + self.data[i] + '.json', 'r') as f:
                names = json.load(f)
                tensor[0, :, :, :] = self.tf(Image.open(img_path + names[0]))
                tensor[1, :, :, :] = self.tf(Image.open(img_path + names[1][self.gt[i]]))
                for j1, j2 in enumerate([k for k in range(10) if k != self.gt[i]]):
                    tensor[2+j1, :, :, :] = self.tf(Image.open(img_path + names[1][j2]))
            torch.save(tensor, file_path)


def init_test(ver=0):
    root = '/mnt/nfs/scratch1/gluo/SUN360/HalfHalf/'

    start = time.time()
    a = Sun360Dataset(root, train=True, dataset='train', version=ver)
    print('Load train dataset in {:.2f} seconds, total length: {:d}'.format(time.time() - start, len(a)))

    start = time.time()
    a = Sun360Dataset(root, train=True, dataset='test', version=ver)
    print('Load test dataset in {:.2f} seconds, total length: {:d}'.format(time.time() - start, len(a)))

    start = time.time()
    a = Sun360Dataset(root, train=True, dataset='train', version=ver)
    print('Load train dataset in {:.2f} seconds, total length: {:d}'.format(time.time() - start, len(a)))

    start = time.time()
    a = Sun360Dataset(root, train=True, dataset='test', version=ver)
    print('Load test dataset in {:.2f} seconds, total length: {:d}'.format(time.time() - start, len(a)))


def test():
    root = '/mnt/nfs/scratch1/gluo/SUN360/HalfHalf'
    a = Sun360Dataset(root, train=True, dataset='train', flip=False, version=0, cut=[0, 10])
    b = DataLoader(dataset=a, batch_size=5)
    c = iter(b)
    for i in c:
        print(i.size())


if __name__ == '__main__':
    if len(sys.argv) == 2:
        init_test(ver=int(sys.argv[1]))
    else:
        init_test()
