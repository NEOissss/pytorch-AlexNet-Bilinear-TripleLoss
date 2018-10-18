# -*- coding: utf-8 -*
"""
This module is served as torchvision.datasets to load SUN360 Half & Half Benchmark.
"""

import os
import csv
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as D


class Sun360Dataset(D.Dataset):
    def __init__(self, root, train=True, data='train', flip=True, version=0):
        super(Sun360Dataset, self).__init__()
        self.root = root  # '/mnt/nfs/scratch1/gluo/SUN360/HalfHalf/'
        self.pt_path = root + 'IMGs_pt/'
        self.train = train
        self.data = data  # 'train', 'test', 'val'
        self.flip = flip
        self.ver = version
        self.tf = transforms.Compose([transforms.Resize(227),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])

        self._param_check()
        self.gt = self._load_gt()
        self.len = len(self.gt)

        self.file_a = 'v{:d}_{:s}_a.pt'.format(self.ver, self.data)
        self.file_p = 'v{:d}_{:s}_a.pt'.format(self.ver, self.data)
        self.file_n = 'v{:d}_{:s}_a.pt'.format(self.ver, self.data)

        if not self._file_check():
            self.tensor_a, self.tensor_p, self.tensor_n = self._file_create()
        else:
            self.tensor_a = torch.load(self.pt_path + self.file_a)
            self.tensor_p = torch.load(self.pt_path + self.file_p)
            self.tensor_n = torch.load(self.pt_path + self.file_n)

    def __getitem__(self, batch):
        # Some data read from a file or image
        pass
        # return (a, p, n)

    def __len__(self):
        return self.len

    def _param_check(self):
        if self.data not in ['train', 'test']:
            raise ValueError('Unavailable dataset part!')
        if self.ver not in [0, 1, 2]:
            raise ValueError('Unavailable dataset version!')

    def _load_gt(self):
        gt_file = 'gt_' + self.data
        if self.ver == 0:
            gt_file += '.csv'
        else:
            gt_file += '_v{:d}.csv'.format(self.ver)
        with open(self.root + gt_file, 'r') as csv_file:
            gt_list = list(csv.reader(csv_file, delimiter=','))
        return gt_list

    def _file_check(self):
        if not os.path.exists(self.pt_path):
            os.mkdir(self.pt_path)
        return os.path.exists(self.pt_path + self.file_a) and os.path.exists(self.pt_path + self.file_p) and os.path.exists(self.pt_path + self.file_n)

    def _file_create(self):
        imgs_path = self.root + 'IMGs/'
        task_path = 'task_' + self.data

        if self.ver == 0:
            task_path += '/'
        else:
            task_path += '_v{:d}/'.format(self.ver)

        tensor_a = torch.zeros(self.len, 3, 227, 227)
        tensor_p = torch.zeros(self.len, 3, 227, 227)
        tensor_n = torch.zeros(self.len, 9, 3, 227, 227)

        for i in range(0, self.len):
            with open(self.root + task_path + self.gt[i][0] + '.json', 'r') as f:
                names = json.load(f)
                tensor_a[i, :, :, :] = self.tf(Image.open(imgs_path + names[0]))
                tensor_p[i, :, :, :] = self.tf(Image.open(imgs_path + names[1][int(self.gt[i][1])]))
                for j1, j2 in enumerate([k for k in range(10) if k != int(self.gt[i][1])]):
                    tensor_n[i, j1, :, :, :] = self.tf(Image.open(imgs_path + names[1][j2]))

        torch.save(tensor_a, self.pt_path + self.file_a)
        torch.save(tensor_n, self.pt_path + self.file_n)
        torch.save(tensor_p, self.pt_path + self.file_p)

        return tensor_a, tensor_p, tensor_n
