import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

import matplotlib.pyplot as plt
from util import *
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mask_dir, transform=None, data_conf='A', use_mask=True):
        self.data_conf = data_conf
        self.data_dir = data_dir + data_conf
        self.mask_dir = mask_dir + data_conf

        self.transform = transform
        self.use_mask = use_mask

        self.to_tensor = ToTensor()

        if os.path.exists(self.data_dir):
            lst_data = os.listdir(self.data_dir)
            lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]
            lst_data.sort()

            lst_mask = os.listdir(self.mask_dir)
            lst_mask = [f for f in lst_mask if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]
            lst_mask.sort()
        else:
            lst_data = []
            lst_mask = []

        self.lst_data = lst_data
        self.lst_mask = lst_mask

    def __len__(self):
        return len(self.lst_data)

    # 1 channel
    def __getitem__(self, index):

        data = {}

        input = cv2.imread(os.path.join(self.data_dir, self.lst_data[index]), -1)
        data['data'] = input

        if self.use_mask:
            mask = cv2.imread(os.path.join(self.mask_dir, self.lst_mask[index]), -1)
            data['mask'] = mask

        if self.transform:
            data = self.transform(data)

        if self.data_conf == 'A' or self.data_conf == 'B':
            data['att_edema'] = np.array(0)
        elif self.data_conf == 'C' or self.data_conf == 'D':
            data['att_edema'] = np.array(1)

        if self.data_conf == 'A' or self.data_conf == 'C':
            data['att_artifact'] = np.array(0)
        elif self.data_conf == 'B' or self.data_conf == 'D':
            data['att_artifact'] = np.array(1)

        data = self.to_tensor(data)

        return data

## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            if key.startswith('att'):
                data[key] = torch.from_numpy(value)
            else:
                value = value[:, :, np.newaxis]
                value = value.transpose((2, 0, 1)).astype(np.float32)
                data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0, std=1., v_min=850, v_max=1150):
        self.mean = mean
        self.std = std
        self.v_min = v_min
        self.v_max = v_max

    def __call__(self, data):
        for key, value in data.items():
            if key.startswith('data'):
                value = np.clip(value, self.v_min, self.v_max)
                value = (value - self.v_min) / (self.v_max - self.v_min)
                value = (value * 2) - 1
                # data[key] = (value - self.mean) / self.std
                data[key] = value
            else:
                data[key] = value
        return data


class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        keys = list(data.keys())

        h, w = data[keys[0]].shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            if key.startswith('att'):
                data[key] = value
            else:
                data[key] = value[id_y, id_x]

        return data


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            if key.startswith('att'):
                data[key] = value
            else:
                data[key] = cv2.resize(value, (self.shape[0], self.shape[1]))

        return data


def get_loader(data_dir, transform, data_conf, use_mask, batch_size, num_workers, type):
    if len(data_conf) == 1:
        if type == 'train':
            dataset = Dataset(os.path.join(data_dir, 'train/train'),
                              os.path.join(data_dir, 'train/mask'),
                              transform, data_conf, use_mask)
        elif type == 'valid':
            dataset = Dataset(os.path.join(data_dir, 'valid/train'),
                              os.path.join(data_dir, 'valid/mask'),
                              transform, data_conf, use_mask)

    elif len(data_conf) == 2:
        if type == 'train':
            dataset_a = Dataset(os.path.join(data_dir, 'train/train'),
                              os.path.join(data_dir, 'train/mask'),
                              transform, data_conf[0], use_mask)
            dataset_b = Dataset(os.path.join(data_dir, 'train/train'),
                                os.path.join(data_dir, 'train/mask'),
                                transform, data_conf[1], use_mask)
            dataset = ConcatDataset([dataset_a, dataset_b])

        elif type == 'valid':
            dataset_a = Dataset(os.path.join(data_dir, 'valid/train'),
                              os.path.join(data_dir, 'valid/mask'),
                              transform, data_conf[0], use_mask)
            dataset_b = Dataset(os.path.join(data_dir, 'valid/train'),
                              os.path.join(data_dir, 'valid/mask'),
                              transform, data_conf[1], use_mask)
            dataset = ConcatDataset([dataset_a, dataset_b])

    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    return loader
