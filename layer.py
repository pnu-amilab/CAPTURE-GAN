import os
import numpy as np

import torch
import torch.nn as nn
from switchable_norm import SwitchNorm1d, SwitchNorm2d


class DECBR2d(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True, norm="bnorm", relu=0.0,
                 affine=False, track_running_stats=False):
        super().__init__()

        layers = []
        # layers += [nn.ReflectionPad2d(padding=padding)]
        layers += [nn.ConvTranspose2d(in_channels=input_nc, out_channels=output_nc,
                                      kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                                      bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=output_nc)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=output_nc, affine=affine, track_running_stats=track_running_stats)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class BRC2d(nn.Module):  # CBR : Conv + BN + Relu
    def __init__(self, input_nc, output_nc, kernel_size=3, stride=1, padding=1, padding_mode='reflection', bias=True, norm="bnorm", relu=0.0, dilation=1):
        super().__init__()

        layers = []

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=input_nc)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=input_nc)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        if padding_mode == 'reflection':
            layers += [nn.ReflectionPad2d(padding)]
        elif padding_mode == 'replication':
            layers += [nn.ReplicationPad2d(padding)]
        elif padding_mode == 'constant':
            value = 0
            layers += [nn.ConstantPad2d(padding, value)]
        elif padding_mode == 'zeros':
            layers += [nn.ZeroPad2d(padding)]

        layers += [nn.Conv2d(in_channels=input_nc, out_channels=output_nc,
                             kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation,
                             bias=bias)]

        self.brc = nn.Sequential(*layers)

    def forward(self, x):
        return self.brc(x)


class BR2d(nn.Module):  # CBR : Conv + BN + Relu
    def __init__(self, input_nc, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=input_nc)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=input_nc)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.br = nn.Sequential(*layers)

    def forward(self, x):
        return self.br(x)


class CBR2d(nn.Module):  # CBR : Conv + BN + Relu
    def __init__(self, input_nc, output_nc, kernel_size=3, stride=1, padding=1, padding_mode='reflection', bias=True, norm="bnorm", relu=0.0,
                 affine=False, track_running_stats=False):
        super().__init__()

        layers = []

        if padding_mode == 'reflection':
            layers += [nn.ReflectionPad2d(padding)]
        elif padding_mode == 'replication':
            layers += [nn.ReplicationPad2d(padding)]
        elif padding_mode == 'constant':
            value = 0
            layers += [nn.ConstantPad2d(padding, value)]
        elif padding_mode == 'zeros':
            layers += [nn.ZeroPad2d(padding)]

        layers += [nn.Conv2d(in_channels=input_nc, out_channels=output_nc,
                             kernel_size=kernel_size, stride=stride, padding=0,
                             bias=bias)]
        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=output_nc)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=output_nc, affine=affine, track_running_stats=track_running_stats)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class ResBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super().__init__()

        layers = []

        # 1st conv
        layers += [CBR2d(input_nc=input_nc, output_nc=output_nc,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias, norm=norm, relu=relu)]

        # 2nd conv
        layers += [CBR2d(input_nc=output_nc, output_nc=output_nc,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias, norm=norm, relu=None)]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)


class PixelUnshuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C, H // ry, ry, W // rx, rx)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * (ry * rx), H // ry, W // rx)

        return x


class PixelShuffle(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()
        self.ry = ry
        self.rx = rx

    def forward(self, x):
        ry = self.ry
        rx = self.rx

        [B, C, H, W] = list(x.shape)

        x = x.reshape(B, C // (ry * rx), ry, rx, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // (ry * rx), H * ry, W * rx)

        return x


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn == 'none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=None, acti_fn=None):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn == 'none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=False, acti_fn=None):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn == 'none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'bnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'inorm':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers


def add_normalization_2d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'bnorm':
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == 'inorm':
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm2d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers


def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    elif fn == 'softmax':
        layers.append(nn.Softmax())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)