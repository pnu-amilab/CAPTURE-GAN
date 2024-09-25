import os
import numpy as np

import torch
import torch.nn as nn
import sys

from layer import *
from core.res_unet_plus import *

## Build Networks
# CycleGAN
# https://arxiv.org/pdf/1703.10593.pdf


class CycleGAN(nn.Module):
    def __init__(self, input_nc, output_nc, nker=64, norm='bnorm', nblk=6, learning_type='plain', network_block='unet', use_mask=False):
        super(CycleGAN, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nker = nker
        self.norm = norm
        self.nblk = nblk
        self.learning_type = learning_type
        self.network_block = network_block
        self.use_mask = use_mask

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        if use_mask:
            self.input_nc += 1

        self.enc1 = CBR2d(self.input_nc, 1 * self.nker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
        self.enc2 = CBR2d(1 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc3 = CBR2d(2 * self.nker, 4 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)

        self.dec3 = DECBR2d(4 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm,
                            relu=0.0)
        self.dec2 = DECBR2d(2 * self.nker, 1 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm,
                            relu=0.0)
        self.dec1 = CBR2d(1 * self.nker, self.output_nc, kernel_size=7, stride=1, padding=3, norm=None,
                          relu=None)
        res = []

        if self.nblk:
            if self.network_block == 'resnet':
                for i in range(self.nblk):
                    res += [ResBlock(4 * self.nker, 4 * self.nker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0)]
            elif self.network_block == 'unet':
                for i in range(self.nblk):
                    res += [UNet(self.input_nc, self.output_nc, nker=self.nker, learning_type=self.learning_type, norm=self.norm)]
            elif self.network_block == 'resunetplus':
                for i in range(self.nblk):
                    res += [ResUnetPlusPlus(self.input_nc)]
            elif self.network_block == 'resunetplus_v3':
                for i in range(self.nblk):
                    res += [ResUnetPlusPlusV3(self.input_nc, self.output_nc, norm=self.norm)]

        self.res = nn.Sequential(*res)
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        if self.network_block == 'resnet':
            if self.use_mask:
                x = self.enc1(torch.cat((x, mask), 1))
            else:
                x = self.enc1(x)
            x = self.enc2(x)
            x = self.enc3(x)

            x = self.res(x)

            x = self.dec3(x)
            x = self.dec2(x)
            x = self.dec1(x)

        else:
            if self.use_mask:
                x = self.res(torch.cat((x, mask), 1))
            else:
                x = self.res(x)

        return self.tanh(x)

class Discriminator_cycle(nn.Module):
    def __init__(self, input_nc, output_nc, nker=64, norm="bnorm"):
        super(Discriminator_cycle, self).__init__()

        self.enc1 = CBR2d(1 * input_nc, 1 * nker, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=0.2, bias=False)

        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc5 = CBR2d(8 * nker, output_nc, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=None, bias=False)

        self.fc_adv = nn.Sequential(
            LinearBlock(16 * nker * 8 * 8, 1024, 'none', 'relu'),
            LinearBlock(1024, 1, 'none', 'sigmoid')
        )

    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        return torch.sigmoid(x)


# U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/abs/1505.04597
class UNet(nn.Module):
    def __init__(self, input_nc, output_nc, nker=64, learning_type="plain", norm="bnorm"):
        super(UNet, self).__init__()

        self.learning_type = learning_type

        # Contracting path
        self.enc1_1 = CBR2d(input_nc=input_nc, output_nc=1 * nker, norm=norm)
        self.enc1_2 = CBR2d(input_nc=1 * nker, output_nc=1 * nker, norm=norm)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(input_nc=nker, output_nc=2 * nker, norm=norm)
        self.enc2_2 = CBR2d(input_nc=2 * nker, output_nc=2 * nker, norm=norm)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(input_nc=2 * nker, output_nc=4 * nker, norm=norm)
        self.enc3_2 = CBR2d(input_nc=4 * nker, output_nc=4 * nker, norm=norm)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(input_nc=4 * nker, output_nc=8 * nker, norm=norm)
        self.enc4_2 = CBR2d(input_nc=8 * nker, output_nc=8 * nker, norm=norm)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(input_nc=8 * nker, output_nc=16 * nker, norm=norm)

        # Expansive path
        self.dec5_1 = CBR2d(input_nc=16 * nker, output_nc=8 * nker, norm=norm)

        self.unpool4 = nn.ConvTranspose2d(in_channels=8 * nker, out_channels=8 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(input_nc=2 * 8 * nker, output_nc=8 * nker, norm=norm)
        self.dec4_1 = CBR2d(input_nc=8 * nker, output_nc=4 * nker, norm=norm)

        self.unpool3 = nn.ConvTranspose2d(in_channels=4 * nker, out_channels=4 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(input_nc=2 * 4 * nker, output_nc=4 * nker, norm=norm)
        self.dec3_1 = CBR2d(input_nc=4 * nker, output_nc=2 * nker, norm=norm)

        self.unpool2 = nn.ConvTranspose2d(in_channels=2 * nker, out_channels=2 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(input_nc=2 * 2 * nker, output_nc=2 * nker, norm=norm)
        self.dec2_1 = CBR2d(input_nc=2 * nker, output_nc=1 * nker, norm=norm)

        self.unpool1 = nn.ConvTranspose2d(in_channels=1 * nker, out_channels=1 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(input_nc=2 * 1 * nker, output_nc=1 * nker, norm=norm)
        self.dec1_1 = CBR2d(input_nc=1 * nker, output_nc=1 * nker, norm=norm)

        self.fc = nn.Conv2d(in_channels=1 * nker, out_channels=output_nc, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        if self.learning_type == "plain":
            x = self.fc(dec1_1)
        elif self.learning_type == "residual":
            x = x + self.fc(dec1_1)

        return x


# Deep Residual Learning for Image Recognition
# https://arxiv.org/abs/1512.03385
class ResNet(nn.Module):
    def __init__(self, input_nc, output_nc, nker=64, learning_type="plain", norm="bnorm", nblk=16):
        super(ResNet, self).__init__()

        self.learning_type = learning_type

        self.enc = CBR2d(input_nc, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=None, relu=0.0)

        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]
        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)

        self.fc = CBR2d(nker, output_nc, kernel_size=1, stride=1, padding=0, bias=True, norm=None, relu=None)

    def forward(self, x):
        x0 = x

        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)

        if self.learning_type == "plain":
            x = self.fc(x)
        elif self.learning_type == "residual":
            x = x0 + self.fc(x)

        return x