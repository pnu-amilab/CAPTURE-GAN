import os
import numpy as np
from scipy.stats import poisson
# from skimage.transform import rescale, resize
from skimage.filters import threshold_otsu
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torch.fft as fft

from torch.nn import functional as F

to_pil = transforms.ToPILImage()
to_gray = transforms.Grayscale(num_output_channels=1)


## Gradients Setting
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


## Weights Initialization
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':  # he-initialization
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



## Save and Load networks
def save(ckpt_dir, epoch, netG_a2b, netG_b2a, netD_a, netD_b, netC, optimG, optimD, standard, gc_only=False):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not standard:
        if gc_only:
            torch.save({'netG_a2b': netG_a2b.state_dict(), 'netG_b2a': netG_b2a.state_dict(),
                        'netC': netC.state_dict()},
                       "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

        else:
            torch.save({'netG_a2b': netG_a2b.state_dict(), 'netG_b2a': netG_b2a.state_dict(),
                        'netD_a': netD_a.state_dict(), 'netD_b': netD_b.state_dict(),
                        'netC': netC.state_dict(),
                        'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                       "%s/model_epoch%d.pth" % (ckpt_dir, epoch))
    else:
        if gc_only:
            torch.save({'netG_a2b': netG_a2b.state_dict(), 'netG_b2a': netG_b2a.state_dict(),
                        'netC': netC.state_dict()},
                       "%s/best_model_%s.pth" % (ckpt_dir, standard))
        else:
            torch.save({'netG_a2b': netG_a2b.state_dict(), 'netG_b2a': netG_b2a.state_dict(),
                        'netD_a': netD_a.state_dict(), 'netD_b': netD_b.state_dict(),
                        'netC': netC.state_dict(),
                        'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                       "%s/best_model_%s.pth" % (ckpt_dir, standard))


def load(ckpt_dir, netG_a2b, netG_b2a, netD_a, netD_b, netC, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG_a2b, netG_b2a, netD_a, netD_b, netC, optimG, optimD, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG_a2b.load_state_dict(dict_model['netG_a2b'])
    netG_b2a.load_state_dict(dict_model['netG_b2a'])
    netD_a.load_state_dict(dict_model['netD_a'])
    netD_b.load_state_dict(dict_model['netD_b'])
    netC.load_state_dict(dict_model['netC'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG_a2b, netG_b2a, netD_a, netD_b, netC, optimG, optimD, epoch

## Generate mask

class QueueMask():
    def __init__(self, length):
        self.max_length = length
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)

        self.queue.append(mask)

    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'

        # masks = self.queue[np.random.randint(0, self.queue.__len__())]
        # for _ in range(3):
        #     masks = torch.concat([masks, self.queue[np.random.randint(0, self.queue.__len__())]])

        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        # print(self.queue.__len__())
        # masks = self.queue[self.queue.__len__() - 4]
        # for i in range(3, 0, -1):
        #     masks = torch.concat([masks, self.queue[self.queue.__len__() - i]])
        return self.queue[self.queue.__len__() - 1]



def mask_generator(artifact, artifact_free, batch_size):
    thres_L = 2

    masks = torch.zeros_like(artifact).cuda()
    # gpu to cpu to gpu
    for i in range(batch_size):
        im_b = ((artifact + 1.0) * 0.5)[i]  # -1~1 to 0~1
        im_a = ((artifact_free + 1.0) * 0.5)[i]  # -1~1 to 0~1

        # im_a = artifact_free[i]
        # im_b = artifact[i]

        diff = im_b - im_a  # difference between shadow image and shadow_free image
        diff = torch.max(diff, torch.zeros_like(diff))
        # diff = (np.asarray(im_f, dtype='float32') - np.asarray(im_s, dtype='float32'))  # difference between shadow image and shadow_free image
        L = threshold_otsu(diff.detach().cpu().numpy()) * thres_L

        masks[i] = (diff >= L)

        masks.requires_grad = False

    return masks


def bone_masking(mask, bone_mask):
    masks = mask * bone_mask
    masks = (masks - 0.5) / 0.5

    masks.requires_grad = False
    return masks
