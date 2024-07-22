from os.path import join

from torch.autograd import Variable

from model import *
from model_conditional import *
from dataset import *
from util import *
import json

import itertools
from datetime import datetime
from torchvision import transforms
import torchvision.utils as vutils

MEAN = 0
STD = 1.


def train(args):
    # training params
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    mask_start_epoch = args.mask_start_epoch
    save_interval = args.save_interval

    data_dir = args.data_dir
    experiment_name = args.experiment_name
    ckpt_dir = join('results', datetime.today().strftime("%Y%m%d"), experiment_name, 'checkpoints')
    sample_img_dir = join('results', datetime.today().strftime("%Y%m%d"), args.experiment_name, 'sample_training')
    cls_save_dir = join('results', datetime.today().strftime("%Y%m%d"), args.experiment_name, 'cls')

    os.makedirs(sample_img_dir, exist_ok=True)
    os.makedirs(cls_save_dir, exist_ok=True)

    with open(join('results', datetime.today().strftime("%Y%m%d"), experiment_name, 'setting.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

    log_dir = args.log_dir
    num_worker = args.num_worker

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker
    nblk = args.nblk
    norm = args.norm

    wgt_cycle = args.wgt_cycle
    wgt_gan = args.wgt_gan
    wgt_ident = args.wgt_ident
    wgt_cls = args.wgt_cls

    lambda_cls = args.lambda_cls

    network = args.network
    learning_type = args.learning_type
    network_block = args.network_block

    use_mask = args.use_mask
    use_externalmask = args.use_externalmask
    pretrain_c_first = args.pretrain_c_first
    pretrain_c_second = args.pretrain_c_second
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("norm: %s" % norm)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)
    print("mask start epoch: %d" % mask_start_epoch)
    print("number of channels: %d" % nch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("network block: %s" % network_block)
    print("learning type: %s" % learning_type)

    print("use mask: %r" % use_mask)
    print("use external mask: %r" % use_externalmask)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)

    print("device: %s" % device)

    # writer = SummaryWriter(join('results', args.experiment_name, 'summary'))
    ## Training networks
    if mode == 'train':
        transform_train = transforms.Compose([Resize(shape=(286, 286, nch)),
                                              RandomCrop((ny, nx)),
                                              Normalization(mean=MEAN, std=STD)])

        transform_train_c = transforms.Compose([Normalization(mean=MEAN, std=STD)])

        loader_train_ac = get_loader(data_dir, transform_train, 'AC',
                                    use_mask, batch_size, num_worker, 'train')
        loader_train_bd = get_loader(data_dir, transform_train, 'BD',
                                    use_mask, batch_size, num_worker, 'train')

        loader_valid_ac = get_loader(data_dir, transform_train_c, 'AC',
                                    use_mask, batch_size, num_worker, 'valid')
        loader_valid_bd = get_loader(data_dir, transform_train_c, 'BD',
                                    use_mask, batch_size, num_worker, 'valid')

        # Generate and list mask
        Tensor = torch.cuda.FloatTensor

        if use_mask:
            mask_non_shadow = Variable(Tensor(args.batch_size, 1, 256, 256).fill_(-1.0),
                                       requires_grad=False)  # non-shadow

            mask_queue = QueueMask(min(loader_train_ac.__len__(), loader_train_bd.__len__()) / 4)

    ## Build network
    if network == "CycleGAN":
        # a2b: artifact-free to artifact
        netG_a2b = CycleGAN(input_nc=nch, output_nc=nch, nker=nker, norm=norm, nblk=nblk, learning_type=learning_type, network_block=network_block, use_mask=use_mask).to(device)
        # b2a: artifact to artifact-free
        netG_b2a = CycleGAN(input_nc=nch, output_nc=nch, nker=nker, norm=norm, nblk=nblk, learning_type=learning_type, network_block=network_block, use_mask=False).to(device)

        netD_a = Discriminator_cycle(input_nc=nch, output_nc=1, nker=nker, norm=norm).to(device)
        netD_b = Discriminator_cycle(input_nc=nch, output_nc=1, nker=nker, norm=norm).to(device)

        netC = Classifier(pretrain_c_first).to(device)

        init_weights(netG_a2b, init_type='kaiming', init_gain=0.02)
        init_weights(netG_b2a, init_type='kaiming', init_gain=0.02)

        init_weights(netD_a, init_type='kaiming', init_gain=0.02)
        init_weights(netD_b, init_type='kaiming', init_gain=0.02)

        optimG = torch.optim.Adam(itertools.chain(netG_a2b.parameters(), netG_b2a.parameters()), lr=lr, betas=(0.5, 0.999))
        optimD = torch.optim.Adam(itertools.chain(netD_a.parameters(), netD_b.parameters()), lr=lr, betas=(0.5, 0.999))

    ## Loss functions
    fn_cycle = nn.L1Loss()
    fn_ident = nn.L1Loss()
    fn_gan = nn.BCELoss()
    fn_cls = nn.CrossEntropyLoss()

    ## Tensorboard SummaryWriter
    # writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

    ## 네트워크 학습시키기
    st_epoch = 0

    # TRAIN MODE
    if network == "CycleGAN":

        if train_continue == "on":

            netG_a2b, netG_b2a, \
            netD_a, netD_b, netC, \
            optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                            netG_a2b=netG_a2b, netG_b2a=netG_b2a,
                                            netD_a=netD_a, netD_b=netD_b, netC=netC,
                                            optimG=optimG, optimD=optimD)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimC, T_max=30)
        netC.load_state_dict(torch.load(pretrain_c_first))

        for epoch in range(st_epoch + 1, num_epoch + 1):

            netG_a2b.train()
            netG_b2a.train()
            netD_a.train()
            netD_b.train()

            for batch, data in enumerate(zip(loader_train_ac, loader_train_bd), 1):
                data_a, data_b = data

                real_a = data_a['data'].to(device)
                real_b = data_b['data'].to(device)
                att_edema_a = data_a['att_edema'].to(device)
                att_edema_b = data_b['att_edema'].to(device)

                if use_mask:
                    bone_mask_a = data_a['mask'].to(device)
                    bone_mask_b = data_b['mask'].to(device)

                #### forward netG ####
                # GAN loss
                set_requires_grad([netG_a2b, netG_b2a], True)
                fake_a = netG_b2a(real_b, None)

                if use_mask:
                    if epoch < mask_start_epoch:
                        fake_b = netG_a2b(real_a, mask_non_shadow)

                    else:
                        mask_ba = mask_generator(real_b, fake_a, batch_size)
                        mask_queue.insert(mask_ba)

                        if use_externalmask:
                            fake_b = netG_a2b(real_a, bone_masking(mask_queue.rand_item(), bone_mask_a))
                        else:
                            mask = (mask_queue.rand_item() - 0.5) / 0.5
                            fake_b = netG_a2b(real_a, mask)
                else:
                    fake_b = netG_a2b(real_a, None)

                if use_mask:
                    if epoch < mask_start_epoch:
                        recon_b = netG_a2b(fake_a, mask_non_shadow)
                    else:
                        if use_externalmask:
                            recon_b = netG_a2b(fake_a, bone_masking(mask_queue.last_item(), bone_mask_b))
                        else:
                            mask = (mask_queue.last_item() - 0.5) / 0.5
                            recon_b = netG_a2b(fake_a, mask)
                else:
                    recon_b = netG_a2b(fake_a, None)

                recon_a = netG_b2a(fake_b, None)

                # backward netD
                set_requires_grad([netD_a, netD_b], True)
                optimD.zero_grad()

                # Valina GAN

                # backward netD_a
                pred_real_a = netD_a(real_a)
                pred_fake_a = netD_a(fake_a.detach())

                loss_D_a_real = fn_gan(pred_real_a, torch.ones_like(pred_real_a))
                loss_D_a_fake = fn_gan(pred_fake_a, torch.zeros_like(pred_fake_a))

                loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

                # backward netD_b
                pred_real_b = netD_b(real_b)
                pred_fake_b = netD_b(fake_b.detach())

                loss_D_b_real = fn_gan(pred_real_b, torch.ones_like(pred_real_b))
                loss_D_b_fake = fn_gan(pred_fake_b, torch.zeros_like(pred_fake_b))

                loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)

                loss_D = loss_D_a + loss_D_b

                loss_D.backward()
                optimD.step()

                # backward netG
                set_requires_grad([netD_a, netD_b, netC], False)
                optimG.zero_grad()

                pred_fake_a = netD_a(fake_a)
                pred_fake_b = netD_b(fake_b)

                # Vanila GAN
                loss_G_a2b = fn_gan(pred_fake_a, torch.ones_like(pred_fake_a))
                loss_G_b2a = fn_gan(pred_fake_b, torch.ones_like(pred_fake_b))

                loss_cycle_a = fn_cycle(real_a, recon_a)
                loss_cycle_b = fn_cycle(real_b, recon_b)

                ident_a = netG_b2a(real_a, None)
                # ident_a = netG_b2a(real_a)

                if use_mask:
                    ident_b = netG_a2b(real_b, mask_non_shadow)
                else:
                    ident_b = netG_a2b(real_b, None)

                loss_ident_a = fn_ident(real_a, ident_a)
                loss_ident_b = fn_ident(real_b, ident_b)

                fake_a_a_3 = ident_a.expand(ident_a.shape[0], 3, *ident_a.shape[2:])  # a to a
                fake_b_a_3 = fake_a.expand(fake_a.shape[0], 3, *fake_a.shape[2:])  # b to a

                pred_fake_a_a_cls = netC(fake_a_a_3)
                pred_fake_b_a_cls = netC(fake_b_a_3)

                loss_cls_a_a = fn_cls(pred_fake_a_a_cls, att_edema_a)
                loss_cls_b_a = fn_cls(pred_fake_b_a_cls, att_edema_b)

                loss_cls_g = (loss_cls_a_a + loss_cls_b_a)

                loss_G = wgt_gan * (loss_G_a2b + loss_G_b2a) + \
                         wgt_cycle * (loss_cycle_a + loss_cycle_b) + \
                         wgt_cycle * wgt_ident * (loss_ident_a + loss_ident_b) + \
                         wgt_cls * loss_cls_g

                loss_G.requires_grad_(True)
                loss_G.backward()
                optimG.step()

            if epoch % save_interval == 0:
                save(ckpt_dir=ckpt_dir, epoch=epoch,
                     netG_a2b=netG_a2b, netG_b2a=netG_b2a,
                     netD_a=netD_a, netD_b=netD_b,
                     netC=netC, optimG=optimG, optimD=optimD,
                     standard=None, gc_only=True)

                samples = [real_b, fake_a, recon_b]
                samples = torch.cat(samples, dim=3)
                vutils.save_image(samples, os.path.join(
                    sample_img_dir,
                    'Epoch_({:d}).jpg'.format(epoch)
                ), nrow=1, normalize=True, value_range=(-1., 1.))





