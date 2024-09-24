## Library
import argparse
import argcomplete

from torch.backends import cudnn

from train import *

## Parser
parser = argparse.ArgumentParser(description="CycleGAN",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--n_gpu_use", default="0", type=str, dest="n_gpu_use")
parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument('--gan_mode', default='dcgan', choices=['wgan', 'dcgan'], type=str, dest='gan_mode')
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=300, type=int, dest="num_epoch")
parser.add_argument("--mask_start_epoch", default=20, type=int, dest="mask_start_epoch")
parser.add_argument("--num_worker", default=2, type=int, dest="num_worker")
parser.add_argument("--cls_start_epoch", default=5, type=int, dest="cls_start_epoch")
parser.add_argument("--save_interval", default=10, type=int, dest="save_interval")
parser.add_argument("--pretrain_c_path", type=str, default="",
                        help="Reload pretrained classifier", dest="pretrain_c_path")


parser.add_argument("--data_dir", default="./data/artifact_yn", type=str, dest="data_dir")
parser.add_argument("--experiment_name", default="unet", type=str, dest="experiment_name")
# parser.add_argument("--ckpt_dir", default="./checkpoint/unet", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log/unet", type=str, dest="log_dir")  # log를 어디에 사용?

parser.add_argument("--task", default="cyclegan", choices=['cyclegan'], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['direction', 0], dest='opts')

parser.add_argument("--ny", default=256, type=int, dest="ny")
parser.add_argument("--nx", default=256, type=int, dest="nx")
parser.add_argument("--nch", default=1, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")
parser.add_argument("--nblk", default=1, type=int, dest="nblk")
parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
parser.add_argument('--n_c', dest='n_c', type=int, default=5, help='# of c updates per d and g update')
parser.add_argument("--norm", default='inorm', type=str, dest="norm")

parser.add_argument("--use_mask", action='store_true', dest="use_mask")
parser.add_argument("--use_externalmask", action='store_true', dest="use_externalmask")

parser.add_argument("--wgt_cycle", default=10, type=float, dest="wgt_cycle")
parser.add_argument("--wgt_gan", default=1, type=float, dest="wgt_gan")
parser.add_argument("--wgt_fourier", default=1e-1, type=float, dest="wgt_fourier")
parser.add_argument("--wgt_ident", default=5e-1, type=float, dest="wgt_ident")
parser.add_argument("--wgt_cls", default=1, type=float, dest="wgt_cls")
parser.add_argument("--wgt_gp", default=10, type=float, dest="wgp_gp")

parser.add_argument("--network", default="CycleGAN", choices=['DCGAN', 'pix2pix', 'CycleGAN'], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")
parser.add_argument("--network_block", default="unet", choices=["resnet", "unet", "resunetplus", "resunetplus_v3"], type=str, dest="network_block")

# parse for windowing
parser.add_argument("--ww", default=300, type=int, dest="ww")
parser.add_argument("--wl", default=1000, type=int, dest="wl")

argcomplete.autocomplete(parser)
args = parser.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_gpu_use

    if args.mode == "train":
        cudnn.benchmark = True
        train(args)
    # elif args.mode == "test":
    #     test(args)


##

