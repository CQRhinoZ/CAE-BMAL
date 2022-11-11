from datetime import datetime
import os
import os.path as osp

# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer2

# Custom includes
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from networks.GAN import BoundaryDiscriminator, UncertaintyDiscriminator

here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')

    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument(
        '--datasetS', type=str, default='Domain4', help='test folder id contain images ROIs to test'
    )
    # sjq add
    parser.add_argument(
        '--redataset', type=str, default='wae_img/Domain4', help='generated image from domain'
    )
    # target sjq
    parser.add_argument(
        '--datasetT', type=str, default='Domain1', help='refuge / Drishti-GS/ RIM-ONE_r3'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8, help='batch size for training the model'
    )
    parser.add_argument(
        '--group-num', type=int, default=1, help='group number for group normalization'
    )
    parser.add_argument(
        '--max-epoch', type=int, default=200, help='max epoch'
    )
    parser.add_argument(
        '--stop-epoch', type=int, default=200, help='stop epoch'
    )
    parser.add_argument(
        '--warmup-epoch', type=int, default=-1, help='warmup epoch begin train GAN'
    )

    parser.add_argument(
        '--interval-validate', type=int, default=10, help='interval epoch number to valide the model'
    )
    parser.add_argument(
        '--lr-gen', type=float, default=1e-3, help='learning rate',
    )
    parser.add_argument(
        '--lr-dis', type=float, default=2.5e-5, help='learning rate',
    )
    parser.add_argument(
        '--lr-decrease-rate', type=float, default=0.1, help='ratio multiplied to initial lr',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--data-dir',
        default='/ai/sjq/dataset/Fundus/',
        help='data root path'
    )

    # sjq add
    parser.add_argument(
        '--re_base_dir',
        default='/ai/sjq/mylab/VAE',
        help='data root path'
    )

    parser.add_argument(
        '--out-stride',
        type=int,
        default=16,
        help='out-stride of deeplabv3+',
    )
    parser.add_argument(
        '--sync-bn',
        type=bool,
        default=True,
        help='sync-bn in deeplabv3+',
    )
    parser.add_argument(
        '--freeze-bn',
        type=bool,
        default=False,
        help='freeze batch normalization of deeplabv3+',
    )

    args = parser.parse_args()

    args.model = 'FCN8s'

    now = datetime.now()
    # target sjq change: T->S
    # args.out = osp.join(here, 'logs', args.datasetT, now.strftime('%Y%m%d_%H%M%S.%f'))
    args.out = osp.join(here, 'logs', args.datasetS, now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    #<This part will be released when the paper is published>

    # target sjq
    #<This part will be released when the paper is published>

    # 2. model
    #<This part will be released when the paper is published>

    # 3. optimizer

    #<This part will be released when the paper is published>


if __name__ == '__main__':
    main()