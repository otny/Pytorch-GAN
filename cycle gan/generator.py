###################
# Generatorは9ブロックのResNet
# BatchNormalizationではなく、InstanceNormalizationを使う
# DCGANなどではノイズから画像を生成していたが、CycleGANでは別ドメインの画像から画像を生成する
# 入力が1次元ノイズではなくて2次元のカラー画像になっている  
###################

import functools
import torch
import torch.nn as nn
from pytorch_memlab import profile

class ResNetBlock(nn.Module):

    def __init__(self, dim, norm_layer, conv_bias):
        super(ResNetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, bias=conv_bias),
                       norm_layer(dim),
                       nn.ReLU(True),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, bias=conv_bias),
                       norm_layer(dim)]
        self.conv_block = nn.Sequential(*conv_block)


    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Identity(nn.Module):
    def forward(self, x):
        return x



def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        norm_layer = get_norm_layer(norm_type=opt.norm)

        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),

            nn.Conv2d(opt.channels, opt.ngf, kernel_size=7, bias=opt.conv_bias),
            norm_layer(opt.ngf),
            nn.ReLU(True),

            nn.Conv2d(opt.ngf, opt.ngf*2, kernel_size=3, stride=2, padding=1, bias=opt.conv_bias),
            norm_layer(opt.ngf*2),
            nn.ReLU(True),

            nn.Conv2d(opt.ngf*2, opt.ngf*4, kernel_size=3, stride=2, padding=1, bias=opt.conv_bias),
            norm_layer(opt.ngf*4),
            nn.ReLU(True),

            ResNetBlock(opt.ngf*4, norm_layer, opt.conv_bias),
            ResNetBlock(opt.ngf*4, norm_layer, opt.conv_bias),
            ResNetBlock(opt.ngf*4, norm_layer, opt.conv_bias),
            ResNetBlock(opt.ngf*4, norm_layer, opt.conv_bias),
            ResNetBlock(opt.ngf*4, norm_layer, opt.conv_bias),
            ResNetBlock(opt.ngf*4, norm_layer, opt.conv_bias),
            ResNetBlock(opt.ngf*4, norm_layer, opt.conv_bias),
            ResNetBlock(opt.ngf*4, norm_layer, opt.conv_bias),
            ResNetBlock(opt.ngf*4, norm_layer, opt.conv_bias),

            nn.ConvTranspose2d(opt.ngf*4, opt.ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=opt.conv_bias),
            norm_layer(opt.ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(opt.ngf*2, opt.ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=opt.conv_bias),
            norm_layer(opt.ngf),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(opt.ngf, opt.channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

        # initialize weights
        self.model.apply(self._init_weights)

    # @profile
    def forward(self, input):
        return self.model(input)


    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def main():
    from load_datasets import UnalignedDataset, get_loader
    import matplotlib.pyplot as plt
    import argparse

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--ngf', type=int, default=64, help='relates to the depth of feature maps carried through the generator')
    parser.add_argument('--data_path', type=str, default='F:/datasets/CycleGAN', help='Path of the folder where Mnist is located')
    parser.add_argument('--load_size', type=int, default=286, help='Original image resized to this size')
    parser.add_argument('--fine_size', type=int, default=256, help='Crop a random this size from a load size image')
    parser.add_argument('--batch_size', type=int, default=1, help='Size of the batches')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of cpu cores used in the DataLoader, Recommended: 2')
    opt = parser.parse_args([])

    G = Generator(opt).to(device)
    print(G)

    train_dataset = UnalignedDataset(is_train=True, opt=opt)
    train_loader = get_loader(train_dataset, opt)
    load_img = iter(train_loader).next()
    img_A = load_img['A'].to(device)
    out_img = G(img_A)



if __name__ == '__main__':
    main()