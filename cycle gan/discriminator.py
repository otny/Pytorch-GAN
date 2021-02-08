###################
# Discriminatorは普通のCNN
# DCGANのように0 or 1をスカラーで返さないで30x30のfeature mapを出力する
###################

import functools
import torch
import torch.nn as nn
from pytorch_memlab import profile
from torch.nn.utils.spectral_norm import spectral_norm

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



class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        norm_layer = get_norm_layer(norm_type=opt.norm)

        self.model = nn.Sequential(
            nn.Conv2d(opt.channels, opt.ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(opt.ndf, opt.ndf*2, kernel_size=4, stride=2, padding=1, bias=opt.conv_bias),
            norm_layer(opt.ndf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(opt.ndf*2, opt.ndf*4, kernel_size=4, stride=2, padding=1, bias=opt.conv_bias),
            norm_layer(opt.ndf*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(opt.ndf*4, opt.ndf*8, kernel_size=4, stride=1, padding=1, bias=opt.conv_bias),
            norm_layer(opt.ndf*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(opt.ndf*8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
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
    import matplotlib.pyplot as plt
    import argparse
    from load_datasets import UnalignedDataset, get_loader
    from generator import Generator
    from others import imshow
    from ImagePool import ImagePool
    from torchvision.utils import make_grid

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--ngf', type=int, default=64, help='relates to the depth of feature maps carried through the generator')
    parser.add_argument('--ndf', type=int, default=64, help='relates to the depth of feature maps carried through the discriminator')
    parser.add_argument('--data_path', type=str, default='F:/datasets/CycleGAN', help='Path of the folder where Mnist is located')
    parser.add_argument('--load_size', type=int, default=286, help='Original image resized to this size')
    parser.add_argument('--fine_size', type=int, default=256, help='Crop a random this size from a load size image')
    parser.add_argument('--batch_size', type=int, default=1, help='Size of the batches')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of cpu cores used in the DataLoader, Recommended: 2')
    opt = parser.parse_args([])

    G = Generator(opt).to(device)
    D = Discriminator(opt).to(device)


    # reporterG = MemReporter(G)
    # reporterD = MemReporter(D)
    # print(G)
    # print(D)
    train_dataset = UnalignedDataset(is_train=True, opt=opt)
    train_loader = get_loader(train_dataset, opt)    
    load_img = iter(train_loader).next()
    img_A = load_img['A'].to(device)

    # reporterG.report()
    # reporterD.report()
    out_img = G(img_A)
    # with LineProfiler(G.forward, D.forward) as prof:
    D_out_real = D(img_A)
    D_out_fake = D(out_img)
    # reporterG.report()
    # reporterD.report()
    # prof.display()

    # print('out_img.shape =', out_img.shape)
    # print('D_out_real.shape =', D_out_real.shape)
    # print('D_out_fake.shape =', D_out_fake.shape)


    ### --- Plot --- ###
    PLOT = False
    if PLOT:
        plt.figure(figsize=(10, 20))
        # print(type(img_A.to('cpu')))

        plt.subplot(1, 2, 1)
        plt.imshow(imshow(make_grid(img_A.to('cpu'), nrow=4)))
        plt.title('Input_REAL_img_A')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(imshow(make_grid(out_img.detach().to('cpu'), nrow=4)))
        plt.title('Output_FAKE_img_AtoB')
        plt.axis('off')
        plt.show()


    # ImagePool Class
    fake_A_pool = ImagePool(1)

    # print('\nimg_A[0][0][0][0] =', img_A[0][0][0][0])
    # print('img_A.shape =', img_A.shape)
    img_A = fake_A_pool.query(img_A)
    # print('img_A[0][0][0][0] =', img_A[0][0][0][0])
    # print('img_A.shape =', img_A.shape)
    img_A = fake_A_pool.query(out_img.detach())
    # print('img_A[0][0][0][0] =', img_A[0][0][0][0])
    # print('img_A.shape =', img_A.shape)
    



if __name__ == '__main__':
    main()