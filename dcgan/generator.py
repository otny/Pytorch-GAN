import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.layer = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( opt.latent_dim, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (opt.ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (opt.ngf*4) x 8 x 8
            nn.ConvTranspose2d( opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # state size. (opt.ngf*2) x 16 x 16
            nn.ConvTranspose2d( opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (opt.ngf) x 32 x 32
            nn.ConvTranspose2d( opt.ngf, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.layer(input)


def main():
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--ngf", type=int, default=64, help="relates to the depth of feature maps carried through the generator")
    opt = parser.parse_args([])

    G = Generator(opt)
    noise = torch.randn(opt.batch_size, opt.latent_dim, 1, 1)
    out = G(noise)
    print(G)
    print(noise.shape)
    print(out.shape)

if __name__ == "__main__":
    main()