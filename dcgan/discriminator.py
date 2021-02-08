import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        # 画像サイズ取得
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)

        self.layer = nn.Sequential(
            # input is (opt.channels) x 64 x 64
            nn.Conv2d(opt.channels, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (opt.ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input):
        # out = self.layer1(input)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.layer5(out)
        return self.layer(input)



def main():
    import matplotlib.pyplot as plt
    import argparse
    from generator import Generator
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--ndf", type=int, default=64, help="relates to the depth of feature maps carried through the generator")
    parser.add_argument("--ngf", type=int, default=64, help="relates to the depth of feature maps carried through the generator")
    opt = parser.parse_args([])

    G = Generator(opt)
    D = Discriminator(opt)
    noise = torch.randn(opt.batch_size, opt.latent_dim, 1, 1)
    G_out = G(noise)
    D_out = D(G_out).view(-1)
    print(D)
    print(D_out.shape)
    



if __name__ == "__main__":
    main()