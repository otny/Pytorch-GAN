import torch
import torch.nn as nn
import numpy as np
import generator

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        # 画像サイズ
        self.img_shape = (opt.channels, opt.img_size, opt.img_size) 
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)    # バッチで入ってくる画像 [batch, ] を, 1画像データ毎にflatにして [batch, hight*width] に変形
        validity = self.model(img_flat)
        return validity



def main():
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    opt = parser.parse_args([])

    G_model = generator.Generator(opt)
    D_model = Discriminator(opt)

    # Generator でノイズから画像生成
    G_model.eval()  # 推論モード
    input_z = torch.randn(10, 100)   # seedノイズ作成
    fake_images = G_model(input_z)    # ノイズから画像生成　[batch_size, channel_num, img_hight, img_width]
    print('\nout_img shape =', fake_images.shape)

    # Discriminatorで画像判別
    D_model.eval()  # 推論モード
    out = D_model(fake_images.detach())
    print('out =', out.shape, '\n', out)




if __name__ == "__main__":
    main()