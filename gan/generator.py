import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # 画像サイズ
        self.img_shape = (opt.channels, opt.img_size, opt.img_size)
        # ノイズ次元
        z_dim = opt.latent_dim

        # ノイズ次元 100 => 128
        # 活性化関数：LeakyReLU
        self.layer1 = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(inplace=True)     # nn.LeakyReLU(0.2, inplace=True)
        )

        # 128 => 256
        # バッチノーマリゼーション:３次元なのでBatchNorm1d
        # 活性化関数：LeakyReLU
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256, 0.8),
            # nn.BatchNorm1d(256, 0.8),
            nn.ReLU(inplace=True)     # nn.LeakyReLU(0.2, inplace=True)
        )

        # 256 => 512
        # バッチノーマリゼーション
        # 活性化関数：LeakyReLU
        self.layer3 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.ReLU(inplace=True)     # nn.LeakyReLU(0.2, inplace=True)
        )

        # 512 => 1024
        # バッチノーマリゼーション
        # 活性化関数：LeakyReLU
        self.layer4 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.ReLU(inplace=True)     # nn.LeakyReLU(0.2, inplace=True)
        )

        # 1024 => チャンネル数 × 高さ × 横
        # 活性化関数：Tanh
        self.layer5 = nn.Sequential(
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )


    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), *self.img_shape)
        return out
        # #（バッチサイズ,チャンネル数,高さ,横）に変換
        # img = out.view(opt.batch_size, *img_shape)
        # return img


def main():
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    opt = parser.parse_args([])

    G = Generator(opt)

    # 入力する乱数 ( shape=[1,100] )
    input_z = torch.randn(1, 100)
    print('input_z.shape =', input_z.shape)
    # print(input_z)

    # 偽画像を作成
    G.eval()    # 推論モード
    fake_images = G(input_z)
    print('\nout_image type =', type(fake_images))
    print('out_image shape =', fake_images.shape)   # shape = [batch_size, channel_num, img_hight, img_width] (= [0, 1, 64, 64])
    # print('out_imge data =', fake_images)

    # detach=>numpy型に変換
    img_transformed = fake_images[0].detach().numpy()   # 作成画像のバッチ塊の最初だけ取り出し(fake_images[0]), 勾配情報を切り離し(detach()), numpyに変換
    print('\nimg_transformed shape =', img_transformed.shape)

    # チャンネル数、高さ、横 => 高さ、横
    img_transformed = np.squeeze(img_transformed)
    print('\nimg_transformed shape =', img_transformed.shape)

    # print(img_transformed, img_transformed.shape)


    plt.imshow(img_transformed, 'gray')
    plt.show()

if __name__ == "__main__":
    main()