import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image, make_grid
import numpy as np

# custom weights initialization called on netG and netD
# pytorch のチュートリアルにあった関数コピペ。なんか重みを良い感じに初期化してくれる見たい
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_loss(result, opt):
    fig = plt.figure()  # グラフの描画先の準備
    x = list(range(1, len(result["log_loss_G"]) + 1 )) # 0 ~ LastEpochまで順番に生成
    y = result["log_loss_G"]
    plt.plot(x, y, label="Generator loss")
    y = result["log_loss_D"]
    plt.plot(x, y, label="Discriminator loss")
    if opt.plot_epoch:
        plt.xlabel("Epoch")
    else:
        plt.xlabel("Iter")
    plt.ylabel("Loss")
    plt.title("Loss values of Generator and Discriminator")
    plt.legend()
    fig.savefig(opt.save_path + "/Loss_{0:d}ephoch.png".format(opt.n_epochs))  # グラフをファイルに保存する
    # plt.show()


def plot_d_out(result, opt):
    fig = plt.figure()  # グラフの描画先の準備
    x = list(range(1, len(result["log_d_out_real"]) + 1 )) # 0 ~ LastEpochまで順番に生成
    y = result["log_d_out_real"]
    plt.plot(x, y, label="input RealImg")
    y = result["log_d_out_fake1"]
    plt.plot(x, y, label="input FakeImg1")
    y = result["log_d_out_fake2"]
    plt.plot(x, y, label="input FakeImg2")
    if opt.plot_epoch:
        plt.xlabel("Epoch")
    else:
        plt.xlabel("Iter")
    plt.ylabel("Output value")
    plt.title("Discriminator's output value")
    plt.legend()
    fig.savefig(opt.save_path + "/Discriminator_out_{0:d}ephoch.png".format(opt.n_epochs))  # グラフをファイルに保存する
    # plt.show()


# 画像とネットワークを保存
def save(epoch, generate_img, opt):
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)

    # 一つの画像に統合
    # paddingの値→隙間, nrow=行あたりの画像数（端数は埋められる）黄金比になるように調節したお
    joined_images_tensor = make_grid(
                        generate_img, nrow=int(np.sqrt(((1+np.sqrt(5))/2)*generate_img.shape[0])), padding=8)
    # PyTorchのテンソル→Numpy配列
    joined_images = joined_images_tensor
    save_image(joined_images, opt.save_path+"/epoch_{0:d}.png".format(epoch+1))