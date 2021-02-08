# コマンドライン引数の設定に必要なimport
import argparse

# パッケージのimport
import random
import math
import time
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
import os
import statistics
import time
import datetime

# pytorch たちのimport
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.autograd import Variable

# 自分のファイルのimport
import generator
import discriminator
from load_mnist import load_MNIST


# Setup seeds　コレいるの？？？？
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# パラメータ定義
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4096, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="Size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
parser.add_argument("--sample_interval", type=int, default=3, help="Interval betwen image samples")
parser.add_argument("--img_path", type=str, default="./data", help="Path of the folder where Mnist is located")
parser.add_argument("--save_path", type=str, default="./gan_saved_test", help="Path of the folder you want to save")
parser.add_argument("--nrow", type=int, default="10", help="Path of the folder you want to save")
opt = parser.parse_args([])


# バッチイメージの最初の一枚だけ取り出してプロットする関数
def plot_img_batch_top(images):
    # detach=>numpy型に変換
    img_transformed = images[0].detach().to('cpu')   # 作成画像のバッチ塊の最初だけ取り出し(fake_images[0]), 勾配情報を切り離し(detach()), numpyに変換

    # チャンネル数、高さ、横 => 高さ、横
    img_transformed = np.squeeze(img_transformed)

    # plot
    plt.imshow(img_transformed, "gray")
    plt.show()


def plot_loss(result):
    x = list(range(len(result["log_loss_G"]))) # 0 ~ LastEpochまで順番に生成
    y = result["log_loss_G"]
    plt.plot(x, y, label="Generator loss")
    y = result["log_loss_D"]
    plt.plot(x, y, label="Discriminator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss values of Generator and Discriminator")
    plt.legend()
    plt.show()


def plot_d_out(result):
    x = list(range(len(result["log_d_out_real"]))) # 0 ~ LastEpochまで順番に生成
    y = result["log_d_out_real"]
    plt.plot(x, y, label="input RealImg")
    y = result["log_d_out_fake"]
    plt.plot(x, y, label="input FakeImg")
    plt.xlabel("Epoch")
    plt.ylabel("Output value")
    plt.title("Discriminator's output value")
    plt.legend()
    plt.show()


# 画像とネットワークを保存
def save(epoch, generate_img, opt):
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)

    # 一つの画像に統合
    # paddingの値→隙間, nrow=行あたりの画像数（端数は埋められる）黄金比になるように調節したお
    joined_images_tensor = make_grid(
                        generate_img, nrow=int(np.sqrt(((1+np.sqrt(5))/2)*opt.batch_size)), padding=8)
    # PyTorchのテンソル→Numpy配列
    joined_images = joined_images_tensor
    save_image(joined_images, opt.save_path+"/epoch_{0:d}.png".format(epoch+1))



def train(loader_train, generator, discriminator, optimizer, loss_fn, device, opt, epoch, result):

    with tqdm(loader_train, ncols=180, leave=False) as pbar:
        log_loss_G, log_loss_D = [], []     # lossのログを取る
        log_d_out_real, log_d_out_fake = [], []
        # ミニバッチ毎に学習
        for _, (data, targets) in enumerate(pbar):

            # 本物の画像をセット
            real_img = data.to(device)
            real_label = targets.to(device)
            ones = torch.ones((data.shape[0], 1), requires_grad=False, device=device)
            zeros = torch.zeros((data.shape[0], 1), requires_grad=False, device=device)


            #########################
            #   Generator の学習
            #########################
            # 偽画像生成
            input_z = torch.randn(data.shape[0], opt.latent_dim, device=device) # ノイズ
            fake_img = generator(input_z)   # generatorで画像生成

            # 偽画像を一時保存, 勾配情報切り離し
            fake_img_tensor = fake_img.detach()

            # discriminatorでチェック
            # out = discriminator(fake_img)

            g_loss = loss_fn(discriminator(fake_img), ones)     # Generatorは偽画像をDiscriminatorに入れた時に, 1(本物と認識)になってほしい
            # 勾配の初期化
            optimizer["G_optimizer"].zero_grad()
            optimizer["D_optimizer"].zero_grad()
            g_loss.backward()
            optimizer["G_optimizer"].step()



            #########################
            #   Discriminator の学習
            #########################
            out_d_real = discriminator(real_img)
            out_d_fake = discriminator(fake_img_tensor)
            d_loss_real = loss_fn(out_d_real, ones)            # 本物画像は1を出力したい
            d_loss_fake = loss_fn(out_d_fake, zeros)    # 偽画像は0を出力したい
            d_loss = d_loss_fake + d_loss_real

            log_d_out_real.append(out_d_real.detach().clone().to('cpu').numpy().flatten().mean())
            log_d_out_fake.append(out_d_fake.detach().clone().to('cpu').numpy().flatten().mean())
            # print(type(out_d_list), out_d_list.shape)

            # 勾配の初期化と更新
            optimizer["G_optimizer"].zero_grad()
            optimizer["D_optimizer"].zero_grad()
            d_loss.backward()
            optimizer["D_optimizer"].step()

            # プログレスバー情報更新
            pbar.set_postfix(OrderedDict(
                epoch="{:>10}".format(epoch),
                Generator_loss="{:.4f}".format(g_loss.item()),
                Discriminator_loss="{:.4f}".format(d_loss.item())))

            # 1バッチ(iter)毎にlossの値をリストに保存
            log_loss_G.append(g_loss.item())
            log_loss_D.append(d_loss.item())


    return statistics.mean(log_loss_G), statistics.mean(log_loss_D), fake_img_tensor, sum(log_d_out_real)/len(log_d_out_real), sum(log_d_out_fake)/len(log_d_out_fake)





def main(opt):
    # GPU を使えるかどうかの確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==================================")
    print("This Program avtivating at", device, "!!")
    print("==================================")
    start = time.time()




    # model の作成と GPU 使えればそっちに持っていく　
    G_model = generator.Generator(opt).to(device)
    D_model = discriminator.Discriminator(opt).to(device)

    # Loss function
    adversarial_loss = torch.nn.BCELoss().to(device)

    # MNIST をロード
    data = load_MNIST(opt=opt)     # data["train], data["test"] にそれぞれ DataLoader が格納

    # Optimizers
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizers = {"G_optimizer" : G_optimizer, "D_optimizer" : D_optimizer}

    with tqdm(range(opt.n_epochs), ncols=200, leave=False) as pbar:
        # エラー推移
        result = {}
        result["log_loss_G"] = []
        result["log_loss_D"] = []
        result["log_d_out_real"] = []
        result["log_d_out_fake"] = []
        for epoch in pbar:
            loss_g, loss_d, fake_img_tensor, real, fake = train(
                                        loader_train=data["train"], generator=G_model,
                                        discriminator=D_model, optimizer=optimizers, 
                                        loss_fn=adversarial_loss, device=device, opt=opt,
                                        epoch=epoch, result=result)

            # 1エポック終了したら, 1エポックでlossの平均を取る
            result["log_loss_G"].append(loss_g)
            result["log_loss_D"].append(loss_d)
            result["log_d_out_real"].append(real)
            result["log_d_out_fake"].append(fake)

            # 1エポック毎に生成画像とネットワークの保存
            save(epoch=epoch, generate_img=fake_img_tensor, opt=opt)

            # プログレスバー情報更新
            pbar.set_postfix(OrderedDict(
                Discriminators_output_when_inputREAL="{:.4f}".format(result["log_d_out_real"][-1]),
                Discriminators_output_when_inputFAKE="{:.4f}".format(result["log_d_out_fake"][-1])
                ))

        elapsed_time = time.time() - start
        print ("time:{0}".format(datetime.timedelta(seconds=elapsed_time)))
        print(elapsed_time)
        plot_loss(result)
        plot_d_out(result)

        print("--END--")



if __name__ == "__main__":
    main(opt)