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
from generator import Generator
from discriminator import Discriminator
from train import train
from load_datasets import *
from others import *

# Setup seeds　コレいるの？？？？
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# パラメータ定義
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="Number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="Dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="Size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--sample_interval", type=int, default=3, help="Interval betwen image samples")
parser.add_argument("--img_path", type=str, default="../data", help="Path of the folder where Mnist is located")
parser.add_argument("--save_path", type=str, default="./dcgan_saved_20epoch", help="Path of the folder you want to save")
parser.add_argument("--nrow", type=int, default="10", help="Path of the folder you want to save")
parser.add_argument("--ngf", type=int, default=64, help="relates to the depth of feature maps carried through the generator")
parser.add_argument("--ndf", type=int, default=64, help="relates to the depth of feature maps carried through the discriminator")
parser.add_argument("--plot_epoch", type=bool, default=False, help="Making the plot's horizontal axis an epoch")
opt = parser.parse_args([])


def main():
    # GPU を使えるかどうかの確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==================================")
    print("DCGAN is avtivating at", device, "!!")
    print("==================================")
    start = time.time()


    # Create the generator
    netG = Generator(opt).to(device)
    netD = Discriminator(opt).to(device)


    print(netD)
    print(netG)

    #  重みをランダムに初期化 mean=0, stdev=0.2.
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers に Adam をセット
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, 0.999))
    optimizers = {"optimizerG" : optimizerG, "optimizerD" : optimizerD}
    
    # MNIST をロード
    # data = load_MNIST(img_size=opt.img_size, batch_size=opt.batch_size, img_path=opt.img_path)     # data["train], data["test"] にそれぞれ DataLoader が格納
    # data = load_CIFAR10(img_size=opt.img_size, batch_size=opt.batch_size, img_path=opt.img_path)
    # data = load_CelebA(img_size=opt.img_size, batch_size=opt.batch_size, img_path=opt.img_path)
    data = load_local_CelebA(img_size=opt.img_size, batch_size=opt.batch_size, img_path="F:/datasets/GAN/data/celebA")
    
    

    with tqdm(range(opt.n_epochs), leave=False) as pbar:
        # エラー推移
        result = {}
        result["log_loss_G"] = []
        result["log_loss_D"] = []
        result["log_d_out_real"] = []
        # Discriminator の生の出力値をモニター
        result["log_d_out_fake1"] = []  # Generator が重み更新直後に偽画像を D に入れた時の出力値（基本0.5に近い値）
        result["log_d_out_fake2"] = []  # Discriminator が重み更新直後に偽画像を D に入れた時の出力値（基本0に近い値）
        for epoch in pbar:
            log = train(
                    loader_train=data["train"], generator=netG,
                    discriminator=netD, optimizer=optimizers, 
                    loss_fn=criterion, device=device, opt=opt,
                    epoch=epoch, result=result)

            # 1エポック終了したら, 1エポックでlossの平均を取る
            result["log_loss_G"].extend(log['loss_g'])
            result["log_loss_D"].extend(log['loss_d'])
            result["log_d_out_real"].extend(log['real'])
            result["log_d_out_fake1"].extend(log['fake1'])
            result["log_d_out_fake2"].extend(log['fake2'])

            # 1エポック毎に生成画像とネットワークの保存
            save(epoch=epoch, generate_img=log['fake_img_tensor'], opt=opt)

            # プログレスバー情報更新
            if opt.plot_epoch:
                pbar.set_postfix(OrderedDict(
                    D_out_REAL="{:.4f}".format(result["log_d_out_real"][-1]),
                    D_out_FAKE1="{:.4f}".format(result["log_d_out_fake1"][-1]),
                    D_out_FAKE2="{:.4f}".format(result["log_d_out_fake2"][-1])
                    ))
            else:
                pbar.set_postfix(OrderedDict(
                    D_out_REAL="{:.4f}".format(sum(log['real'])/len(log['real'])),
                    D_out_FAKE1="{:.4f}".format(sum(log['fake1'])/len(log['fake1'])),
                    D_out_FAKE2="{:.4f}".format(sum(log['fake2'])/len(log['fake2']))
                    ))

    elapsed_time = time.time() - start
    print ("time:{0}".format(datetime.timedelta(seconds=elapsed_time)))
    plot_loss(result, opt)
    plot_d_out(result, opt)

    print("--END--")

if __name__ == "__main__":
    main()