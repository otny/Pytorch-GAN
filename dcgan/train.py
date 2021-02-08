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


def train(loader_train, generator, discriminator, optimizer, loss_fn, device, opt, epoch, result, real_label=1., fake_label=0.):
    with tqdm(loader_train, leave=False) as pbar:
        log_loss_G, log_loss_D = [], []     # lossのログを取る
        log_d_out_real, log_d_out_fake1, log_d_out_fake2 = [], [], []   # fake1, fake2はそれぞれGが学習後の予測値と, Dが学習後の予測値のログ
        # ミニバッチ毎に学習
        for i, (data, targets) in enumerate(pbar):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            discriminator.zero_grad()   # Discriminator重み初期化
            real_img = data.to(device)  # real img 取得
            batch_size = data.shape[0]
            label = torch.full((batch_size,), real_label, device=device) # ラベル作成

            output = discriminator(real_img).view(-1)   # Discriminator 順伝播
            errD_real = loss_fn(output, label)          # real img 損失計算
            errD_real.backward()                        # 独立して誤差逆伝播
            log_d_out_real.append(output.mean().item()) # real img を流した時の discriminator の出力を保存

            # ノイズ作成
            noise = torch.randn(batch_size, opt.latent_dim, 1, 1, device=device)
            fake_img = generator(noise)     # 偽画像生成
            label.fill_(fake_label)     # ラベルの数字を 0. にまるごと変える

            output = discriminator(fake_img.detach()).view(-1)  # Discriminator に偽画像を流す
            errD_fake = loss_fn(output, label)                  # fake img 損失計算
            errD_fake.backward()                                # 独立して誤差逆伝播
            log_d_out_fake1.append(output.mean().item())         # fake img を流した時の discriminator の出力を保存
            errD = errD_real + errD_fake
            optimizer["optimizerD"].step()                      # D 重み更新


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()                # 重み初期化
            label.fill_(real_label)         # ラベルを real label に変換
            output = discriminator(fake_img).view(-1)   # Fake 画像をDiscriminatorに流す
            errG = loss_fn(output, label)   # Generator の損失計算
            errG.backward()                 # 重み更新
            optimizer["optimizerG"].step()  
            log_d_out_fake2.append(output.mean().item())



            # プログレスバー情報更新
            pbar.set_postfix(OrderedDict(
                epoch="{:>10}".format(epoch),
                G_loss="{:.4f}".format(errG.item()),
                D_loss="{:.4f}".format(errD.item())))

            # 1バッチ(iter)毎にlossの値をリストに保存
            log_loss_G.append(errG.item())
            log_loss_D.append(errD.item())

    if opt.plot_epoch:
        return {    'loss_g'            : [statistics.mean(log_loss_G)],
                    'loss_d'            : [statistics.mean(log_loss_D)],
                    'real'              : [sum(log_d_out_real)/len(log_d_out_real)],
                    'fake1'             : [sum(log_d_out_fake1)/len(log_d_out_fake1)],
                    'fake2'             : [sum(log_d_out_fake2)/len(log_d_out_fake2)],
                    'fake_img_tensor'   : fake_img.detach()
                }
    else:
        return {    'loss_g'            : log_loss_G,
                    'loss_d'            : log_loss_D,
                    'real'              : log_d_out_real,
                    'fake1'             : log_d_out_fake1,
                    'fake2'             : log_d_out_fake2,
                    'fake_img_tensor'   : fake_img.detach()
                }
