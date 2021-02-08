import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image, make_grid
import numpy as np




# データの描画
def imshow(img):
    npimg = img.numpy()
    npimg = 0.5 * (npimg + 1)  # [-1,1] => [0, 1]
    # [c, h, w] => [h, w, c]
    return np.transpose(npimg, (1, 2, 0))


# 2種類の画像を描画
def plot_imgs(img1, label1, img2, label2, batch_size):
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(imshow(make_grid(img1, nrow=int(batch_size/2))))
    plt.title(label1)
    plt.axis('off')


    plt.subplot(1, 2, 2)
    plt.imshow(imshow(make_grid(img2, nrow=int(batch_size/2))))
    plt.title(label2)
    plt.axis('off')
    plt.show()



def plot_loss(result, opt):
    fig = plt.figure()  # グラフの描画先の準備
    x = list(range(1, len(result['log_loss_G']) + 1 )) # 0 ~ LastEpochまで順番に生成
    y = result['log_loss_G']
    plt.plot(x, y, label='Generator loss')
    y = result['log_loss_D']
    plt.plot(x, y, label='Discriminator loss')
    if opt.plot_epoch:
        plt.xlabel('Epoch')
    else:
        plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title('Loss values of Generator and Discriminator')
    plt.legend()
    fig.savefig(opt.save_path + '/Loss_{0:d}ephoch.png'.format(opt.n_epochs))  # グラフをファイルに保存する
    # plt.show()


def plot_d_out(result, opt):
    fig = plt.figure()  # グラフの描画先の準備
    x = list(range(1, len(result['log_d_out_real']) + 1 )) # 0 ~ LastEpochまで順番に生成
    y = result['log_d_out_real']
    plt.plot(x, y, label='input RealImg')
    y = result['log_d_out_fake1']
    plt.plot(x, y, label='input FakeImg1')
    y = result['log_d_out_fake2']
    plt.plot(x, y, label='input FakeImg2')
    if opt.plot_epoch:
        plt.xlabel('Epoch')
    else:
        plt.xlabel('Iter')
    plt.ylabel('Output value')
    plt.title('Discriminator\'s output value')
    plt.legend()
    fig.savefig(opt.save_path + '/Discriminator_out_{0:d}ephoch.png'.format(opt.n_epochs))  # グラフをファイルに保存する
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
    save_image(joined_images, opt.save_path+'/epoch_{0:d}.png'.format(epoch+1))