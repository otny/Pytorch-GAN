# コマンドライン引数の設定に必要なimport
import argparse

# パッケージのimport
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# pytorch たちのimport
import torch
from torchvision.utils import make_grid

# 自分のファイルのimport
from load_datasets import UnalignedDataset, get_loader
from others import plot_imgs
from CycleGAN import CycleGAN


# 再現性確保のために指定のseed値を設定
torch.manual_seed(1234)
np.random.seed(1234)


# パラメータ定義
parser = argparse.ArgumentParser()
parser.add_argument('--train_mode', type=bool, default=True, help='Run in "training mode" or "test mode" ')
parser.add_argument('--num_epoch', type=int, default=200, help='Number of epochs of training')
parser.add_argument('--batch_size', type=int, default=3, help='Size of the batches')
parser.add_argument('--lr', type=float, default=0.0002*3, help='Adam: learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam: decay of first order momentum of gradient')
parser.add_argument('--weight_decay', type=float, default=0, help='Adam weight_decay, defalut: 0')
parser.add_argument('--load_size', type=int, default=286, help='Original image resized to this size')
parser.add_argument('--fine_size', type=int, default=256, help='Crop a random this size from a load size image')
parser.add_argument('--channels', type=int, default=3, help='Number of image channels')
parser.add_argument('--num_workers', type=int, default=2, help='Number of cpu cores used in the DataLoader, Recommended: 2')    # os.cpu_count() CPUスレッド数
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [ instance | batch  | none | (Spectral) 未実装 ]')
parser.add_argument('--conv_bias', type=bool, default=False, help='Using Bias for Pre-Normalization Convolution')
parser.add_argument('--save_epoch_freq', type=int, default=5, help='Interval betwen saving samples')
parser.add_argument('--log_dir_path', type=str, default='../logs/cyclegan/Results_after_optimization', help='Path of the Logging folder')
parser.add_argument('--data_path', type=str, default='F:/datasets/CycleGAN', help='Path of the folder where Mnist is located')
parser.add_argument('--save_path', type=str, default='./cycleGAN_save', help='Path of the folder you want to save')
parser.add_argument('--ngf', type=int, default=64, help='relates to the depth of feature maps carried through the generator')
parser.add_argument('--ndf', type=int, default=64, help='relates to the depth of feature maps carried through the discriminator')
opt = parser.parse_args([])



def main():
    # GPU を使えるかどうかの確認
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n==================================')
    print('CycleGAN is running on', opt.device, '!!')
    print('==================================')
    start = time.time()



    if opt.train_mode:
        ###############
        # train mode  #
        ###############

        # dataset
        train_dataset = UnalignedDataset(is_train=True, opt=opt)
        train_loader = get_loader(train_dataset, opt)
        opt.total_iter = len(train_loader)*opt.num_epoch

        model = CycleGAN(opt)                       # make model
        writer = SummaryWriter(opt.log_dir_path)    # tensorboardのログ用

        print(model.netD_A)
        print(model.netG_AtoB)

        # 未学習時のモデルを保存
        model.save('epoch0')

        with tqdm(range(opt.num_epoch), leave=False) as pbar:
            for epoch in pbar:
                losses = model.train(train_loader, epoch, np.ceil(train_dataset.__len__()/opt.batch_size), writer)

                if epoch+1 % opt.save_epoch_freq == 0:
                    model.save('epoch%d' % epoch+1)

        writer.close()
        elapsed_time = time.time() - start
        print ('Program Time:{0}'.format(datetime.timedelta(seconds=elapsed_time)))

    else:
        ###############
        #  test mode  #
        ###############
        model = CycleGAN(opt)
        model.load('epoch195')

        # dataset
        test_dataset = UnalignedDataset(is_train=False, opt=opt)
        test_loader = get_loader(test_dataset, opt)
        batch = iter(test_loader).next()


        fakeB = model.netG_AtoB(batch['A'].to(opt.device)).detach().to('cpu')
        elapsed_time = time.time() - start
        plot_imgs(batch['A'], 'real_A', fakeB, 'fake_B', opt.batch_size)


        fakeA = model.netG_BtoA(batch['B'].to(opt.device)).detach().to('cpu')
        elapsed_time = time.time() - start
        plot_imgs(batch['B'], 'real_B', fakeA, 'fake_A', opt.batch_size)



    print('--END--')

if __name__ == '__main__':
    main()