#############################
#   data loader
#############################

# 自作データセットを作るためには Dataset を継承
# __getitem__(index) を実装して [] 演算子で個別データにアクセスできるようにする
# データは辞書形式で返す
# 入力画像は [-1, 1] の範囲に正規化

import torch
import torchvision.transforms as transforms
import random
import os
import matplotlib.pyplot as plt
from PIL import Image

class UnalignedDataset(torch.utils.data.Dataset):
    '''Some Information about UnalignedDataset'''
    def __init__(self, is_train, opt):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        root_dir = os.path.join(opt.data_path, 'horse2zebra')

        if is_train:
            dir_A = os.path.join(root_dir, 'trainA')
            dir_B = os.path.join(root_dir, 'trainB')
        else:
            dir_A = os.path.join(root_dir, 'testA')
            dir_B = os.path.join(root_dir, 'testB')

        self.image_paths_A = self._make_dataset(dir_A)
        self.image_paths_B = self._make_dataset(dir_B)

        self.size_A = len(self.image_paths_A)
        self.size_B = len(self.image_paths_B)

        self.transform = self._make_transform(is_train)


    def __getitem__(self, index):
        index_A = index % self.size_A
        path_A = self.image_paths_A[index_A]

        # クラスBの画像はランダムに選択
        index_B = random.randint(0, self.size_B - 1)
        path_B = self.image_paths_B[index_B]

        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')

        # データ拡張
        A = self.transform(img_A)
        B = self.transform(img_B)

        return {'A': A, 'B': B, 'path_A': path_A, 'path_B': path_B}


    def __len__(self):
        return max(self.size_A, self.size_B)


    # jpgファイルの path list を取得
    def _make_dataset(self, dir):
        images = []
        for fname in os.listdir(dir):
            if fname.endswith('.jpg'):
                path = os.path.join(dir, fname)
                images.append(path)
        sorted(images)
        return images


    # Transform を一括構成
    def _make_transform(self, is_train):
        transforms_list = []
        transforms_list.append(transforms.Resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC))
        transforms_list.append(transforms.RandomCrop(self.opt.fine_size))
        if is_train:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # [0, 1] => [-1, 1]
        return transforms.Compose(transforms_list)



def get_loader(dataset, opt):
    return torch.utils.data.DataLoader( dataset, 
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=opt.num_workers)  # os.cpu_count()



def main():
    import argparse
    from others import imshow
    from torchvision.utils import make_grid
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='F:/datasets/CycleGAN', help='Path of the folder where Mnist is located')
    parser.add_argument('--load_size', type=int, default=286, help='Original image resized to this size')
    parser.add_argument('--fine_size', type=int, default=256, help='Crop a random this size from a load size image')
    parser.add_argument('--batch_size', type=int, default=1, help='Size of the batches')
    opt = parser.parse_args([])

    train_dataset = UnalignedDataset(is_train=True, opt=opt)
    train_loader = get_loader(train_dataset, opt)

    # Sample Test
    print(type(train_loader))
    batch = iter(train_loader).next()
    print(batch['A'].shape)
    print(batch['B'].shape)
    print(batch['path_A'])
    print(batch['path_B'])
    images_A = batch['A']  # horses
    images_B = batch['B']  # zebras
    print('TYPE =', type(images_A))


    plt.figure(figsize=(10, 20))

    plt.subplot(1, 2, 1)
    plt.imshow(imshow(make_grid(images_A, nrow=4)))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(imshow(make_grid(images_B, nrow=4)))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()