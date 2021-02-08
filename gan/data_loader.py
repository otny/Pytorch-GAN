import torch
import torch.utils.data as data
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import argparse

def make_datapath_list():
    """学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する。 """

    train_img_list = list()  # 画像ファイルパスを格納

    for img_idx in range(200):
        img_path = "./data/img_78/img_7_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8_" + str(img_idx)+'.jpg'
        train_img_list.append(img_path)

    return train_img_list

class ImageTransform():
    """画像の前処理クラス"""

    def __init__(self, mean, std, opt):
        self.data_transform = transforms.Compose([
                                                  transforms.Resize(opt.img_size),  #パラメータで定義した画像サイズ
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)
        ])
    def __call__(self, img):
        return self.data_transform(img)

class GAN_Img_Dataset(data.Dataset):
    """画像のDatasetクラス。PyTorchのDatasetクラスを継承"""

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    # shuffleで必要
    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''前処理をした画像のTensor形式のデータを取得'''
        # 画像取得
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅]白黒

        # 画像の前処理
        img_transformed = self.transform(img)

        return img_transformed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    # ファイルリストを作成
    train_img_list = make_datapath_list()

    # Datasetを作成
    mean = (0.5,)
    std = (0.5,)
    opt = parser.parse_args([])

    train_dataset = GAN_Img_Dataset(
        file_list = train_img_list,
        transform = ImageTransform(mean, std, opt))

    # DataLoaderを作成
    batch_size = 64

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True
        )

    # 動作の確認
    batch_iterator = iter(train_dataloader)  # イテレータに変換
    imgs = next(batch_iterator)  # 1番目の要素を取り出す
    print(imgs.size())  # torch.Size([64, 1, 64, 64])
    print(imgs.shape[0])

# DataLoaderの作成と動作確認
if __name__ == "__main__":
    main()