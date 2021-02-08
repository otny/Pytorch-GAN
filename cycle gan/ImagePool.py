from torch.autograd import Variable
import torch
import random

# 生成した画像とかその他いろんな画像を一時的に溜めておく(Pool) クラス
class ImagePool():

    def __init__(self, pool_size):
        self.pool_size = pool_size  # 保持しておく画像の数
        if self.pool_size > 0:
            self.num_imgs = 0       # 現在溜まっている画像枚数
            self.image_memory = []  # この変数に画像をためていき, 適宜ここから抽出

    # 入ってくるバッチ分の画像(imgaes)の半分はそのままに。もう半分は、一時的に溜めていた画像からランダムに選んで返す関数
    def query(self, images):
        # プールを使わないときはそのまま返す
        if self.pool_size == 0:
            return Variable(images)
        return_images = []

        # バッチ画像群からを1枚1枚見ていくfor文
        for image in images:
            # バッチの次元を削除して3Dテンソルに
            image = torch.unsqueeze(image, 0)

            if self.num_imgs < self.pool_size:
                # img が pool_size個溜まっていなければ溜めていく
                self.num_imgs = self.num_imgs + 1
                self.image_memory.append(image)
                return_images.append(image)
            else:
                # 画像が溜まりきっていれば確率0.5で入ってきた画像をそのまま返す
                p = random.uniform(0, 1)
                if p > 0.5:
                    # 確率0.5で, 溜まった画像からランダムに1つ抽出され, 選ばれたindexの画像は新しい画像に更新
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.image_memory[random_id].clone()
                    self.image_memory[random_id] = image
                    return_images.append(tmp)
                else:
                    # 確率0.5で, 入ってきた画像をそのまま返す
                    return_images.append(image)
        return Variable(torch.cat(return_images, 0))
        # return return_images