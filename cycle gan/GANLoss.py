import torch.nn as nn
import torch
from torch.autograd import Variable

# MSELossを計算するだけのクラス
# 計算を行うのが Real か Fake かで自分で target_tensorラベル を作っている
class GANLoss(nn.Module):
    
    def __init__(self, device):
        super(GANLoss, self).__init__()
        self.device = device
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.MSELoss()
    
    # label を real or fake にセット
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            # 高速化のため？
            # varがNoneのままか形状が違うときに作り直す
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.ones(input.size()).to(self.device)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.zeros(input.size()).to(self.device)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    # MSELossで損失計算
    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)