import torch
import itertools
import os
import numpy as np
import pytorch_warmup as warmup
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
from generator import Generator
from discriminator import Discriminator
from ImagePool import ImagePool
from GANLoss import GANLoss

class CycleGAN(object):

    def __init__(self, opt):
        self.device = opt.device
        self.num_epoch = opt.num_epoch
        self.netG_AtoB = Generator(opt).to(self.device)
        self.netG_BtoA = Generator(opt).to(self.device)
        self.netD_B = Discriminator(opt).to(self.device)
        self.netD_A = Discriminator(opt).to(self.device)

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        # targetが本物か偽物かで代わるのでオリジナルのGANLossクラスを作成
        self.criterionGAN = GANLoss(opt.device)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        # Generatorは2つのパラメータを同時に更新
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_AtoB.parameters(), self.netG_BtoA.parameters()),
            lr=opt.lr,
            betas=(opt.beta1, 0.999),
            weight_decay=opt.weight_decay)
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D_B)
        self.optimizers.append(self.optimizer_D_A)

        # warmup and weight decay
        self.lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=opt.total_iter)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D_A, T_max=opt.total_iter)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D_B, T_max=opt.total_iter)
        self.warmup_scheduler_G = warmup.UntunedLinearWarmup(self.optimizer_G)
        self.warmup_scheduler_D_A = warmup.UntunedLinearWarmup(self.optimizer_D_A)
        self.warmup_scheduler_D_B = warmup.UntunedLinearWarmup(self.optimizer_D_B)

        self.log_dir = opt.log_dir_path
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.scaler = torch.cuda.amp.GradScaler() 
        torch.backends.cudnn.benchmark = True



    # 1バッチ分の画像のセット
    def set_input(self, input):
        input_A = input['A'].to(self.device, non_blocking=True) # non_blocking=Trueで転送時にcpu強制稼働
        input_B = input['B'].to(self.device, non_blocking=True) # non_blocking=Trueで転送時にcpu強制稼働
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['path_A']



    # Generator の backward()
    def backward_G(self, real_A, real_B):
        # Generatorに関連するlossと勾配計算処理
        lambda_idt = 0.5
        lambda_A = 10.0
        lambda_B = 10.0

        # Runs the forward pass with autocasting.
        with torch.cuda.amp.autocast():
            # === Identity Mapping Loss === #
            # G_AtoB, G_BtoAは変換先ドメインの本物画像を入力したときはそのまま出力するべき
            # netG_AtoBはドメインAの画像からドメインBの画像を生成するGeneratorだが
            # ドメインBの画像も入れることができる
            # その場合は何も変換してほしくないという制約
            idt_B = self.netG_AtoB(real_B)  # B を生成する G_AtoB に B を流す。(画像は変わってほしくない)
            loss_idt_B = self.criterionIdt(idt_B, real_B) * lambda_B * lambda_idt   # 変換前後でL1ノルムをとる

            idt_A = self.netG_BtoA(real_A)  # A を生成する G_BtoA に A を流す。(画像は変わってほしくない)
            loss_idt_A = self.criterionIdt(idt_A, real_A) * lambda_A * lambda_idt   # 変換前後でL1ノルムをとる


            # === Adversarial Loss === #
            # Generator は Discriminator を騙すように学習
            # GAN loss D_B( G_AtoB(A) )
            # G_AtoB としては生成した偽物画像が本物（True）とみなしてほしい
            fake_B = self.netG_AtoB(real_A)
            pred_fake = self.netD_B(fake_B)
            loss_G_AtoB = self.criterionGAN(pred_fake, True)

            # GAN loss D_A(G_B(B))
            # G_Bとしては生成した偽物画像が本物（True）とみなしてほしい
            fake_A = self.netG_BtoA(real_B)
            pred_fake = self.netD_A(fake_A)
            loss_G_BtoA = self.criterionGAN(pred_fake, True)


            # === Cycle Consistency Loss === #
            # real_A => fake_B => rec_Aが元のreal_Aに近いほどよい
            rec_A = self.netG_BtoA(fake_B)
            loss_cycle_A = self.criterionCycle(rec_A, real_A) * lambda_A
            
            # real_B => fake_A => rec_Bが元のreal_Bに近いほどよい
            rec_B = self.netG_AtoB(fake_A)
            loss_cycle_B = self.criterionCycle(rec_B, real_B) * lambda_B


            # combined loss
            loss_G = loss_G_AtoB + loss_G_BtoA + loss_cycle_A + loss_cycle_B + loss_idt_B + loss_idt_A
            # loss_G.backward()


        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        self.scaler.scale(loss_G).backward()
        del loss_G  # 誤差逆伝播を実行後、計算グラフを削除


        # 次のDiscriminatorの更新でfake画像が必要なので一緒に返す
        return loss_G_AtoB.item(), loss_G_BtoA.item(), loss_cycle_A.item(), loss_cycle_B.item(), loss_idt_B.item(), loss_idt_A.item(), fake_A.detach(), fake_B.detach()




    def backward_D_A(self, real_A, fake_A):
        # ドメインBから生成したfake_Aが本物か偽物か見分ける

        # fake_Aを直接使わずに過去に生成した偽画像から半分は新しくランダムサンプリング
        fake_A = self.fake_A_pool.query(fake_A)
        
        # Runs the forward pass with autocasting.
        with torch.cuda.amp.autocast():
            # 本物画像を入れたときは本物と認識するほうがよい
            pred_real = self.netD_A(real_A)
            loss_D_real = self.criterionGAN(pred_real, True)

            # 偽物画像を入れたときは偽物と認識するほうがよい
            pred_fake = self.netD_A(fake_A)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            
            # combined loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            # loss_D_A.backward()

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        self.scaler.scale(loss_D_A).backward()
        loss_item = loss_D_A.item()
        del loss_D_A  # 誤差逆伝播を実行後、計算グラフを削除

        return loss_item, torch.mean(pred_real).item(), torch.mean(pred_fake).item()




    def backward_D_B(self, real_B, fake_B):
        # ドメインAから生成したfake_Bが本物か偽物か見分ける

        # fake_Bを直接使わずに過去に生成した偽画像から半分は新しくランダムサンプリング
        fake_B = self.fake_B_pool.query(fake_B)

        # Runs the forward pass with autocasting.
        with torch.cuda.amp.autocast():
            # 本物画像を入れたときは本物と認識するほうがよい
            pred_real = self.netD_B(real_B)
            loss_D_real = self.criterionGAN(pred_real, True)

            # ドメインAから生成した偽物画像を入れたときは偽物と認識するほうがよい
            # fake_Bを生成したGeneratorまで勾配が伝搬しないようにdetach()する
            pred_fake = self.netD_B(fake_B)
            loss_D_fake = self.criterionGAN(pred_fake, False)

            # combined loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            # loss_D_B.backward()
        
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        self.scaler.scale(loss_D_B).backward()
        loss_item = loss_D_B.item()
        del loss_D_B  # 誤差逆伝播を実行後、計算グラフを削除

        return loss_item, torch.mean(pred_real).item(), torch.mean(pred_fake).item()



    def optimize(self):
        real_A = Variable(self.input_A)
        real_B = Variable(self.input_B)

        # update Generator (G_AtoB and G_BtoA)
        # self.optimizer_G.zero_grad()
        for param_G_AtoB, param_G_BtoA in zip(self.netG_AtoB.parameters(), self.netG_BtoA.parameters()):
            param_G_AtoB.grad = None
            param_G_BtoA.grad = None
        loss_G_AtoB, loss_G_BtoA, loss_cycle_A, loss_cycle_B, loss_idt_B, loss_idt_A, fake_A, fake_B = self.backward_G(real_A, real_B)
        self.scaler.step(self.optimizer_G)   # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # self.optimizer_G.step()

        # update D_A
        # self.optimizer_D_A.zero_grad()
        for param_D_A in self.netD_A.parameters():
            param_D_A.grad = None
        loss_D_A, D_A_out_real, D_A_out_fake = self.backward_D_A(real_A, fake_A)
        self.scaler.step(self.optimizer_D_A)   # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # self.optimizer_D_A.step()

        # update D_B
        # self.optimizer_D_B.zero_grad()
        for param_D_B in self.netD_B.parameters():
            param_D_B.grad = None
        loss_D_B, D_B_out_real, D_B_out_fake = self.backward_D_B(real_B, fake_B)
        self.scaler.step(self.optimizer_D_B)   # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # self.optimizer_D_B.step()



        # Updates the scale for next iteration
        self.scaler.update() 

        ret_loss = [loss_G_AtoB, loss_D_B,
                    loss_G_BtoA, loss_D_A,
                    loss_cycle_A, loss_cycle_B,
                    loss_idt_B, loss_idt_A,
                    D_A_out_real, D_A_out_fake,
                    D_B_out_real, D_B_out_fake]

        return np.array(ret_loss)




    def train(self, data_loader, epoch, max_iter, writer):
        running_loss = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with tqdm(data_loader, leave=False) as pbar:
            for batch_idx, data in enumerate(pbar):
                self.set_input(data)        # データのセット
                losses = self.optimize()    # foward/backward/step とかいろいろやってくれる
                running_loss += losses      # 1batch(iter)の Loss の値

                # プログレスバーの更新
                pbar.set_postfix(OrderedDict(
                    epoch='{:>10}/{:<6}'.format(epoch, self.num_epoch),
                    G_All_Loss='{:.4f}'.format(np.sum(losses)-losses[1]-losses[3]),
                    D_All_Loss='{:.4f}'.format(losses[1]+losses[3])))
                
                # tensorboard のログ取り
                writer.add_scalars( 'Generator Loss/G A2B', 
                                    {   'Adversarial Loss':losses[0],
                                        'Cycle Consistency Loss':losses[4],
                                        'Identity Mapping Loss':losses[6]   },
                                    epoch*(max_iter)+batch_idx)
                writer.add_scalars( 'Generator Loss/G B2A', 
                                    {   'Adversarial Loss':losses[2],
                                        'Cycle Consistency Loss':losses[5],
                                        'Identity Mapping Loss':losses[7]   },
                                    epoch*(max_iter)+batch_idx)
                writer.add_scalars( 'Discriminator Loss/Discriminator A and B', 
                                    {   'D_A Loss':losses[1],
                                        'D_B Loss':losses[3]    },
                                    epoch*(max_iter)+batch_idx)
                writer.add_scalars( 'Discriminator Out Value/A out values', 
                                    {   'D_A out REAL':losses[8],
                                        'D_A out FAKE':losses[9]    },
                                    epoch*(max_iter)+batch_idx)
                writer.add_scalars( 'Discriminator Out Value/B out values', 
                                    {   'D_B out REAL':losses[10],
                                        'D_B out FAKE':losses[11]    },
                                    epoch*(max_iter)+batch_idx)
            running_loss /= len(data_loader)

        # 学習率スケジュール
        self.lr_scheduler_G.step()
        self.warmup_scheduler_G.dampen()
        self.lr_scheduler_D_A.step()
        self.warmup_scheduler_D_A.dampen()
        self.lr_scheduler_D_B.step()
        self.warmup_scheduler_D_B.dampen()

        return running_loss



    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.log_dir, save_filename)
        # GPUで動いている場合はCPUに移してから保存
        # これやっておけばCPUでモデルをロードしやすくなる？
        torch.save(network.to('cpu').state_dict(), save_path)
        # GPUに戻す
        if torch.cuda.is_available():
            network.to(self.device)



    def load_network(self, network, network_label, epoch_label):
        load_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self.log_dir, load_filename)
        print(load_filename, load_path)
        network.load_state_dict(torch.load(load_path))



    def save(self, label):
        self.save_network(self.netG_AtoB,   'G_AtoB',   label)
        self.save_network(self.netG_BtoA,   'G_BtoA',   label)
        self.save_network(self.netD_A,      'D_A',      label)
        self.save_network(self.netD_B,      'D_B',      label)



    def load(self, label):
        self.load_network(self.netG_AtoB,   'G_AtoB',   label)
        self.load_network(self.netG_BtoA,   'G_BtoA',   label)
        self.load_network(self.netD_A,      'D_A',      label)
        self.load_network(self.netD_B,      'D_B',      label)