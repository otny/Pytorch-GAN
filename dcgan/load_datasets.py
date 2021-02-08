#############################
#   data loader
#############################
import torch
from torchvision import datasets, transforms


# data load 関数
def load_MNIST(img_size=28, batch_size=64, img_path="./dataset", intensity=1.0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(img_path+"/mnist",
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.Resize(img_size),
                           transforms.ToTensor(),
                           transforms.Normalize([0.5], [0.5]),  # 平均と標準偏差を0.5に正規化
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(img_path+"/mnist",
                       train=False,
                       transform=transforms.Compose([
                           transforms.Resize(img_size),
                           transforms.ToTensor(),
                           transforms.Normalize([0.5], [0.5]),  # 平均と標準偏差を0.5に正規化
                           transforms.Lambda(lambda x: x * intensity)
                       ])),
        batch_size=batch_size,
        shuffle=True
    )

    return {'train': train_loader, 'test': False}



def load_CelebA(img_size=64, batch_size=64, img_path="./dataset", intensity=1.0):
    train_loader = torch.utils.data.DataLoader(
                datasets.CelebA(root=img_path+"/celebA",
                            # split="all",
                            download=True,
                            transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])), 
                batch_size=batch_size,
                shuffle=True)

    return {'train': train_loader, 'test': False}



def load_local_CelebA(img_size=64, batch_size=64, img_path="./dataset", intensity=1.0):
    train_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(root=img_path,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])), 
                batch_size=batch_size,
                shuffle=True)

    return {'train': train_loader, 'test': False}



def load_CIFAR10(img_size=64, batch_size=64, img_path="./dataset", intensity=1.0):
    train_loader = torch.utils.data.DataLoader(
        dataset = datasets.CIFAR10(img_path+"/cifar10",
                            download=True,
                            transform=transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
                    ),
        batch_size=batch_size,
        shuffle=True
    )
    return {'train': train_loader, 'test': False}
