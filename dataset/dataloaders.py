from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T


def get_dataloader_train(transform, dataset_root="tiny-imagenet/tiny-imagenet-200", batch_size=32):
    tiny_imagenet_data_train = ImageFolder(
        root=f"{dataset_root}/train/", transform=transform
    )
    train_loader = DataLoader(
        tiny_imagenet_data_train, batch_size=32, shuffle=True, num_workers=8
    )
    return train_loader


def get_dataloader_val(transform, dataset_root="tiny-imagenet/tiny-imagenet-200", batch_size=32):
    tiny_imagenet_data_val = ImageFolder(
        root=f"{dataset_root}/val/", transform=transform
    )
    val_loader = DataLoader(tiny_imagenet_data_val, batch_size=32, shuffle=False)

    return val_loader
