from torchvision import transforms as T


def get_train_transforms(img_size=64, mean=(0.5,) * 3, std=(0.5,) * 3):
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def get_val_transforms(img_size=64, mean=(0.5,) * 3, std=(0.5,) * 3):
    val_transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    return val_transform
