# main.py
import os
import torch
from models.customnet import CustomNet
from torch import nn
from utils.utils import reorganize_tiny_imagenet_val
from dataset.dataloaders import get_dataloader_train, get_dataloader_val
from dataset.transforms import get_train_transforms, get_val_transforms
from train import train
from eval import validate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    batch_size = 32
    lr = 0.001
    momentum = 0.9
    num_epochs = 10

    dataset_root = os.environ.get(
        "DATASET_ROOT",
        "tiny-imagenet/tiny-imagenet-200",
    )

    print(f"Using dataset from: {dataset_root}")

    reorganize_tiny_imagenet_val(dataset_root)

    train_tf = get_train_transforms(
        img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )

    eval_tf = get_val_transforms(
        img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )

    train_loader = get_dataloader_train(
        train_tf, dataset_root=dataset_root, batch_size=batch_size
    )
    eval_loader = get_dataloader_val(
        eval_tf, dataset_root=dataset_root, batch_size=batch_size
    )

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_acc = 0.0

    print(f"Run the process for {num_epochs} epochs.")

    for epoch in range(1, num_epochs + 1):
        print(f"Running epoch {epoch}...")
        train(
            epoch,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        current_val_acc = validate(
            model, val_loader=eval_loader, criterion=criterion, device=device
        )

        best_acc = max(current_val_acc, best_acc)

    print(f"Best validation accuracy: {best_acc}")


if __name__ == "__main__":
    main()
