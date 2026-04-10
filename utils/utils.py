import torch
import os
import shutil


def reorganize_tiny_imagenet_val(dataset_root="tiny-imagenet/tiny-imagenet-200"):
    val_dir = os.path.join(dataset_root, "val")
    annotations_path = os.path.join(val_dir, "val_annotations.txt")
    images_dir = os.path.join(val_dir, "images")

    if not os.path.exists(images_dir):
        return

    with open(annotations_path) as f:
        for line in f:
            fn, cls, *_ = line.split("\t")
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

            shutil.copyfile(
                os.path.join(images_dir, fn),
                os.path.join(val_dir, cls, fn),
            )

    shutil.rmtree(images_dir)
