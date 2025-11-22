from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

# CIFAR-100 statistics
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_cifar100_transforms():
    """
    Returns train and test transforms for CIFAR-100.
    Train transform has strong but label-preserving augmentations.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value=0,
            inplace=False,
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    return train_transform, test_transform


def get_cifar100_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 512,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders for CIFAR-100.
    """
    train_transform, test_transform = get_cifar100_transforms()

    trainset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, testloader