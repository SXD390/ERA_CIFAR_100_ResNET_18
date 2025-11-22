import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from src.model import resnet18_cifar
from src.dataset import get_cifar100_dataloaders


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scheduler: OneCycleLR,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch [{epoch}/{total_epochs}] (Train)")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / total
        avg_acc = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}%"})

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch [{epoch}/{total_epochs}] (Eval)")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / total
        avg_acc = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}%"})

    return running_loss / total, 100.0 * correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-100 from scratch.")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-100 data directory.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.1, help="Max learning rate for OneCycle.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save checkpoints.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    trainloader, testloader = get_cifar100_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = resnet18_cifar(num_classes=100).to(device)

    # Label smoothing to help generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(trainloader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )

    scaler = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu")

    best_acc = 0.0
    best_path = os.path.join(args.save_dir, "best_resnet_cifar100.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            trainloader,
            criterion,
            optimizer,
            scheduler,
            device,
            scaler,
            epoch,
            args.epochs,
        )
        test_loss, test_acc = evaluate(
            model,
            testloader,
            criterion,
            device,
            epoch,
            args.epochs,
        )

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                },
                best_path,
            )
            print(f"✨ New best test acc: {best_acc:.2f}% at epoch {epoch}. Saved to {best_path}")

    print(f"Training complete. Best test accuracy: {best_acc:.2f}%")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()