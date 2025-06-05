import os, sys
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch.nn as nn
import timm
import torch_pruning as tp
import pbench
pbench.forward_patch.patch_timm_forward()
from tqdm import tqdm
import argparse
import torchvision as tv
import wandb # Import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='Timm ResNet Pruning')
    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--is-torchvision', default=True, action='store_true')
    parser.add_argument('--data-path', default='./data', type=str)
    parser.add_argument('--taylor-batchs', default=10, type=int)
    parser.add_argument('--train-batch-size', default=32, type=int)
    parser.add_argument('--val-batch-size', default=128, type=int)
    parser.add_argument('--pruning-ratio', default=0.5, type=float)
    parser.add_argument('--pruning-type', default='taylor', type=str)
    parser.add_argument('--test-accuracy', default=True, action='store_true')
    parser.add_argument('--global-pruning', default=True, action='store_true')
    parser.add_argument('--round-to', default=1, type=int)
    parser.add_argument('--save-as', default='./pruned_resnet50.pth', type=str)
    # Add WandB arguments
    parser.add_argument('--wandb-project', default='Isomorphic-Pruning', type=str,
                        help='WandB project name')
    parser.add_argument('--wandb-name', default=None, type=str,
                        help='WandB run name (optional)')
    parser.add_argument('--wandb-mode', default='online', type=str,
                        choices=['online', 'offline', 'disabled'],
                        help='WandB run mode')
    return parser.parse_args()

def prepare_cifar10(data_path, train_batch_size, val_batch_size):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    train_set = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_set = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def validate_model(model, loader, device):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset), loss / len(loader.dataset)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- WandB Initialization ---
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args), # Logs all argparse arguments
        mode=args.wandb_mode
    )
    # --- End WandB Initialization ---

    train_loader, val_loader = prepare_cifar10(args.data_path, args.train_batch_size, args.val_batch_size)

    model = tv.models.__dict__[args.model](weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    imp = tp.importance.GroupTaylorImportance() if args.pruning_type == 'taylor' else tp.importance.RandomImportance()
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)

    # Log initial model stats
    wandb.log({"initial_macs_G": base_macs / 1e9, "initial_params_M": base_params / 1e6})

    acc_ori, loss_ori = validate_model(model, val_loader, device)
    print("Before pruning: Acc = %.4f, Loss = %.4f" % (acc_ori, loss_ori))
    # Log initial accuracy and loss
    wandb.log({"before_pruning_accuracy": acc_ori, "before_pruning_loss": loss_ori})


    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        global_pruning=args.global_pruning,
        pruning_ratio=args.pruning_ratio,
        round_to=args.round_to
    )

    model.zero_grad()
    for i, (x, y) in enumerate(train_loader):
        if i >= args.taylor_batchs: break
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)
        loss.backward()

    for group in pruner.step(interactive=True):
        group.prune()

    # Re-initialize FC layer after pruning, as its input features might change.
    model.fc = nn.Linear(model.fc.in_features, 10).to(device)


    acc_pruned, loss_pruned = validate_model(model, val_loader, device)
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)

    print("After pruning: Acc = %.4f, Loss = %.4f" % (acc_pruned, loss_pruned))
    print("MACs: %.2f G => %.2f G" % (base_macs / 1e9, pruned_macs / 1e9))
    print("Params: %.2f M => %.2f M" % (base_params / 1e6, pruned_params / 1e6))

    # Log final pruning results
    wandb.log({
        "after_pruning_accuracy": acc_pruned,
        "after_pruning_loss": loss_pruned,
        "pruned_macs_G": pruned_macs / 1e9,
        "pruned_params_M": pruned_params / 1e6,
        "macs_reduction_ratio": (base_macs - pruned_macs) / base_macs,
        "params_reduction_ratio": (base_params - pruned_params) / base_params,
    })

    if args.save_as:
        torch.save(model, args.save_as)
        print(f"Model saved to {args.save_as}")

    wandb.finish() # End the WandB run

if __name__ == '__main__':
    main()