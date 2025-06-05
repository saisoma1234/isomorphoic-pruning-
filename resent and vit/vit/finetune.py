import os
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch.nn as nn
import timm
from tqdm import tqdm
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import OrderedDict
import json # Import json for loading and saving parameters
import wandb # Import wandb for logging

# --- IMPORTANT: You need torch_pruning to calculate MACs/Params at the end ---
try:
    import torch_pruning as tp
except ImportError:
    print("Warning: torch_pruning not found. MACs/Params calculation at the end will be skipped.")
    tp = None

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune pruned Vision Transformer model')
    parser.add_argument('--ckpt', required=True, type=str,
                        help='Path to the pruned model checkpoint.')
    parser.add_argument('--model-name', default='deit_small_patch16_224', type=str,
                        help='Name of the timm Vision Transformer model (e.g., deit_small_patch16_224, vit_base_patch16_224).')
    parser.add_argument('--data-path', default='./data', type=str,
                        help='Path to the CIFAR-10 dataset.')
    parser.add_argument('--epochs', default=3, type=int,
                        help='Number of epochs for fine-tuning.')
    parser.add_argument('--train-batch-size', default=32, type=int,
                        help='Batch size for training data loader.')
    parser.add_argument('--val-batch-size', default=64, type=int,
                        help='Batch size for validation data loader.')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='Learning rate for fine-tuning.')
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw'],
                        help='Optimizer for fine-tuning.')
    parser.add_argument('--wd', default=0.05, type=float,
                        help='Weight decay for optimizer.')
    parser.add_argument('--lr-scheduler', default='cosine', type=str, choices=['cosine'],
                        help='Learning rate scheduler type.')
    parser.add_argument('--save-as', default='./finetuned_vit.pth', type=str,
                        help='Path to save the fine-tuned model state_dict.')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='Number of data loading workers.')
    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='finetune-vit-pruning',
                        help='Weights & Biases project name.')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging.')
    return parser.parse_args()


def prepare_cifar10(data_path, train_batch_size, val_batch_size, num_workers, img_size):
    """Prepares CIFAR-10 data loaders with specified augmentations and resizing."""
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_set = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def validate_model(model, loader, device):
    """Evaluates model accuracy and loss."""
    model.eval()
    correct = 0
    total_loss = 0 # Renamed 'loss' to 'total_loss' to avoid conflict with loop variable
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    accuracy = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)
    return accuracy, avg_loss


def safe_load_model(ckpt_path, model_name, device, img_size):
    """
    Safely loads a timm model, reconstructing its architecture based on saved pruning parameters.
    This function is crucial for loading pruned models correctly.
    """
    try:
        # Load the saved pruning parameters
        base_ckpt_path = os.path.splitext(ckpt_path)[0]
        params_path = base_ckpt_path + "_params.json"
        
        pruned_params = {}
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                pruned_params = json.load(f)
            print(f"Loaded pruning parameters from {params_path}")
        else:
            # If params file is not found, assume it's a standard unpruned model
            print(f"Warning: Pruning parameters file {params_path} not found. Proceeding with default model architecture.")
            model = timm.create_model(model_name, pretrained=False)
            # Adjust classifier head for 10 classes
            if hasattr(model, 'head'):
                if hasattr(model.head, 'fc'):
                    model.head.fc = nn.Linear(model.head.fc.in_features, 10)
                else:
                    model.head = nn.Linear(model.head.in_features, 10)
            elif hasattr(model, 'fc'):
                model.fc = nn.Linear(model.fc.in_features, 10)
            model.to(device)
            state_dict = torch.load(ckpt_path, map_location=device)
            if all(k.startswith('module.') for k in state_dict.keys()):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                state_dict = new_state_dict
            model.load_state_dict(state_dict, strict=False) # Still use strict=False for robustness
            return model


        # Reconstruct the model with the correct pruned dimensions
        new_embedding_dim = pruned_params.get('new_embedding_dim', None)
        num_heads = pruned_params.get('num_heads', None)
        mlp_hidden_features = pruned_params.get('mlp_hidden_features', [])

        if new_embedding_dim is None or num_heads is None:
            raise ValueError("Missing 'new_embedding_dim' or 'num_heads' in loaded pruning parameters. Cannot reconstruct model.")

        # Create a base timm model
        model = timm.create_model(model_name, pretrained=False)

        # --- Manual reconstruction to match pruned structure ---
        # 1. Adjust patch_embed.proj output channels if it was pruned
        if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
            if model.patch_embed.proj.out_channels != new_embedding_dim:
                print(f"Adjusting patch_embed.proj from {model.patch_embed.proj.out_channels} to {new_embedding_dim}")
                model.patch_embed.proj = nn.Conv2d(
                    model.patch_embed.proj.in_channels,
                    new_embedding_dim,
                    kernel_size=model.patch_embed.proj.kernel_size,
                    stride=model.patch_embed.proj.stride,
                    padding=model.patch_embed.proj.padding,
                    bias=model.patch_embed.proj.bias is not None
                ).to(device)

        # 2. Adjust cls_token and pos_embed
        if hasattr(model, 'cls_token'):
            if model.cls_token.shape[2] != new_embedding_dim:
                print(f"Re-initializing cls_token to (1, 1, {new_embedding_dim}) for loading consistency.")
                model.cls_token = nn.Parameter(torch.zeros(1, 1, new_embedding_dim)).to(device)
                nn.init.trunc_normal_(model.cls_token, std=.02)

        if hasattr(model, 'pos_embed'):
            if model.pos_embed.shape[2] != new_embedding_dim:
                patch_size = model.patch_embed.proj.kernel_size[0]
                num_patches = (img_size // patch_size) * (img_size // patch_size)
                print(f"Re-initializing pos_embed to (1, {num_patches + 1}, {new_embedding_dim}) for loading consistency.")
                model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, new_embedding_dim)).to(device)
                nn.init.trunc_normal_(model.pos_embed, std=.02)

        # 3. Iterate through each block and re-initialize its internal layers
        new_head_dim = new_embedding_dim // num_heads
        
        for i, block in enumerate(model.blocks):
            # Attention module
            attn = block.attn
            attn.qkv = nn.Linear(new_embedding_dim, new_embedding_dim * 3, bias=attn.qkv.bias is not None).to(device)
            attn.proj = nn.Linear(new_embedding_dim, new_embedding_dim, bias=attn.proj.bias is not None).to(device)
            
            attn.num_heads = num_heads
            attn.head_dim = new_head_dim
            attn.scale = new_head_dim ** -0.5
            
            # MLP module
            mlp = block.mlp
            # Use the actual hidden_features from the pruned model's saved parameters.
            current_mlp_hidden_features = mlp_hidden_features[i] if i < len(mlp_hidden_features) and mlp_hidden_features[i] is not None else (4 * new_embedding_dim)

            mlp.fc1 = nn.Linear(new_embedding_dim, current_mlp_hidden_features, bias=mlp.fc1.bias is not None).to(device)
            mlp.fc2 = nn.Linear(current_mlp_hidden_features, new_embedding_dim, bias=mlp.fc2.bias is not None).to(device)

        # 4. Re-initialize the final classifier head
        if hasattr(model, 'head'):
            if hasattr(model.head, 'fc'):
                model.head.fc = nn.Linear(new_embedding_dim, 10).to(device)
            else:
                model.head = nn.Linear(new_embedding_dim, 10).to(device)
        elif hasattr(model, 'fc'):
            model.fc = nn.Linear(new_embedding_dim, 10).to(device)
        
        model.to(device)

        # Now load the state_dict
        state_dict = torch.load(ckpt_path, map_location=device)
        if all(k.startswith('module.') for k in state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict, strict=False) # Keep strict=False for robustness

        return model

    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        print("Possible solutions:")
        print("1. Ensure the model architecture matches --model-name.")
        print("2. Verify the checkpoint file and associated params.json are not corrupted or mismatch.")
        print("3. Check the new_embedding_dim, num_heads, and mlp_hidden_features saved by prune.py.")
        raise


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize wandb globally if not disabled
    if args.no_wandb:
        print("Weights & Biases logging is disabled.")
        # Replace wandb.init/log/finish with dummy functions if disabled
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.finish = lambda *args, **kwargs: None
        wandb.run = type('obj', (object,), {'summary': {}})() # Dummy summary object
    else:
        # Ensure wandb is imported and available
        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed. Please run 'pip install wandb' to enable logging.")
            args.no_wandb = True # Force disable if not installed
            wandb.init = lambda *args, **kwargs: None
            wandb.log = lambda *args, **kwargs: None
            wandb.finish = lambda *args, **kwargs: None
            wandb.run = type('obj', (object,), {'summary': {}})() # Dummy summary object
    
    # Start wandb run
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=f"Finetune_{args.model_name}_epochs{args.epochs}", config=args)


    # Load model with error handling
    try:
        # Get img_size for data loading consistently
        model_tmp = timm.create_model(args.model_name, pretrained=False)
        img_size = model_tmp.default_cfg.get('input_size', (3, 224, 224))[1]
        del model_tmp # Free memory

        print(f"\nLoading model {args.model_name} from {args.ckpt}")
        # The key change: Pass the correct ckpt_path for safe_load_model
        # For the initial load, it's args.ckpt (pruned_vit.pth)
        model = safe_load_model(args.ckpt, args.model_name, device, img_size)
        print("Model loaded successfully for fine-tuning.")
    except Exception as e:
        if not args.no_wandb:
            wandb.finish(status="failed") # Log failure to wandb
        return 
    
    train_loader, val_loader = prepare_cifar10(args.data_path, args.train_batch_size, args.val_batch_size, args.num_workers, img_size)

    # Optimizer and Scheduler for Transformer Models (ViT)
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported for ViT.")

    if args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    else:
        raise ValueError(f"LR scheduler {args.lr_scheduler} not supported.")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    best_acc = 0.0

    # Load original pruning parameters to save with fine-tuned model
    original_pruned_params = {}
    original_pruned_params_path = os.path.splitext(args.ckpt)[0] + "_params.json"
    if os.path.exists(original_pruned_params_path):
        with open(original_pruned_params_path, 'r') as f:
            original_pruned_params = json.load(f)
    else:
        print(f"Warning: Original pruned parameters file {original_pruned_params_path} not found. Parameters for fine-tuned model might be incomplete.")


    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training")
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (i + 1), 'lr': optimizer.param_groups[0]['lr']})
            if not args.no_wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]['lr'], "epoch": epoch + i / len(train_loader)})


        avg_train_loss = running_loss / len(train_loader)
        
        acc, val_loss = validate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Acc = {acc:.4f}, Val Loss = {val_loss:.4f}")
        if not args.no_wandb:
            wandb.log({"val/accuracy": acc, "val/loss": val_loss, "epoch": epoch + 1})


        if acc > best_acc:
            best_acc = acc
            # --- Save fine-tuned model state_dict AND its parameters ---
            torch.save(model.state_dict(), args.save_as)
            
            # Load the pruning parameters from the original pruned model
            # This ensures the fine-tuned model's params.json reflects its pruned architecture
            finetuned_params_save_path = os.path.splitext(args.save_as)[0] + "_params.json"
            with open(finetuned_params_save_path, 'w') as f:
                json.dump(original_pruned_params, f)
            print(f"Fine-tuned model parameters saved to {finetuned_params_save_path}")
            print(f"New best model saved with Acc: {best_acc:.4f}")
            if not args.no_wandb:
                wandb.run.summary["best_val_accuracy"] = best_acc # Log best accuracy to summary

    print("\nFine-tuning complete.")
    print(f"Best validation accuracy: {best_acc:.4f}")

    # Final evaluation of the best model
    if os.path.exists(args.save_as):
        print("\nLoading best fine-tuned model for final evaluation...")
        # Load the fine-tuned model using its own saved params.json
        model_final_eval = safe_load_model(args.save_as, args.model_name, device, img_size)
        final_acc, final_loss = validate_model(model_final_eval, val_loader, device)
        
        if tp is not None:
            example_input_macs_params = torch.randn(1, 3, img_size, img_size).to(device)
            macs, params = tp.utils.count_ops_and_params(model_final_eval, example_input_macs_params)

            print("\n=== Final Evaluation Results (Fine-tuned) ===")
            print(f"Accuracy: {final_acc:.4f}")
            print(f"Loss: {final_loss:.4f}")
            print(f"MACs: {macs/1e9:.4f} G")
            print(f"Params: {params/1e6:.4f} M")
            if not args.no_wandb:
                wandb.log({
                    "final_accuracy": final_acc,
                    "final_loss": final_loss,
                    "final_macs_G": macs / 1e9,
                    "final_params_M": params / 1e6
                })
                wandb.run.summary["final_accuracy"] = final_acc # Log final accuracy to summary
        else:
            print("\n=== Final Evaluation Results (Fine-tuned) ===")
            print(f"Accuracy: {final_acc:.4f}")
            print(f"Loss: {final_loss:.4f}")
            print("Cannot calculate MACs/Params: torch_pruning not available.")
            if not args.no_wandb:
                wandb.log({
                    "final_accuracy": final_acc,
                    "final_loss": final_loss
                })
                wandb.run.summary["final_accuracy"] = final_acc # Log final accuracy to summary
    
    if not args.no_wandb:
        wandb.finish() # End wandb run

if __name__ == '__main__':
    main()