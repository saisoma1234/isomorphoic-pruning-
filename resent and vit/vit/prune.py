import os
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch.nn as nn
import timm
import torch_pruning as tp
from tqdm import tqdm
import argparse
import torch.optim as optim
import json # Import json for saving parameters
import wandb # Import wandb for logging

def parse_args():
    parser = argparse.ArgumentParser(description='Prune timm Vision Transformer model with Taylor Importance')
    parser.add_argument('--model-name', default='deit_small_patch16_224', type=str,
                        help='Name of the timm Vision Transformer model (e.g., deit_small_patch16_224, vit_base_patch16_224).')
    parser.add_argument('--data-path', default='./data', type=str,
                        help='Path to the CIFAR-10 dataset.')
    parser.add_argument('--train-batch-size', default=32, type=int,
                        help='Batch size for training data loader (used for importance calculation).')
    parser.add_argument('--val-batch-size', default=128, type=int,
                        help='Batch size for validation data loader (used for initial evaluation).')
    parser.add_argument('--taylor-batchs', default=10, type=int,
                        help='Number of batches to use for Taylor importance calculation.')
    parser.add_argument('--pruning-ratio', default=0.5, type=float,
                        help='Overall pruning ratio for the model (e.g., 0.5 for 50% pruning).')
    parser.add_argument('--round-to', default=1, type=int,
                        help='Round pruning channels to this multiple (e.g., 1, 8).')
    parser.add_argument('--global-pruning', default=True, action='store_true',
                        help='Perform global pruning across all prunable layers.')
    parser.add_argument('--save-as', default='./pruned_vit.pth', type=str,
                        help='Path to save the pruned model state_dict.')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='Number of data loading workers.')
    parser.add_argument('--pruning-type', default='taylor', type=str, choices=['taylor'],
                        help='Type of importance score for pruning (taylor is the only option for this script).')
    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='pruning-vit-experiments',
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
        wandb.init(project=args.wandb_project, name=f"Prune_{args.model_name}_ratio{args.pruning_ratio}", config=args)
        

    # Create model
    model = timm.create_model(args.model_name, pretrained=True)
    
    # Infer input image size from model config, default to 224 if not found
    img_size = model.default_cfg.get('input_size', (3, 224, 224))[1]
    print(f"Inferred model input image size: {img_size}x{img_size}")

    train_loader, val_loader = prepare_cifar10(args.data_path, args.train_batch_size, args.val_batch_size, args.num_workers, img_size)
    
    # Adjust classifier head for CIFAR-10 (10 classes)
    if hasattr(model, 'head'):
        if hasattr(model.head, 'fc'):
            model.head.fc = nn.Linear(model.head.fc.in_features, 10)
        else:
            model.head = nn.Linear(model.head.in_features, 10)
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        print("Warning: Could not automatically detect and reset classifier. Please ensure your model's head is configured for 10 classes.")
    model.to(device)

    example_inputs = torch.randn(1, 3, img_size, img_size).to(device)
    
    imp = tp.importance.GroupTaylorImportance()

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    
    original_embedding_dim = None
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
        original_embedding_dim = model.patch_embed.proj.out_channels
    print(f"Original embedding dimension: {original_embedding_dim}")

    print(f"Initial model: {args.model_name}")
    print(f"Base MACs: {base_macs / 1e9:.2f} G, Params: {base_params / 1e6:.2f} M")
    if not args.no_wandb:
        wandb.log({"base_macs_G": base_macs / 1e9, "base_params_M": base_params / 1e6})


    acc_ori, loss_ori = validate_model(model, val_loader, device)
    print(f"Before pruning (pretrained, untrained on CIFAR-10): Acc = {acc_ori:.4f}, Loss = {loss_ori:.4f}")
    if not args.no_wandb:
        wandb.log({"acc_before_pruning": acc_ori, "loss_before_pruning": loss_ori})


    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    unwrapped_parameters = []
    if hasattr(model, 'cls_token'):
        unwrapped_parameters.append((model.cls_token, 0))
    if hasattr(model, 'pos_embed'):
        unwrapped_parameters.append((model.pos_embed, 1))

    ignored_layers = []
    if hasattr(model, 'head'):
        if hasattr(model.head, 'fc'):
            ignored_layers.append(model.head.fc)
        else:
            ignored_layers.append(model.head)
    elif hasattr(model, 'fc'):
        ignored_layers.append(model.fc)
    
    # Also ignore all LayerNorm layers, as pruning them is tricky and usually not done
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            ignored_layers.append(module)


    print(f"Ignored layers during pruning: {[n.__class__.__name__ for n in ignored_layers]}")
    print(f"Unwrapped parameters: {[p[0].__class__.__name__ for p in unwrapped_parameters] if unwrapped_parameters else 'None'}")

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        global_pruning=args.global_pruning,
        pruning_ratio=args.pruning_ratio,
        round_to=args.round_to,
        unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,
    )

    model.train()
    model.zero_grad()
    print(f"Calculating Taylor Importance scores over {args.taylor_batchs} batches...")
    for i, (x, y) in enumerate(train_loader):
        if i >= args.taylor_batchs:
            break
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        loss.backward()

    pruner.step()
    print("Model pruned!")

    # --- CRITICAL FIX: Re-initialize specific ViT layers after pruning ---
    new_embedding_dim = None
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
        new_embedding_dim = model.patch_embed.proj.out_channels
    
    if new_embedding_dim is None:
        print("ERROR: Could not determine new embedding dimension after pruning from patch_embed. This is critical.")
        if not args.no_wandb:
            wandb.finish(status="failed") # Log failure to wandb
        return

    print(f"Inferred new embedding dimension: {new_embedding_dim}")

    # Determine original num_heads
    num_heads = model.blocks[0].attn.num_heads 
    
    # If new_embedding_dim is not divisible by num_heads, adjust it.
    if new_embedding_dim % num_heads != 0:
        new_embedding_dim = (new_embedding_dim // num_heads) * num_heads
        print(f"Adjusted new_embedding_dim to be a multiple of num_heads ({num_heads}): {new_embedding_dim}")
    
    new_head_dim = new_embedding_dim // num_heads

    # Re-initialize positional embedding and class token if dimensions changed
    if hasattr(model, 'cls_token'):
        if model.cls_token.shape[2] != new_embedding_dim:
            old_cls_token_shape = model.cls_token.shape
            model.cls_token = nn.Parameter(torch.zeros(1, 1, new_embedding_dim)).to(device)
            nn.init.trunc_normal_(model.cls_token, std=.02)
            print(f"Resized cls_token from {old_cls_token_shape} to {model.cls_token.shape}")

    if hasattr(model, 'pos_embed'):
        if model.pos_embed.shape[2] != new_embedding_dim:
            old_pos_embed_shape = model.pos_embed.shape
            patch_size = model.patch_embed.proj.kernel_size[0]
            num_patches = (img_size // patch_size) * (img_size // patch_size)
            model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, new_embedding_dim)).to(device)
            nn.init.trunc_normal_(model.pos_embed, std=.02)
            print(f"Resized pos_embed from {old_pos_embed_shape} to {model.pos_embed.shape}")

    # --- Iterate through each block and re-initialize its internal layers ---
    # Store the actual hidden_features sizes for MLP layers from the pruned model
    # before re-initializing them to match the new_embedding_dim.
    # This is crucial because fc1.out_features (hidden_features) might have been pruned independently.
    block_mlp_hidden_features = []
    for block in model.blocks:
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc1'):
            block_mlp_hidden_features.append(block.mlp.fc1.out_features)
        else:
            block_mlp_hidden_features.append(None) # Placeholder if not found for some reason

    for i, block in enumerate(model.blocks):
        print(f"Re-initializing layers in block {i+1}/{len(model.blocks)}")
        
        # 1. Re-initialize Attention module
        attn = block.attn
        attn.qkv = nn.Linear(new_embedding_dim, new_embedding_dim * 3, bias=attn.qkv.bias is not None).to(device)
        nn.init.xavier_uniform_(attn.qkv.weight)
        if attn.qkv.bias is not None:
            nn.init.constant_(attn.qkv.bias, 0)

        attn.proj = nn.Linear(new_embedding_dim, new_embedding_dim, bias=attn.proj.bias is not None).to(device)
        nn.init.xavier_uniform_(attn.proj.weight)
        if attn.proj.bias is not None:
            nn.init.constant_(attn.proj.bias, 0)
        
        attn.num_heads = num_heads
        attn.head_dim = new_head_dim
        attn.scale = new_head_dim ** -0.5
        
        # 2. Re-initialize MLP module
        mlp = block.mlp
        # Use the actual hidden_features from the pruned model for fc1.out_features
        # and fc2.in_features.
        current_mlp_hidden_features = block_mlp_hidden_features[i]
        
        if current_mlp_hidden_features is None: # Fallback if not found
            print(f"Warning: MLP hidden features for block {i+1} not found, defaulting to 4 * new_embedding_dim.")
            current_mlp_hidden_features = 4 * new_embedding_dim # Default if not determined

        mlp.fc1 = nn.Linear(new_embedding_dim, current_mlp_hidden_features, bias=mlp.fc1.bias is not None).to(device)
        nn.init.xavier_uniform_(mlp.fc1.weight)
        if mlp.fc1.bias is not None:
            nn.init.normal_(mlp.fc1.bias, std=1e-6)

        mlp.fc2 = nn.Linear(current_mlp_hidden_features, new_embedding_dim, bias=mlp.fc2.bias is not None).to(device)
        nn.init.xavier_uniform_(mlp.fc2.weight)
        if mlp.fc2.bias is not None:
            nn.init.constant_(mlp.fc2.bias, 0)


    # Re-initialize the final classifier head
    if hasattr(model, 'head'):
        if hasattr(model.head, 'fc'):
            model.head.fc = nn.Linear(new_embedding_dim, 10).to(device)
        else:
            model.head = nn.Linear(new_embedding_dim, 10).to(device)
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(new_embedding_dim, 10).to(device)
    
    model.to(device)

    acc_pruned, loss_prned = validate_model(model, val_loader, device)
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)

    print(f"After pruning (without finetuning): Acc = {acc_pruned:.4f}, Loss = {loss_prned:.4f}")
    print(f"MACs: {base_macs / 1e9:.2f} G => {pruned_macs / 1e9:.2f} G ({pruned_macs/base_macs*100:.2f}%)")
    print(f"Params: {base_params / 1e6:.2f} M => {pruned_params / 1e6:.2f} M ({pruned_params/base_params*100:.2f}%)")
    
    if not args.no_wandb:
        wandb.log({
            "acc_after_pruning": acc_pruned,
            "loss_after_pruning": loss_prned,
            "pruned_macs_G": pruned_macs / 1e9,
            "pruned_params_M": pruned_params / 1e6,
            "macs_reduction_percent": (1 - pruned_macs / base_macs) * 100,
            "params_reduction_percent": (1 - pruned_params / base_params) * 100
        })

    # --- Save additional model parameters for loading ---
    model_params_to_save = {
        'new_embedding_dim': new_embedding_dim,
        'num_heads': num_heads, # Also save num_heads for finetune.py
        'mlp_hidden_features': block_mlp_hidden_features # Save the list of hidden features
    }
    
    # Get the base path for saving the state dict
    base_save_path = os.path.splitext(args.save_as)[0]
    params_save_path = base_save_path + "_params.json"

    with open(params_save_path, 'w') as f:
        json.dump(model_params_to_save, f)
    print(f"Pruned model parameters saved to {params_save_path}")

    torch.save(model.state_dict(), args.save_as)
    print(f"Pruned model state_dict saved to {args.save_as}")

    if not args.no_wandb:
        wandb.finish() # End wandb run

if __name__ == '__main__':
    main()