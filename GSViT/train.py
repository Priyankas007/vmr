import random, os, time
import cv2, torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from vmr_model import GSViT
from vmr_data import create_data_loader, visualize_batch
import argparse
from tqdm import tqdm
import wandb
from typing import Tuple, List, Optional
from datetime import datetime

def train_epoch(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: str, 
                epoch: int) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    iters_per_epoch = len(train_loader)

    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        batch_start_time = time.time()
        
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        visualize_batch((images.cpu(), labels.cpu()))
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        # Calculate batch statistics
        batch_loss = loss.item()
        running_loss += batch_loss * images.size(0)
        _, predicted = outputs.max(1)
        batch_correct = predicted.eq(labels).sum().item()
        correct += batch_correct
        total += labels.size(0)
        
        
        if i % 10 == 0: # log every 10 iters
            batch_time = time.time() - batch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to wandb (iteration level metrics)
            wandb.log({
                "batch/loss": batch_loss,
                "batch/accuracy": 100.0 * batch_correct / labels.size(0),
                "batch/learning_rate": current_lr,
                "batch/time": batch_time,
                "batch/running_loss": running_loss / total,
                "batch/running_accuracy": 100.0 * correct / total,
                "batch/iteration": i + epoch * iters_per_epoch
            })

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total
    epoch_time = time.time() - start_time

    return epoch_loss, epoch_accuracy

def validate_epoch(model: nn.Module, val_loader: torch.utils.data.DataLoader, 
                   loss_fn: nn.Module, device: str) -> Tuple[float, float, List, List]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total
    return epoch_loss, epoch_accuracy, all_predictions, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_pretrained', action='store_true', help='use pretrained weights')
    parser.add_argument('--finetune_mode', type=str, default='linear_probe', help='linear_probe or finetune')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=12, help='number of workers for data loaders')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate for model')
    parser.add_argument('--scheduler_gamma', type=float, default=0.95, help='gamma value for ExponentialLR scheduler')
    parser.add_argument('--data_root', type=str, default='/scratch/users/abhi1/vmr_surg/vmr_data', help='path to data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/', help='directory to save outputs')
    parser.add_argument('--checkpoint_path', type=str, default='GSViT.pkl', help='path to pretrained checkpoint')
    parser.add_argument('--num_classes', type=int, default=8, help='number of surgical steps')
    args = parser.parse_args()
    assert args.finetune_mode in ['finetune', 'linear_probe']

    np.random.seed(0)
    torch.manual_seed(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # model
    gsvit = GSViT(
        num_classes=args.num_classes, 
        finetune_mode=args.finetune_mode,
        dropout=args.dropout
    )
    gsvit = gsvit.to(device)

    # load pretrained weights
    if args.use_pretrained and args.checkpoint_path:
        load_result = gsvit.load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=False)
        print(f"Successfully loaded pretrained weights from {args.checkpoint_path}")
        print(f"Missing keys: {len(load_result.missing_keys)}")
        print(f'First few missing keys: {list(load_result.missing_keys)[:10]}')
        print(f"Unexpected keys: {len(load_result.unexpected_keys)}")
        print(f'First few unexpected keys: {list(load_result.unexpected_keys)[:10]}')
    else:
        print("Not using pretrained weights")
    
    # data loaders
    train_loader = create_data_loader(
        data_root=args.data_root,
        split = 'train',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    val_loader = create_data_loader(
        data_root=args.data_root,
        split = 'val',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer
    print(f"Using initial learning rate: {args.learning_rate}")
    optimizer = torch.optim.Adam(gsvit.parameters(), lr=args.learning_rate)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

    # init experiment
    run_name = f"GSViT_{args.finetune_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="surgical_step_recognition",
        name=run_name,
        config={
            **vars(args),
            'model_type': 'GSViT',
            'optimizer': 'Adam',
            'scheduler': 'ExponentialLR',
            'scheduler_gamma': args.scheduler_gamma,
            'architecture': {
                'backbone': 'EfficientViT_M5',
                'classifier_dims': [384, 2048, 512, args.num_classes],
                'dropout': args.dropout
            }
        }
    )

    # make checkpoint directory
    os.makedirs(os.path.join(args.output_dir, run_name, 'checkpoints'), exist_ok=True)


    # log initial validation metrics before training
    print("Evaluating initial model performance...")
    initial_val_loss, initial_val_accuracy, initial_val_preds, initial_val_labels = validate_epoch(
        gsvit, val_loader, loss_fn, device
    )
    wandb.log({
        "epoch/number": -1,
        "epoch/val_loss": initial_val_loss,
        "epoch/val_accuracy": initial_val_accuracy,
        "initial/val_loss": initial_val_loss,
        "initial/val_accuracy": initial_val_accuracy
    })
    print(f"Initial validation - Loss: {initial_val_loss:.4f}, Accuracy: {initial_val_accuracy:.2f}%")

    best_val_accuracy = initial_val_accuracy
    best_val_loss = initial_val_loss
        
    # Train loop
    start_time = time.time()
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_epoch(gsvit, train_loader, optimizer, loss_fn, device, epoch)
        val_loss, val_accuracy, val_preds, val_labels = validate_epoch(gsvit, val_loader, loss_fn, device)

        # epoch-level logging remains the same
        wandb.log({
            "epoch/number": epoch,
            "epoch/train_loss": train_loss,
            "epoch/train_accuracy": train_accuracy,
            "epoch/val_loss": val_loss,
            "epoch/val_accuracy": val_accuracy,
            "epoch/learning_rate": scheduler.get_last_lr()[0],
        })

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_loss < best_val_loss):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            try:
                save_path = os.path.join(args.output_dir, run_name, f"best_model.pth")
                torch.save(gsvit.state_dict(), save_path)
                print(f"Saved best model! Val Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.4f}")
            except Exception as e:
                print(f"Error saving best model: {e}")

        # Save checkpoint every 4 epochs
        if (epoch) % 4 == 0:
            try:
                save_path = os.path.join(args.output_dir, run_name, f"checkpoint_epoch_{epoch}.pth")
                torch.save(gsvit.state_dict(), save_path)
                print(f"Saved checkpoint for epoch {epoch}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

        scheduler.step()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/3600:.2f} hours")


    print("Testing...")

    gsvit_best = GSViT(
        num_classes=args.num_classes, 
        finetune_mode=args.finetune_mode,
        dropout=args.dropout
    )
    gsvit_best = gsvit_best.to(device)

    # load pretrained weights
    best_model_path = os.path.join(args.output_dir, run_name, f"best_model.pth")

    load_best_result = gsvit_best.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
    print(f"Successfully loaded pretrained weights from {best_model_path}")
    print(f"Missing keys: {len(load_best_result.missing_keys)}")
    print(f'First few missing keys: {list(load_best_result.missing_keys)[:10]}')
    print(f"Unexpected keys: {len(load_best_result.unexpected_keys)}")
    print(f'First few unexpected keys: {list(load_best_result.unexpected_keys)[:10]}')

    test_loader = create_data_loader(
        data_root=args.data_root,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    loss_fn = nn.CrossEntropyLoss()

    test_loss, test_accuracy, test_preds, test_labels = validate_epoch(gsvit_best, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    wandb.log({
        "test/loss": test_loss,
        "test/accuracy": test_accuracy
    })
    wandb.finish()
