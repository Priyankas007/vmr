import random, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import create_data_loader
from tcn import CausalDinoTCN
import argparse
from tqdm import tqdm
import wandb
from typing import Tuple, List
from datetime import datetime
import numpy as np

def train_epoch(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: str, 
                epoch: int, log_interval: int = 50) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    iters_per_epoch = len(train_loader)

    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate batch statistics
        batch_loss = loss.item()
        running_loss += batch_loss * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        if (i + 1) % log_interval == 0:
            wandb.log({
                "batch/loss": batch_loss,
                "batch/accuracy": 100.0 * (predicted.eq(labels).sum().item() / labels.size(0))
            })

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total
    return epoch_loss, epoch_accuracy

def validate_epoch(model: nn.Module, val_loader: torch.utils.data.DataLoader, 
                   loss_fn: nn.Module, device: str) -> Tuple[float, float, List[int], List[int]]:
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
    parser.add_argument('--finetune_mode', type=str, default='finetune', help='linear_probe or finetune')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers for data loaders')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate for model')
    parser.add_argument('--data_root', type=str, default='/scratch/users/abhi1/vmr_surg/vmr_data', help='path to data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/', help='directory to save outputs')
    parser.add_argument('--num_classes', type=int, default=8, help='number of surgical steps')
    parser.add_argument('--sequence_length', type=int, default=16, help='sequence length for the TCN')
    parser.add_argument('--kernel_size', type=int, default=3, help='kernel size for TCN')
    parser.add_argument('--num_levels', type=int, default=3, help='number of TCN levels')
    parser.add_argument('--factor', type=int, default=2, help='factor for the number of sequences to use for training (1 is use 1/16 of the sequences, 16 is use all of the sequences)')
    args = parser.parse_args()

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    os.environ['PYTHONHASHSEED'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CausalDinoTCN(
        num_classes=args.num_classes,
        sequence_length=args.sequence_length,
        num_levels = args.num_levels,
        num_channels=64,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        model_name='dinov2_vits14',
        train_end_to_end = args.finetune_mode == 'finetune'
    ).to(device)

    train_loader = create_data_loader(
        data_root=args.data_root,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
        factor=args.factor
    )
    val_loader = create_data_loader(
        data_root=args.data_root,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length
    )
    loss_fn = nn.CrossEntropyLoss()

    if args.finetune_mode == 'finetune':
        optimizer = torch.optim.AdamW([
            {'params': model.feature_extractor.parameters(), 'lr': 1e-5},
            {'params': model.input_projection.parameters(), 'lr': args.learning_rate},
            {'params': model.temporal_layers.parameters(), 'lr': args.learning_rate},
            {'params': model.classifier.parameters(), 'lr': args.learning_rate}
        ])
        print(f"Learning rate for the feature extractor: {1e-5}")
        print(f"Learning rate for the TCN head: {args.learning_rate}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        print(f"Learning rate: {args.learning_rate}")

    run_name = f"CausalDinoTCN_{args.finetune_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="surgical_step_recognition",
        name=run_name,
        config={
            **vars(args),
            'model_type': 'CausalDinoTCN',
            'optimizer': 'AdamW',
        }
    )

    # initial validation metrics
    initial_val_loss, initial_val_accuracy, _, _ = validate_epoch(
        model, val_loader, loss_fn, device
    )
    wandb.log({
        "epoch/number": -1,
        "epoch/val_loss": initial_val_loss,
        "epoch/val_accuracy": initial_val_accuracy
    })
    print(f"Initial validation - Loss: {initial_val_loss:.4f}, Accuracy: {initial_val_accuracy:.2f}%")

    best_val_accuracy = initial_val_accuracy
    best_val_loss = initial_val_loss

    # training loop
    start_time = time.time()
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        val_loss, val_accuracy, _, _ = validate_epoch(model, val_loader, loss_fn, device)

        wandb.log({
            "epoch/number": epoch,
            "epoch/train_loss": train_loss,
            "epoch/train_accuracy": train_accuracy,
            "epoch/val_loss": val_loss,
            "epoch/val_accuracy": val_accuracy,
        })

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # save best model
        if val_accuracy >= best_val_accuracy and val_loss <= best_val_loss: # best accuracy and best loss
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            try:
                os.makedirs(os.path.join(args.output_dir, run_name), exist_ok=True)
                save_path = os.path.join(args.output_dir, run_name, f"best_model.pth")
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model! Val Accuracy: {val_accuracy:.2f}%, Val Loss: {val_loss:.4f}")
            except Exception as e:
                print(f"Error saving best model: {e}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time/3600:.2f} hours")

    # eval on test set

    model = CausalDinoTCN(
        num_classes=args.num_classes,
        sequence_length=args.sequence_length,
        num_levels = args.num_levels,
        num_channels=64,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        model_name='dinov2_vits14',
    ).to(device)

    best_model_path = os.path.join(args.output_dir, run_name, f"best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Successfully Loaded best model from {best_model_path}")

    test_loader = create_data_loader(
        data_root=args.data_root,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length
    )

    test_loss, test_accuracy, _, _ = validate_epoch(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    wandb.log({
        "test/loss": test_loss,
        "test/accuracy": test_accuracy
    })
    wandb.finish()
