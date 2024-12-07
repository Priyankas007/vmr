import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import SurgicalStepDataset, create_data_loader
import argparse
import time
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from models.dinov2_classifier import DINOv2ClassifierScratch, DINOv2ClassifierLinearProbe, DINOv2ClassifierFinetune, DINOv2ClassifierFinetuneLoRA

def train_model(train_loader, val_loader, model, device, labels, epochs=10, lr=1e-4, experiment_type=None):
    """
    Train the model and validate after each epoch.

    Parameters:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        model (nn.Module): The DINOv2-based classifier.
        device (torch.device): Device to use (CPU or GPU).
        labels (list): List of phase labels.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
    """
    criterion = nn.CrossEntropyLoss()
    if experiment_type == 'finetune':
        optimizer = torch.optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
        ])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc='Training', leave=False)
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)
            
            # forward pass
            outputs = model(images)
            print(f"logits shape: {outputs.shape}")
            loss = criterion(outputs, labels)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            train_progress.set_postfix({"Batch Loss": loss.item()})

        # Validation phase
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Validation Accuracy: {correct / total:.2f}")

def test_model(test_loader, model, device):
    model.eval()  # Set model to evaluation mode
    total, correct = 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2f}")

def get_dinov2_model(model_name, experiment_type, num_classes):
    if experiment_type == 'finetune':
        model = DINOv2ClassifierFinetune(model_name, num_classes)
    elif experiment_type == 'scratch':
        model = DINOv2ClassifierScratch(model_name, num_classes)
    elif experiment_type == 'linearProbe':
        model = DINOv2ClassifierLinearProbe(model_name, num_classes)
    elif experiment_type == 'LoRa':
        model = DINOv2ClassifierFinetuneLoRA(model_name, num_classes)
    else:
        raise ValueError(f"Invalid experiment_type: {experiment_type}. Choose from ['finetune', 'scratch', 'linearProbe', 'LoRa'].")
    return model



if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a DINOv2-based surgical phase classifier.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the DINOv2 model (e.g., vits14, vitb14, vitl14, vitg14).")
    parser.add_argument("--experiment_type", type=str, required=True, help="Type of experiment")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training.")
    args = parser.parse_args()
    
    
    # Paths
    root_dir = "/scratch/users/shrestp/vmr_cs286/DINOv2/vmr_data"
    save_dir = "/scratch/users/shrestp/vmr_cs286/DINOv2/outputs/checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size=args.batch_size
    lr=5e-5

    # Phase labels
    phase_labels = ['AD', 'DMF', 'END', 'PC', 'PF', 'SD', 'SMF', 'START']

    # Datasets and DataLoaders
    train_loader = create_data_loader(root_dir, split='train', batch_size=batch_size)
    val_loader = create_data_loader(root_dir, split='val', batch_size=batch_size)
    test_loader = create_data_loader(root_dir, split='test', batch_size=batch_size)
    # DEBUGGING
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}, Images shape: {images.shape}, Labels shape: {labels.shape}")
        break  # Exit after processing one batch
    
    # Model
    model = get_dinov2_model(args.model_name, args.experiment_type,  num_classes=len(phase_labels)).to(device)

    # Train the model
    start_time = time.time()
    train_model(train_loader, val_loader, model, device, labels=phase_labels, epochs=args.epochs, lr=args.lr, experiment_type=args.experiment_type)
    end_time = time.time()
    
    training_time = end_time - start_time
    
    checkpoint = {
    "model_state_dict": model.state_dict(),
    "training_time": training_time,
    "epoch": args.epochs
    }
    
    # Save the model
    save_name = args.model_name + "_" + args.experiment_type + ".pth"
    save_path = os.path.join(save_dir, save_name)
    torch.save(checkpoint, save_path)
    print(f"Model save to {save_path}.")
    
    # Test the model
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded for testing.")
    test_model(test_loader, model, device)
    
