import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import SurgicalStepDataset, create_data_loader
import argparse
from configs import configs
import time
import wandb
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from models.dinov2_classifier import DINOv2ClassifierScratch, DINOv2ClassifierLinearProbe, DINOv2ClassifierFinetune, DINOv2ClassifierFinetuneLoRA

def train_model(train_loader, val_loader, model, device, labels, epochs=10, lr=1e-4, experiment_type=None, patience=3):
    """
    Train the model and validate after each epoch.

    Parameters:
        ...
    """
    criterion = nn.CrossEntropyLoss()
    if experiment_type == 'finetune':
        optimizer = torch.optim.AdamW([
            {'params': model.feature_extractor.parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    best_val_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc='Training', leave=False)
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_progress.set_postfix({"Batch Loss": loss.item()})

        # Log training loss to W&B
        wandb.log({"train_loss": running_loss / len(train_loader)})

        # Validation phase
        model.eval()
        total, correct = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total

        # Log validation metrics to W&B
        wandb.log({"val_loss": val_loss / len(val_loader), "val_accuracy": val_accuracy})

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            wandb.run.summary["best_val_accuracy"] = best_val_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded the best model state based on validation accuracy.")

    return model, best_val_accuracy

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
    wandb.log({"test_accuracy": accuracy})
    print(f"Test Accuracy: {accuracy:.4f}")

def get_dinov2_model(model_name, experiment_type, num_classes, lora_rank=None, lora_alpha=None):
    if experiment_type == 'finetune':
        model = DINOv2ClassifierFinetune(model_name, num_classes)
    elif experiment_type == 'scratch':
        model = DINOv2ClassifierScratch(model_name, num_classes)
    elif experiment_type == 'linearProbe':
        model = DINOv2ClassifierLinearProbe(model_name, num_classes)
    elif experiment_type == 'LoRa':
        model = DINOv2ClassifierFinetuneLoRA(model_name=model_name, 
                                            num_classes=num_classes,
                                            r=lora_rank,
                                            alpha=lora_alpha)
    else:
        raise ValueError(f"Invalid experiment_type: {experiment_type}. Choose from ['finetune', 'scratch', 'linearProbe', 'LoRa'].")
    return model

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a DINOv2-based surgical phase classifier.")
    parser.add_argument("--config_idx", type=int, required=True, help="Index of the hyperparameter configuration.")
    args = parser.parse_args()
    
    # Paths
    root_dir = "/scratch/users/shrestp/vmr_cs286/DINOv2/vmr_data"
    save_dir = "/scratch/users/shrestp/vmr_cs286/DINOv2/outputs/checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Phase labels
    phase_labels = ['AD', 'DMF', 'END', 'PC', 'PF', 'SD', 'SMF', 'START']
    print(args.config_idx)

    if args.config_idx is not None:
        config = configs[args.config_idx]
        rank = config["rank"]
        alpha = config["alpha"]
        batch_size = config["batch_size"]
        lr = config["lr"]
        epochs = config["epochs"]
        model_name = config["model_name"]
        experiment_type = config["experiment_type"]
    else:
        raise ValueError("Invalid configuration index.")

    # Initialize W&B
    wandb.init(
        project="surgical_phase_classification",
        name=f"rank{rank}_alpha{alpha}_lr{lr}_bs{batch_size}_{model_name}_{experiment_type}",
        config={
            "model_name": model_name,
            "experiment_type": experiment_type,
            "rank": rank,
            "alpha": alpha,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
        }
    )
    
    # Datasets and DataLoaders
    train_loader = create_data_loader(root_dir, split='train', batch_size=batch_size)
    val_loader = create_data_loader(root_dir, split='val', batch_size=batch_size)
    test_loader = create_data_loader(root_dir, split='test', batch_size=batch_size)
    
    # Model
    model = get_dinov2_model(
        model_name=model_name, 
        experiment_type=experiment_type,
        lora_rank=rank,
        lora_alpha=alpha,  
        num_classes=len(phase_labels)).to(device)

    # Train the model
    start_time = time.time()
    model, val_accuracy = train_model(train_loader, 
                                      val_loader, 
                                      model, 
                                      device, 
                                      labels=phase_labels, 
                                      epochs=epochs, 
                                      lr=lr, 
                                      experiment_type=experiment_type,
                                      patience=3)
    end_time = time.time()
    wandb.run.summary["training_time"] = end_time - start_time
    
    # Save the model
    if experiment_type == "LoRa":
        save_path = f"{save_dir}/{model_name}_rank{rank}_alpha{alpha}_lr{lr}_bs{batch_size}.pth"
    else:
        save_path = f"{save_dir}/{model_name}_{experiment_type}_lr{lr}_bs{batch_size}.pth"
    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)
    print(f"Model saved to {save_path}.")
    
    # Test the model
    test_model(test_loader, model, device)
    
    # Finish W&B run
    wandb.finish()
