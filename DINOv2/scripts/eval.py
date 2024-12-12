import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from pathlib import Path
import torchvision.transforms as T
to_pil_image = T.ToPILImage()

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dataset import create_data_loader
#from tcn import CausalDinoTCN
#from models.dinov2_classifier import DINOv2ClassifierScratch, DINOv2ClassifierLinearProbe, DINOv2ClassifierFinetune, DINOv2ClassifierFinetuneLoRA
from tqdm import tqdm
from train import get_dinov2_model

def plot_confusion_matrices(cm, class_labels, save_dir, prefix=''):
    row_sums = cm.sum(axis=1)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # calculate per-class accuracies
    accuracies = {}
    for i, label in enumerate(class_labels):
        total_samples = cm[i].sum()
        if total_samples > 0:
            accuracy = cm[i, i] / total_samples
            accuracies[label] = {
                'accuracy': float(accuracy),
                'correct_samples': int(cm[i, i]),
                'total_samples': int(total_samples)
            }
        else:
            accuracies[label] = {
                'accuracy': 0.0,
                'correct_samples': 0,
                'total_samples': 0,
                'note': 'No samples'
            }

    # plot raw counts
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='viridis',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Number of Samples"},
        linewidths=0.5,
        linecolor='grey',
        square=True
    )
    plt.title('Confusion Matrix (Raw Counts)', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{prefix}confusion_matrix_raw.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved raw counts confusion matrix to {save_path}")

    # plot normalized matrix
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='viridis',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Percentage"},
        linewidths=0.5,
        linecolor='grey',
        square=True,
        vmin=0,
        vmax=1
    )
    plt.title('Confusion Matrix (Per-Class Accuracy)', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{prefix}confusion_matrix_normalized.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved normalized confusion matrix to {save_path}")

    print("\nPer-class Accuracy:")
    for label, stats in accuracies.items():
        if stats['total_samples'] > 0:
            print(f"{label}: {stats['accuracy']:.1%} ({stats['correct_samples']}/{stats['total_samples']} samples)")
        else:
            print(f"{label}: No samples")
            
    return accuracies

def plot_failure_cases(all_labels, all_preds, data_loader, id_to_step, target_classes, examples_per_class=3, save_dir='./outputs'):
    """
    Plot failure cases for specific ground truth classes with specified examples.

    Args:
        all_labels (list): Ground truth labels.
        all_preds (list): Predicted labels.
        data_loader (DataLoader): DataLoader for fetching image samples.
        id_to_step (dict): Mapping from class IDs to class names.
        target_classes (list): List of target ground truth classes to display (e.g., ["START", "PF"]).
        examples_per_class (int): Number of examples to display per class.
        save_dir (str): Directory to save the figure.
    """
    # Get class IDs for the target ground truth classes
    target_class_ids = [k for k, v in id_to_step.items() if v in target_classes]

    # Prepare a DataLoader iterator
    data_iter = iter(data_loader)

    # Initialize plot
    num_rows = len(target_classes)
    fig, axes = plt.subplots(num_rows, examples_per_class, figsize=(15, 5 * num_rows))
    axes = np.atleast_2d(axes)  # Ensure 2D for single row

    # Collect and plot failure cases for each target class
    for i, target_class_id in enumerate(target_class_ids):
        examples = []
        for img, label in data_iter:
            for idx, (image, gt_label) in enumerate(zip(img, label)):
                print("idx: ", idx)
                print("image: ", image)
                print("gt_label: ", gt_label)
                # Ensure ground truth is the target class and it is a failure case
                if gt_label.item() == target_class_id and len(examples) < examples_per_class:
                    pred_label = all_preds[idx]
                    if gt_label.item() != pred_label:
                        examples.append((image, id_to_step[gt_label.item()], id_to_step[pred_label]))
            if len(examples) >= examples_per_class:
                break

        # Plot examples for the current target class
        for j, (image, true_label, pred_label) in enumerate(examples):
            ax = axes[i, j]
            ax.imshow(image.permute(1, 2, 0).cpu().numpy())
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
            ax.axis("off")

    # Save and display the figure
    plt.tight_layout()
    save_path = os.path.join(save_dir, "specific_failure_cases.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Failure cases for {target_classes} saved to {save_path}")



def process_and_visualize_predictions(data_iter, model, id_to_step=None, device="cpu", save_dir="output_images", num_images_to_save=10):
    """
    Processes all images in the dataset, compares ground truth labels with predictions,
    and optionally saves or visualizes a subset of the images.

    Parameters:
        data_iter (DataLoader): DataLoader providing batches of images and labels.
        model (torch.nn.Module): Trained model to generate predictions.
        id_to_step (dict, optional): Mapping of numeric labels to string labels. Defaults to None.
        device (str): Device to use for inference ('cpu' or 'cuda'). Defaults to 'cpu'.
        save_dir (str): Directory to save visualized images. Defaults to 'output_images'.
        num_images_to_save (int): Number of images to save or visualize. Defaults to 10.

    Returns:
        list: A list of tuples (image, ground_truth_label, predicted_label).
    """
    model.eval()  # Set the model to evaluation mode
    all_predictions = []  # Store results
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    image_count = 0  # Counter to track saved images

    # Loop through the batches
    for img_batch, label_batch in data_iter:
        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)

        # Get predictions for the batch
        with torch.no_grad():
            outputs = model(img_batch)
            _, predicted_labels = torch.max(outputs, 1)

        # Iterate through individual images in the batch
        for idx, (image, gt_label, pred_label) in enumerate(zip(img_batch, label_batch, predicted_labels)):
            # Convert image to CPU for visualization
            image_np = image.cpu().permute(1, 2, 0).numpy()
            gt_label_id = gt_label.item()
            pred_label_id = pred_label.item()

            # Map numeric labels to string labels if mapping is provided
            gt_label_name = id_to_step[gt_label_id] if id_to_step else gt_label_id
            pred_label_name = id_to_step[pred_label_id] if id_to_step else pred_label_id

            # Append prediction results
            all_predictions.append((image, gt_label_id, pred_label_id))

            # Save or visualize the first `num_images_to_save` images
            if image_count < num_images_to_save:
                save_path = os.path.join(save_dir, f"image_{image_count}_gt_{gt_label_name}_pred_{pred_label_name}.png")
                plt.imshow(image_np)
                plt.axis("off")
                plt.title(f"GT: {gt_label_name}, Pred: {pred_label_name}")
                plt.savefig(save_path)
                plt.close()
                print(f"Saved image to {save_path}")
                image_count += 1

    return all_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="None",help='Path to best_model.pth')
    parser.add_argument('--save_folder', type=str, default="None",help='Folder to save metrics in')
    parser.add_argument('--data_root', type=str, default='/scratch/users/shrestp/vmr_cs286/DINOv2/vmr_data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for eval')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--output_metrics', type=str, default='metrics.json', help='Name of the metrics output file')
    parser.add_argument('--output_confusion', type=str, default='confusion_matrix.png', help='Name of the confusion matrix image file')
    parser.add_argument('--model_name', type=str, default='dinov2_vitb14', help='Name of the DINOv2 model variant')
    parser.add_argument('--experiment_type', type=str, default='scratch', help='Type of experiment')
    parser.add_argument('--rank', type=int, default='4', help='rank for LoRA')
    parser.add_argument('--alpha', type=int, default='4', help='alpha for LoRA')

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check if the folder exists
    if not os.path.exists(args.save_folder):
        # Create the folder if it doesn't exist
        os.makedirs(args.save_folder)
        print(f"Folder '{args.save_folder}' created successfully!")
    else:
        print(f"Folder '{args.save_folder}' already exists.")

    # load best model
    model_path = args.model_path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    model = get_dinov2_model(args.model_name, args.experiment_type, args.num_classes, lora_rank=args.rank, lora_alpha=args.alpha)
    model.to(device)

    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except: 
         model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    test_loader = create_data_loader(
        data_root=args.data_root,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    id_to_step = {
        0: 'START', 1: 'SD', 2: 'PF', 3: 'AD',
        4: 'DMF', 5: 'SMF', 6: 'PC', 7: 'END'
    }

    all_predictions = process_and_visualize_predictions(test_loader, model, id_to_step=id_to_step, device=device, save_dir="./", num_images_to_save=10)

    all_preds = []
    all_labels = []
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / total
    test_accuracy = 100.0 * correct / total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # get confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_labels = [id_to_step[i] for i in range(args.num_classes)]

    metrics_path = os.path.join('outputs', args.save_folder, args.output_metrics)
    if not os.path.exists(metrics_path):
        # plot confusion matrix and get accuracies
        print("Confusion Matrix:\n", cm)
        print(type(cm))
        print(cm.shape)
        class_accuracies = plot_confusion_matrices(cm, class_labels, save_dir=os.path.join('outputs', args.save_folder))

        # get classification report
        report = classification_report(all_labels, all_preds, labels=range(args.num_classes), 
                                    target_names=class_labels, output_dict=True)

        # create final metrics dictionary
        metrics_dict = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'per_class_accuracies': class_accuracies
        }

        # save metrics
        metrics_path = os.path.join('outputs', args.save_folder, args.output_metrics)
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Saved metrics with accuracies to {metrics_path}")

    failure_path = os.path.join('outputs', args.save_folder)
    plot_failure_cases(
    target_classes=['AD', 'DMF', 'END', 'PC', 'PF', 'SD', 'SMF', 'START'],
    all_labels=all_labels,
    all_preds=all_preds,
    data_loader=test_loader,
    id_to_step=id_to_step,
    examples_per_class=10,
    save_dir=failure_path)
    print(f"Saved failure cases to {failure_path}")