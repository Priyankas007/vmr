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
from data import create_data_loader
from tcn import CausalDinoTCN
from tqdm import tqdm

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder', type=str, default="CausalDinoTCN_finetune_20241209_031431",
                      help='Path to the folder where best_model.pth is saved')
    parser.add_argument('--data_root', type=str, default='/scratch/users/abhi1/vmr_surg/vmr_data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for eval')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader')
    parser.add_argument('--sequence_length', type=int, default=16, help='Sequence length for TCN')
    parser.add_argument('--num_levels', type=int, default=3, help='Number of levels for TCN')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for TCN')
    parser.add_argument('--output_metrics', type=str, default='metrics.json', help='Name of the metrics output file')
    parser.add_argument('--output_confusion', type=str, default='confusion_matrix.png', help='Name of the confusion matrix image file')
    parser.add_argument('--model_name', type=str, default='dinov2_vits14', help='Name of the DINOv2 model variant')
    parser.add_argument('--factor', type=int, default=1, help='Factor for data loader, as per training')

    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load best model
    model_path = os.path.join('outputs', args.run_folder, 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    model = CausalDinoTCN(
        num_classes=args.num_classes,
        sequence_length=args.sequence_length,
        num_levels = args.num_levels,
        num_channels=64,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        model_name=args.model_name,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_loader = create_data_loader(
        data_root=args.data_root,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
        factor=args.factor
    )

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

    id_to_step = {
        0: 'START', 1: 'SD', 2: 'PF', 3: 'AD',
        4: 'DMF', 5: 'SMF', 6: 'PC', 7: 'END'
    }
    class_labels = [id_to_step[i] for i in range(args.num_classes)]

    # plot confusion matrix and get accuracies
    print("Confusion Matrix:\n", cm)
    print(type(cm))
    print(cm.shape)
    class_accuracies = plot_confusion_matrices(cm, class_labels, save_dir=os.path.join('outputs', args.run_folder))

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
    metrics_path = os.path.join('outputs', args.run_folder, args.output_metrics)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved metrics with accuracies to {metrics_path}")