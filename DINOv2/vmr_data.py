import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

class SurgicalStepDataset(Dataset):
    """
    Dataset for surgical step recognition adapted for DINOv2
    """    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.step_id = {
            'START': 0, 'SD': 1, 'PF': 2, 'AD': 3,
            'DMF': 4, 'SMF': 5, 'PC': 6, 'END': 7
        }
        
        # collect all frame paths and their labels
        self.samples = []
        for step in self.step_id.keys():
            step_dir = self.root_dir / step
            if not step_dir.exists():
                continue
                
            for video_folder in step_dir.glob("*_frames"):
                for frame_path in video_folder.glob("*.png"):
                    self.samples.append((str(frame_path), self.step_id[step]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_path, label = self.samples[idx]

        image = cv2.imread(frame_path) # BGR format
        
        # resize shorter side to 224 and center crop 
        h, w = image.shape[:2]
        scale = 224 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        start_h = (new_h - 224) // 2
        start_w = (new_w - 224) // 2
        image = image[start_h:start_h + 224, start_w:start_w + 224]
        
        # convert to tensor and normalize (THIS IS IMPLMEMENTED TO MATCH GSVIT PRETRAINING PIPEPLINE)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # HWC to CHW
        image = image / 255.0  # Normalize to [0, 1]

        image = image[[2, 1, 0], :, :]  # BGR to RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loader(data_root, split = 'train', batch_size = 128, num_workers = 4, transform=None):
    """Create data loader for a specific split."""

    assert split in ['train', 'val', 'test']
    
    
    dataset = SurgicalStepDataset(root_dir = data_root, split = split, transform=transform)
    print(f"Creating data set for {split} split with {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True if split == 'train' else False,
        num_workers = num_workers,
        pin_memory = True
    )
    print(f"Created data loader for {split} split")
    return loader


def visualize_batch(batch, save_path='batch_visualization.png'):
    """
    Visualize a batch of images with their labels.
    
    Assumes the batch is a tuple of (images, labels) from the Dataloader and batch size=4.
    Parameters:
        batch (tuple): batch of data (images, labels)
        save_path (str): Path to save the visualization image.
    """
    images, labels = batch
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    id_to_step = {
        0: 'START', 1: 'SD', 2: 'PF', 3: 'AD',
        4: 'DMF', 5: 'SMF', 6: 'PC', 7: 'END'
    }
    
    for idx in range(4):
        i, j = idx // 2, idx % 2
        img = images[idx].permute(1, 2, 0).numpy()  # CHW to HWC
        axes[i, j].imshow(img)
        axes[i, j].axis('off')
        label = id_to_step[labels[idx].item()]
        axes[i, j].set_title(f'{label}\n{tuple(img.shape)}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    
    train_loader = create_data_loader(
        data_root='/scratch/users/shrestp/vmr/vmr_data',
        split = 'train',
        batch_size=4,
        num_workers=4
    )
    print(f"Number of batches: {len(train_loader)}")
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape)
        print(labels.shape)
        visualize_batch((images, labels))
        break
