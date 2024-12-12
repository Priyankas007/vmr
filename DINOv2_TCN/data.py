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
    (frames_tensor, label) -> frames_tensor is (T, C, H, W).
    """
    
    def __init__(self, root_dir, split='train', transform=None, sequence_length=16, factor=1):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.sequence_length = sequence_length
        self.step_id = {
            'START': 0, 'SD': 1, 'PF': 2, 'AD': 3,
            'DMF': 4, 'SMF': 5, 'PC': 6, 'END': 7
        }
        
        self.samples = []
        
        for step, step_label in self.step_id.items():
            step_dir = self.root_dir / step
            if not step_dir.exists():
                continue

            for video_folder in step_dir.glob("*_frames"):
                # print(f'Processing {video_folder}')
                frame_paths = list(video_folder.glob("*.png"))
                frame_paths = sorted(frame_paths, key=lambda x: x.name)
                
                num_frames = len(frame_paths)
                if num_frames >= self.sequence_length:
                    step_interval = self.sequence_length // factor if split == 'train' else self.sequence_length
        
                    for start_idx in range(0, num_frames - self.sequence_length + 1, step_interval):
                        seq_frames = frame_paths[start_idx:start_idx+self.sequence_length]
                        self.samples.append((seq_frames, step_label))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq_frame_paths, label = self.samples[idx]

        frames = []
        for frame_path in seq_frame_paths:
            image = cv2.imread(str(frame_path))  # BGR format
            image = self.preprocess_image(image)
            frames.append(image)
        
        frames_tensor = torch.stack(frames, dim=0)
        
        return frames_tensor, label
    
    def preprocess_image(self, image):
        # resize shorter side to 224 and center crop
        h, w = image.shape[:2]
        scale = 224 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        start_h = (new_h - 224) // 2
        start_w = (new_w - 224) // 2
        image = image[start_h:start_h + 224, start_w:start_w + 224]

        # convert to tensor and normalize
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # HWC to CHW
        image = image / 255.0  # Normalize to [0, 1]

        image = image[[2, 1, 0], :, :]  # BGR to RGB

        if self.transform:
            image = self.transform(image)

        return image

def create_data_loader(data_root, split='train', batch_size=128, num_workers=4, sequence_length=16, factor=1):
    assert split in ['train', 'val', 'test']
    
    dataset = SurgicalStepDataset(
        root_dir=data_root, 
        split=split, 
        sequence_length=sequence_length,
        factor=factor
    )
    print(f"Creating {split} dataset with {len(dataset)} sequences")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"Created data loader for {split} split with batch_size={batch_size}, sequence_length={sequence_length}")
    return loader


def visualize_batch(batch, save_path='batch_visualization.png'):
    images, labels = batch
    B, T, C, H, W = images.shape
    
    id_to_step = {
        0: 'START', 1: 'SD', 2: 'PF', 3: 'AD',
        4: 'DMF', 5: 'SMF', 6: 'PC', 7: 'END'
    }
    num_samples_to_show = min(B, 4)

    fig, axes = plt.subplots(num_samples_to_show, T, figsize=(3*T, 3*num_samples_to_show))

    if num_samples_to_show == 1:
        axes = [axes]

    for i in range(num_samples_to_show):
        seq_frames = images[i]  # [T, C, H, W]
        label_str = id_to_step[labels[i].item()]

        for t in range(T):
            img = seq_frames[t].permute(1, 2, 0).numpy()
            axes[i][t].imshow(img)
            axes[i][t].axis('off')
            axes[i][t].set_title(f'{label_str}\nFrame {t+1}/{T}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_sequence_as_images(batch, output_folder='vis_one_batch'):
    images, labels = batch
    B, T, C, H, W = images.shape

    if B != 1:
        raise ValueError("Batch size must be 1 for this function.")
    
    os.makedirs(output_folder, exist_ok=True)
    
    id_to_step = {
        0: 'START', 1: 'SD', 2: 'PF', 3: 'AD',
        4: 'DMF', 5: 'SMF', 6: 'PC', 7: 'END'
    }
    
    seq_frames = images[0]  # [T, C, H, W]
    label_str = id_to_step[labels[0].item()]

    for t in range(T):
        img = seq_frames[t].permute(1, 2, 0).numpy()  # [H, W, C]
        img = (img * 255).astype(np.uint8)

        frame_filename = os.path.join(output_folder, f"{label_str}_frame_{t+1:02d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved: {frame_filename}")


if __name__ == "__main__":
    train_loader = create_data_loader(
        data_root='/scratch/users/abhi1/vmr_surg/vmr_data',
        split='train',
        batch_size=1,
        num_workers=12,
        sequence_length=16
    )
    print(f"Number of batches: {len(train_loader)}")
    for i, (images, labels) in enumerate(train_loader):
        print("Batch of images shape:", images.shape)  # should be [B, T, C, H, W]
        print("Batch of labels shape:", labels.shape)
        print("Labels:", labels)
        visualize_batch((images, labels))
        save_sequence_as_images((images, labels), output_folder='vis_one_batch')
        break