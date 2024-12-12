import os
import shutil
import random
from multiprocessing import Pool

def copy_frames(video_id,
                split_name,
                source_root,
                target_root,
                steps):

    for step in steps:
        frames_dir = os.path.join(source_root, step, f"#{video_id}_{step}_frames")
        if os.path.exists(frames_dir):
            target_dir = os.path.join(target_root, split_name, step, f"#{video_id}_{step}_frames")
            shutil.copytree(frames_dir, target_dir, dirs_exist_ok=True)
    print(f"Completed copying frames for video ID {video_id} to {split_name}.")

def organize_data_splits(
        source_root,
        target_root,
        train_ratio = 0.8,
        val_ratio = 0.1,
        test_ratio = 0.1,
        random_seed = 42,
        num_workers = 4
):
    random.seed(random_seed)

    # creaet dirs
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(target_root, split), exist_ok=True)
    
    # get unique video IDs across all steps
    video_ids = set()
    steps = ['START', 'SD', 'PF', 'AD', 'DMF', 'SMF', 'PC', 'END']
    for step in steps:
        step_dir = os.path.join(source_root, step)
        if not os.path.exists(step_dir):
            continue
        for item in os.listdir(step_dir):
            if '_frames' in item:
                video_id = item.split('_')[0].replace('#', '')
                video_ids.add(video_id)
    
    video_ids = sorted(list(video_ids))
    random.shuffle(video_ids)

    print(f"Found {len(video_ids)} unique video IDs across all steps.")
    print(f"Video IDs: {video_ids}")    

    n_videos = len(video_ids)
    n_train = int(n_videos * train_ratio)
    n_val = int(n_videos * val_ratio)
    
    train_ids = video_ids[:n_train]
    val_ids = video_ids[n_train:n_train + n_val]
    test_ids = video_ids[n_train + n_val:]

    print(f"Data split complete:")
    print(f"Training videos: {len(train_ids)}")
    print(f'Train IDs: {sorted(train_ids)}')
    print(f"Validation videos: {len(val_ids)}")
    print(f'Val IDs: {sorted(val_ids)}')
    print(f"Test videos: {len(test_ids)}")
    print(f'Test IDs: {sorted(test_ids)}')
    
    # create split directories with step subdirectories
    for split in splits:
        for step in steps:
            os.makedirs(os.path.join(target_root, split, step), exist_ok=True)
    
    # multi-processing
    tasks = []
    for video_id in train_ids:
        tasks.append((video_id, 'train', source_root, target_root, steps))
    for video_id in val_ids:
        tasks.append((video_id, 'val', source_root, target_root, steps))
    for video_id in test_ids:
        tasks.append((video_id, 'test', source_root, target_root, steps))

    with Pool(processes=num_workers) as pool:
        pool.starmap(copy_frames, tasks)

def check_data_integrity(source_root, target_root, steps):
    source_frames = []
    target_frames = {'train': [], 'val': [], 'test': []}

    # find all frames in source_root
    for step in steps:
        step_dir = os.path.join(source_root, step)
        if os.path.exists(step_dir):
            directories = [d for d in os.listdir(step_dir) if os.path.isdir(os.path.join(step_dir, d)) and d.endswith('_frames')]
            for directory in directories:
                directory_path = os.path.join(step_dir, directory)
                files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
                source_frames.extend(files)
    # find all frames in target_root
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(target_root, split)
        for step in steps:
            step_dir = os.path.join(split_dir, step)
            if os.path.exists(step_dir):
                directories = [d for d in os.listdir(step_dir) if os.path.isdir(os.path.join(step_dir, d)) and d.endswith('_frames')]
                for directory in directories:
                    directory_path = os.path.join(step_dir, directory)
                    files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
                    target_frames[split].extend(files)

    all_target_frames = sum(target_frames.values(), [])

    #compare them sorted
    source_frames.sort()
    all_target_frames.sort()

    if source_frames != all_target_frames:
        missing_in_target = set(source_frames) - set(all_target_frames)
        extra_in_target = set(all_target_frames) - set(source_frames)

        print(f"First few missing frames: {list(missing_in_target)[:10]}")
        print(f"First few extra frames: {list(extra_in_target)[:10]}")

        raise ValueError("Data mismatch: Frames in the source and target directories are not identical.")

    print("Data integrity check passed: All frames match between source and target.")
    print(f"Total frames in source: {len(source_frames)}")
    print(f"Total frames in target: {len(all_target_frames)}")
    print(f"Frames in each split:")
    for split, frames in target_frames.items():
        print(f"  {split.capitalize()}: {len(frames)}")


if __name__ == '__main__':
    organize_data_splits(
        source_root='/scratch/groups/syyeung/vmr_dataset/video_sections',
        target_root='/scratch/users/abhi1/vmr_surg/vmr_data',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
        num_workers=8
    )
    check_data_integrity(
        source_root='/scratch/groups/syyeung/vmr_dataset/video_sections',
        target_root='/scratch/users/abhi1/vmr_surg/vmr_data',
        steps=['START', 'SD', 'PF', 'AD', 'DMF', 'SMF', 'PC', 'END']
    )