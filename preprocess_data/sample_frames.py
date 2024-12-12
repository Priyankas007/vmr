import subprocess
import pandas as pd
import os
import cv2
from tqdm import tqdm

def sample_video(input_file, output_dir, rate=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        base_name = os.path.basename(input_file)
        filename, extension = base_name.split('.')
        output_pattern = os.path.join(output_dir, f"{filename}_frame_%04d.png")
        ffmpeg_command = [
            'ffmpeg', '-i', input_file, '-vf', f'fps={rate}', output_pattern
        ]
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed for file {input_file} with error: {e}")
    except Exception as e:
        print(f"Error processing video {input_file}: {e}")

def process_directories(video_sections_dir):
    sections = [d for d in os.listdir(video_sections_dir) if os.path.isdir(os.path.join(video_sections_dir, d))]

    for section in tqdm(sections, desc='Processing videos', position=0):
        section_dir_path = os.path.join(video_sections_dir, section)

        video_files = sorted([x for x in os.listdir(section_dir_path) if x in [f"#4_{section}.mp4", f"#5_{section}.mp4", f"#6_{section}.mov", f"#8_{section}.mpg"]])
        for video_file in tqdm(video_files, desc=f'Processing {section}', position=1, leave=False):
            input_file = os.path.join(section_dir_path, video_file)
            output_dir = os.path.join(section_dir_path, f'{video_file.split(".")[0]}_frames')
            sample_video(input_file, output_dir)

video_sections_dir = '/scratch/groups/syyeung/vmr_dataset/video_sections'
section_dirs = [d for d in os.listdir(video_sections_dir) if os.path.isdir(os.path.join(video_sections_dir, d))]
print(f'Processing the following sections: {section_dirs}')

sections = ['START', 'SD', 'PF', 'AD', 'DMF', 'SMF', 'PC', 'END']
assert sorted(section_dirs) == sorted(sections)

process_directories(video_sections_dir)


