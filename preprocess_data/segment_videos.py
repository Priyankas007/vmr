import os
import subprocess
import cv2
import pandas as pd
from tqdm import tqdm

def get_frame_rate(video_file):
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_file}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def timecode_to_ffmpeg_format(timecode, fps):
    hours, minutes, seconds, frames = map(int, timecode.split(':'))
    frames_in_seconds = frames / fps
    return f"{hours:02}:{minutes:02}:{seconds:02}.{int(frames_in_seconds * 100):02}"

def process_video_with_ffmpeg(input_file, output_file, start_timecode, end_timecode):
    
    if start_timecode == end_timecode:
        print(f"Skipping processing for {input_file}. Start and End timecodes are the same: {start_timecode}")
        return
    
    try:
        frame_rate = get_frame_rate(input_file)
        start_time = timecode_to_ffmpeg_format(start_timecode, frame_rate)
        end_time = timecode_to_ffmpeg_format(end_timecode, frame_rate)
        ffmpeg_command = [
            'ffmpeg',
            '-i', input_file,
            '-ss', start_time,
            '-to', end_time,
            '-c', 'copy',
            output_file
        ]
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed for file {input_file} with error: {e}")
    except Exception as e:
        print(f"Error processing video {input_file}: {e}")


metadata_dir_path = '/scratch/groups/syyeung/vmr_dataset/metadata'
video_dir_path = '/scratch/groups/syyeung/vmr_dataset/videos'

#####
# REPLACE WITH VIDEO/METADATA NAMES TO PROCESS
#####
metadata_names = sorted(["#4.csv", "#5.csv", "#6.csv", "#8.csv"])
video_names = sorted(["#4.mp4", "#5.mp4", "#6.mov", "#8.mpg"])

assert [x.split('.')[0] for x in metadata_names] == [x.split('.')[0] for x in video_names]

sections_original = ['START', 'SD1', 'SD2', 'PF1', 'PF2', 'AD1', 'AD2', 'DMF1', 'DMF2', 'SMF1', 'SMF2', 'PC1', 'PC2', 'END']
sections = ['START', 'SD', 'PF', 'AD', 'DMF', 'SMF', 'PC', 'END']

# create new video sections directories
for section in sections:
    os.makedirs(f'/scratch/groups/syyeung/vmr_dataset/video_sections/{section}', exist_ok=True)

print(f"Processing the following videos: {video_names}")

# process videos
for section in tqdm(sections, desc='Processing videos', position=0):
    for file in tqdm(video_names, desc=f'Processing {section}', position=1, leave=False):

        filename, extension = file.split('.')

        metadata = pd.read_csv(f'{metadata_dir_path}/{filename}.csv')
        metadata['Comment'] = metadata['Comment'].str.strip().str.upper()

        if section == 'START':
            section_start_timecode = metadata[metadata['Comment']  == 'START']['Timecode'].values[0]
            section_end_timecode = metadata[metadata['Comment']  == 'SD1']['Timecode'].values[0]
        elif section == 'END':
            section_start_timecode = metadata[metadata['Comment']  == 'PC2']['Timecode'].values[0]
            section_end_timecode = metadata[metadata['Comment']  == 'END']['Timecode'].values[0]
        else:
            section_start_timecode = metadata[metadata['Comment']  == f'{section}1']['Timecode'].values[0]
            section_end_timecode = metadata[metadata['Comment'] == f'{section}2']['Timecode'].values[0]

        input_file = f'{video_dir_path}/{filename}.{extension}'
        output_file = f'/scratch/groups/syyeung/vmr_dataset/video_sections/{section}/{filename}_{section}.{extension}'

        process_video_with_ffmpeg(input_file, output_file, section_start_timecode, section_end_timecode)
