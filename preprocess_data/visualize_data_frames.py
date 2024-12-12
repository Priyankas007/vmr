import os
import matplotlib.pyplot as plt
import json

root = '/scratch/groups/syyeung/vmr_dataset/video_sections'

section_frame_counts = {
    'START': 0,
    'SD': 0,
    'PF': 0,
    'AD': 0,
    'DMF': 0,
    'SMF': 0,
    'PC': 0,
    'END': 0
}

for step in ['START', 'SD', 'PF', 'AD', 'DMF', 'SMF', 'PC', 'END']:
    step_dir = os.path.join(root, step)
    if not os.path.exists(step_dir):
        print(f"Step {step} not found in {root}")
        continue
    for item in os.listdir(step_dir):
        if '_frames' in item:
            section_frame_counts[step] += len([x for x in os.listdir(os.path.join(step_dir, item)) if x.endswith('.png')])

print(json.dumps(section_frame_counts, indent=4))
print(f'Total frames: {sum(section_frame_counts.values())}')

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

bars = ax.bar(list(section_frame_counts.keys()), list(section_frame_counts.values()), width=0.8)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,  
        height,                             
        f'{int(height)}',                   
        ha='center',                        
        va='bottom',                        
        fontsize=10                         
    )

ax.set_xticks(list(section_frame_counts.keys()))
ax.set_xticklabels(list(section_frame_counts.keys()))
ax.set_ylabel('Number of frames')
ax.set_title('Distribution of frames across surgical steps')

fig.tight_layout()
fig.savefig('frames_distribution.png')
plt.close()
