from utils.video import read_frames_decord
from models.modeling_captioners import AutoCaptioner
import os
import torch
import json

video_folder = "/home/dataset/CaReBench/videos"

video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']

video_files = [
    f for f in os.listdir(video_folder)
    if os.path.isfile(os.path.join(video_folder, f)) and os.path.splitext(f)[1].lower() in video_extensions
]

json_path = '/home/dataset/CaReBench/json/metadata.json'

with open(json_path, "r") as f:
    dataset = json.load(f)

captioner = AutoCaptioner.from_pretrained('/home/hg_models/CaRe-7B')

results = []

for idx, data in enumerate(dataset):
    video_path = os.path.join('/home/dataset/CaReBench/videos', data['video'])
    try:
        frames = read_frames_decord(video_path=video_path, num_frames=6)
        description = captioner.describe(frames.unsqueeze(0))

        results.append({
                "video_path": video_path,
                "text": description[0],
            })

        print(f"[{idx}] Success: {os.path.basename(video_path)}")

    except Exception as e:
        print(f"[{idx}] Failed: {os.path.basename(video_path)} - Error: {e}")

output_path = "/home/junseoklee/CaReBench/caption_results.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n총 {len(results)}개의 결과가 저장되었습니다: {output_path}")