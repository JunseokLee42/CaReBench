from utils.video import read_frames_decord
from models.modeling_encoders import AutoEncoder
from torch.nn.functional import cosine_similarity
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

os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = torch.device("cuda")
encoder = AutoEncoder.from_pretrained('/home/hg_models/CaRe-7B', device_map='auto')

results = []

for idx, data in enumerate(dataset):
    video_path = os.path.join('/home/dataset/CaReBench/videos', data['video'])
    text = data['caption']
    try:
        frames = read_frames_decord(video_path=video_path, num_frames=2)
        vision_emb = encoder.encode_vision(frames.unsqueeze(0))
        text_emb = encoder.encode_text(text)
    
        cos_sim = cosine_similarity(vision_emb, text_emb).item()

        results.append({
            "video_path": video_path,
            "text": text,
            "vision_emb": vision_emb,
            "text_emb": text_emb,
            "cosine_similarity": cos_sim
        })

        print(f"[{idx}] Success: {os.path.basename(video_path)}")

    except Exception as e:
        print(f"[{idx}] Failed: {os.path.basename(video_path)} - Error: {e}")

output_path = "/home/junseoklee/CaReBench/embedding_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n총 {len(results)}개의 결과가 저장되었습니다: {output_path}")