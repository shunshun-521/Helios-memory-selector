import json, os, sys
from pathlib import Path

video_dir = "/root/autodl-tmp/output_4_13/baseline"  # e.g. /root/autodl-fs/bolt_training_data/videos
prompt_file = "/root/autodl-tmp/Helios/example/video_prompts.txt"  # e.g. example/prompt_bolt_training.txt
output_json = "/root/autodl-tmp/output_4_13/baseline.json"  # e.g. /root/autodl-fs/bolt_training_data/bolt_train.json

with open(prompt_file) as f:
    prompts = [l.strip() for l in f if l.strip()]

entries = []
for mp4 in sorted(Path(video_dir).glob("*.mp4")):
    # 从文件名或索引匹配 prompt
    idx = int(mp4.stem.split("_")[0]) - 1  # 假设文件名为 0001_97_xxx.mp4
    if idx >= len(prompts):
        continue
    
    # 用 ffprobe 获取实际帧数（或从文件名解析）
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True
    )
    num_frames = int(result.stdout.strip()) if result.stdout.strip() else 97
    
    entries.append({
        "cut": [0, num_frames],
        "crop": [0, 832, 0, 480],
        "fps": 24.0,
        "num_frames": num_frames,
        "resolution": {"height": 480, "width": 832},
        "cap": [prompts[idx]],
        "path": f"root/autodl-tmp/output_4_13/baseline/{mp4.name}"
    })

with open(output_json, "w") as f:
    json.dump(entries, f, indent=2, ensure_ascii=False)
print(f"Generated {len(entries)} entries → {output_json}")