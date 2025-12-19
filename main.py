import os
import re
from pathlib import Path
from tqdm import tqdm

from shot_detect.detect import detect, split_video
from prompts import GLOBAL_CAPTION, GATHER_CAPTION

import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

class Config:
    video_root_path = "/home/work/pengyimin/datasets/最强大学生"


def encode_image(video_path):
    with open(video_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def group_shots(shots, group_size=30):
    groups = []

    for group_idx, i in enumerate(range(0, len(shots), group_size)):
        groups.append({
            "group_id": group_idx,
            "shots": shots[i:i + group_size]
        })

    return groups

def materialize_shot_base64(shot):
    """
    从原视频中裁剪一个 shot，临时生成视频并转 base64
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", str(shot["start_time"]),
            "-i", shot["video_path"],
            "-t", str(shot["duration"]),
            "-c", "copy",
            tmp.name
        ]
        subprocess.run(cmd, check=True)
        return encode_video(tmp.name)

def call_llm_for_group(group):
    """
    group: {group_id, shots}
    """
    content = [
        {
            "type": "text",
            "text": GATHER_CAPTION
        }
    ]

    for shot in group["shots"]:
        base64_video = materialize_shot_base64(shot)
        content.append({
            "type": "video_url",
            "video_url": {
                "url": f"data:video/quicktime;base64,{base64_video}"
            }
        })

    response = client.chat.completions.create(
        model="/root/paddlejob/workspace/env_run/multi-shot-master/models/Qwen3-VL-32B-Instruct",
        messages=[{"role": "user", "content": content}],
    )

    return response.choices[0].message.content

def process_single_video(config, video_path):
    print(f"getting video: {video_path}")

    # 1. TransNet v2 - Shot Detect
    output = detect(video_path)
    shots = split_video(video_path, output)
    print(f"[INFO] Detected {len(shots)} shots")

    # 2. 分组采样
    # 暂时设定策略为均分
    groups = group_shots(shots)
    print(f"[INFO] Grouped into {len(groups)} groups")

    # 3. 调用qwen进行caption
    for group in groups:
        print(f"[INFO] Calling LLM for group {group['group_id']}")
        result = call_llm_for_group(group)
        print(result)




def get_all_mp4_files(root_path):
    root_path = Path(root_path)

    def extract_number(path: Path):
        """
        从文件名中提取数字，用于排序
        e.g. '10.mp4' -> 10
        """
        match = re.search(r'\d+', path.stem)
        return int(match.group()) if match else float('inf')

    return sorted(
        root_path.rglob("*.mp4"),
        key=extract_number
    )

def main():
    config = Config()

    video_files = get_all_mp4_files(config.video_root_path)
    for video_path in tqdm(video_files, desc="Processing videos"):
        tqdm.write(f"Processing video: {video_path}")
        process_single_video(config, video_path)
        exit()


if __name__ == '__main__':
    main()
