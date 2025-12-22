import os
import re
import json
import tempfile
import subprocess
from pathlib import Path
from tqdm import tqdm

from shot_detect.detect import detect, split_video
from prompts import GLOBAL_CAPTION, GATHER_CAPTION, SHOT_CAPTION

import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

class Config:
    video_root_path = "/root/paddlejob/workspace/env_run/multi-shot-master/datasets/最强大学生"
    description_keys = ["全局描述", "镜头描述"]

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


def encode_video(video_path):
    with open(video_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def parse_llm_description(config, text: str) -> str:
    """
    通用解析 LLM 输出中的描述字段
    支持 key: 全局描述 / 镜头描述

    返回:
        description (str)

    异常:
        ValueError
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM output")

    raw = text.strip()

    # 1. 去除 ```json ``` 包裹
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # 2. 尝试直接解析
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 3. 正则提取 JSON 再解析
    if data is None:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError(f"Cannot find JSON in LLM output:\n{text}")

        json_str = match.group(0)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON parsing failed.\nJSON:\n{json_str}\nError:{e}"
            )

    # 4. 提取描述字段
    for key in config.description_keys:
        if key in data and isinstance(data[key], str):
            return data[key].strip()

    # 5. 兜底：如果只有一个字符串字段
    str_fields = [v for v in data.values() if isinstance(v, str)]
    if len(str_fields) == 1:
        return str_fields[0].strip()

    raise ValueError(f"No valid description field found in JSON: {data}")

def group_shots(shots, group_size=3):
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
            "-y",  # ⭐ 关键：自动覆盖
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
            "text": GLOBAL_CAPTION
        }
    ]

    # print("接收到的输入为：", content)

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

def call_llm_for_shot(shot, global_caption):

    content = [
        {
            "type": "text",
            "text": SHOT_CAPTION.format(global_caption=global_caption)
        }
    ]

    # print("接收到的输入为：", content)

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
        # 3.1 进行全局描述生成
        global_caption = call_llm_for_group(group)
        global_caption = parse_llm_description(config, global_caption)
        print("全局描述为：", global_caption)

        # 3.2 进行单独镜头描述
        for idx, shot in enumerate(group["shots"]):
            shot_caption = call_llm_for_shot(shot, global_caption)
            shot_caption = parse_llm_description(config, shot_caption)
            print(f"第{idx}个镜头的描述为：{shot_caption}")
        
def main():
    config = Config()

    video_files = get_all_mp4_files(config.video_root_path)
    for video_path in tqdm(video_files, desc="Processing videos"):
        tqdm.write(f"Processing video: {video_path}")
        process_single_video(config, video_path)
        exit()


if __name__ == '__main__':
    main()
