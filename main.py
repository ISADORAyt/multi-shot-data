import os
import re
import json
import time
import traceback
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

def materialize_shot_base64(shot, *, default_video_path=None):
    """
    将不同形态的 shot 统一转成 base64 视频
    支持：
      - str: 已切好的视频路径
      - dict: 包含裁剪信息的 shot
    """
    # 情况 1：shot 是路径
    if isinstance(shot, str):
        if not os.path.exists(shot):
            raise FileNotFoundError(f"Shot path not found: {shot}")
        return encode_video(shot)

    # 情况 2：shot 是 dict
    if isinstance(shot, dict):
        video_path = shot.get("video_path") or default_video_path
        if video_path is None:
            raise ValueError(f"shot missing video_path: {shot}")

        start = shot.get("start_time")
        if start is None:
            start = shot.get("start")

        if start is None:
            raise ValueError(f"shot missing start time: {shot}")

        # duration / end 处理
        duration = shot.get("duration")
        if duration is None:
            end = shot.get("end_time")
            if end is None:
                end = shot.get("end")
            if end is not None:
                duration = float(end) - float(start)

        if duration is None:
            raise ValueError(f"shot missing duration: {shot}")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-ss", str(start),
                "-i", str(video_path),
                "-t", str(duration),
                "-c", "copy",
                tmp.name
            ]
            subprocess.run(cmd, check=True)
            return encode_video(tmp.name)

    raise TypeError(f"Unsupported shot type: {type(shot)}")

def compute_group_time(group, shots):
    """
    根据 group 的 shot_ids，从 shot 级时间推导 group 时间

    Args:
        group: {"group_id": int, "shot_ids": [...]}
        shots: List[shot dict]

    Returns:
        dict with start_sec / end_sec / duration_sec
    """
    first_shot = shots[group["shot_ids"][0]]
    last_shot = shots[group["shot_ids"][-1]]

    start_sec = first_shot["start_sec"]
    end_sec = last_shot["end_sec"]

    return {
        "group_start_sec": start_sec,
        "group_end_sec": end_sec,
        "group_duration_sec": end_sec - start_sec
    }



def call_llm(
    *,
    config,
    prompt_text: str,
    shots: list,
    model_path: str = "/root/paddlejob/workspace/env_run/multi-shot-master/models/Qwen3-VL-32B-Instruct",
):
    """
    通用 LLM 调用函数

    Args:
        prompt_text: 发送给 LLM 的文本 prompt
        shots: shot dict 的列表（支持 1 个或多个）
        model_path: 模型路径

    Returns:
        raw LLM output (str)
    """
    content = [
        {
            "type": "text",
            "text": prompt_text
        }
    ]

    for shot in shots:
        base64_video = materialize_shot_base64(shot)
        content.append({
            "type": "video_url",
            "video_url": {
                "url": f"data:video/quicktime;base64,{base64_video}"
            }
        })

    response = client.chat.completions.create(
        model=model_path,
        messages=[{"role": "user", "content": content}],
    )

    raw_text = response.choices[0].message.content

    return parse_llm_description(config, raw_text)

def safe_call_llm(*, config, prompt_text, shots, role, meta=None):
    t0 = time.time()
    try:
        result = call_llm(
            config=config,
            prompt_text=prompt_text,
            shots=shots
        )
        return {
            "ok": True,
            "result": result,
            "time_sec": time.time() - t0
        }
    except Exception as e:
        return {
            "ok": False,
            "result": None,
            "error": f"{type(e).__name__}: {e}",
            "time_sec": time.time() - t0
        }


# def process_single_video(config, video_path):
#     print(f"getting video: {video_path}")

#     # 1. TransNet v2 - Shot Detect
#     output = detect(video_path)
#     shots = split_video(video_path, output)
#     print(f"[INFO] Detected {len(shots)} shots")

#     # 2. 分组采样
#     # 暂时设定策略为均分
#     groups = group_shots(shots)
#     print(f"[INFO] Grouped into {len(groups)} groups")

#     # 3. 调用qwen进行caption
#     for group in groups:
#         print(f"[INFO] Calling LLM for group {group['group_id']}")
#         # 3.1 进行全局描述生成
#         global_caption = call_llm(
#             config=config,
#             prompt_text=GLOBAL_CAPTION, 
#             shots=group["shots"]
#         )
#         print("全局描述为：", global_caption)

#         # 3.2 进行单独镜头描述
#         for idx, shot in enumerate(group["shots"]):
#             shot_caption = call_llm(
#                 config=config,
#                 prompt_text=SHOT_CAPTION.format(global_caption=global_caption),
#                 shots=[shot]
#             )
#             print(f"第{idx}个镜头的描述为：{shot_caption}")
#         exit()

def sec_to_timecode(sec: float) -> str:
    """
    将秒数转换为 HH:MM:SS.mmm 形式的 timecode
    """
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# def process_single_video(config, video_path, video_id=0):
#     print(f"getting video: {video_path}")

#     # 1. Shot Detect
#     output = detect(video_path)
#     raw_shots = split_video(video_path, output)
#     print(f"[INFO] Detected {len(raw_shots)} shots")

#     # 2. 结构化 shot-level 数据
#     shots = []
#     for s in raw_shots:
#         start_sec = float(s["start_time"])
#         duration = float(s["duration"])
#         end_sec = start_sec + duration

#         shots.append({
#             "shot_id": s["shot_id"],

#             "start_sec": start_sec,
#             "end_sec": end_sec,
#             "duration_sec": duration,

#             "start_timecode": sec_to_timecode(start_sec),
#             "end_timecode": sec_to_timecode(end_sec),

#             "start_frame": s.get("start_frame"),
#             "end_frame": s.get("end_frame"),
#             "frames_cnt": s.get("end_frame") - s.get("start_frame")
#                           if s.get("start_frame") is not None else None,

#             "shot_caption": None  # 后面填
#         })

#     # 3. 分组
#     groups_raw = group_shots(shots, group_size=3)

#     groups = []

#     for group in groups_raw:
#         print(f"[INFO] Calling LLM for group {group['group_id']}")

#         # 3.1 group caption
#         group_caption = call_llm(
#             config=config,
#             prompt_text=GLOBAL_CAPTION,
#             shots=[raw_shots[i] for i in range(
#                 group["shots"][0]["shot_id"],
#                 group["shots"][-1]["shot_id"] + 1
#             )]
#         )

#         group_entry = {
#             "group_id": group["group_id"],
#             "shot_ids": [shot["shot_id"] for shot in group["shots"]],
#             "group_caption": group_caption
#         }

#         # 3.2 shot caption
#         for shot in group["shots"]:
#             caption = call_llm(
#                 config=config,
#                 prompt_text=SHOT_CAPTION.format(global_caption=group_caption),
#                 shots=[raw_shots[shot["shot_id"]]]
#             )
#             shots[shot["shot_id"]]["shot_caption"] = caption

#         groups.append(group_entry)

#     # 4. video-level 汇总
#     video_annotation = {
#         "video_id": video_id,
#         "video_path": str(video_path),

#         "video_duration_sec": shots[-1]["end_sec"],
#         "fps": None,
#         "total_frames": shots[-1]["end_frame"],

#         "shots_cnt": len(shots),
#         "groups_cnt": len(groups),
#         "group_size": 3,

#         "shots": shots,
#         "groups": groups
#     }

#     # 5. 写盘
#     out_path = Path("annotations") / f"{video_id:05d}.json"
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     # 验证shot级时间转到group级时间的准确性
#     gt = compute_group_time(video_annotation["groups"][0], video_annotation["shots"])
#     print(gt)

#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(video_annotation, f, ensure_ascii=False, indent=2)

#     print(f"[INFO] Saved annotation to {out_path}")

def process_single_video(config, video_path, video_id=0, group_size=3):
    video_t0 = time.time()
    print(f"[VIDEO START] {video_path}")

    # 1. Shot Detect
    output = detect(video_path)
    raw_shots = split_video(video_path, output)
    print(f"[INFO] Detected {len(raw_shots)} shots")

    # 显式映射（彻底解除 ID 连续假设）
    raw_shot_by_id = {s["shot_id"]: s for s in raw_shots}

    # 2. 结构化 shot-level annotation
    shots = []
    shot_ann_by_id = {}

    for s in raw_shots:
        start_sec = float(s["start_time"])
        duration = float(s["duration"])
        end_sec = start_sec + duration

        ann = {
            "shot_id": s["shot_id"],
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration,
            "start_timecode": sec_to_timecode(start_sec),
            "end_timecode": sec_to_timecode(end_sec),
            "start_frame": s.get("start_frame"),
            "end_frame": s.get("end_frame"),
            "frames_cnt": (
                s["end_frame"] - s["start_frame"]
                if s.get("start_frame") is not None else None
            ),
            "shot_caption": None,
            "shot_caption_ok": None
        }
        shots.append(ann)
        shot_ann_by_id[s["shot_id"]] = ann

    # 3. 分组（不假设连续）
    groups_raw = group_shots(shots, group_size=group_size)

    groups = []

    for group in groups_raw:
        group_id = group["group_id"]
        shot_ids = [s["shot_id"] for s in group["shots"]]
        print(f"[GROUP START] group={group_id} shot_ids={shot_ids}")

        group_t0 = time.time()

        # --- group caption ---
        shots_for_group = [raw_shot_by_id[sid] for sid in shot_ids]

        ret = safe_call_llm(
            config=config,
            prompt_text=GLOBAL_CAPTION,
            shots=shots_for_group,
            role="group",
            meta={"group_id": group_id}
        )

        group_entry = {
            "group_id": group_id,
            "shot_ids": shot_ids,
            "group_caption": ret["result"],
            "group_caption_ok": ret["ok"],
            "group_caption_time_sec": ret["time_sec"]
        }
        if not ret["ok"]:
            group_entry["group_caption_error"] = ret["error"]

        print(
            f"[LLM][GROUP] group={group_id} "
            f"ok={ret['ok']} time={ret['time_sec']:.2f}s"
        )

        # --- shot caption ---
        for sid in shot_ids:
            raw_shot = raw_shot_by_id[sid]
            ann = shot_ann_by_id[sid]

            ret = safe_call_llm(
                config=config,
                prompt_text=SHOT_CAPTION.format(
                    global_caption=group_entry["group_caption"]
                ),
                shots=[raw_shot],
                role="shot",
                meta={"shot_id": sid}
            )

            ann["shot_caption"] = ret["result"]
            ann["shot_caption_ok"] = ret["ok"]
            ann["shot_caption_time_sec"] = ret["time_sec"]

            if not ret["ok"]:
                ann["shot_caption_error"] = ret["error"]

            print(
                f"[LLM][SHOT] shot={sid} "
                f"ok={ret['ok']} time={ret['time_sec']:.2f}s"
            )

        group_cost = time.time() - group_t0
        print(f"[GROUP DONE] group={group_id} time={group_cost:.1f}s")

        groups.append(group_entry)

    # 4. Video-level annotation
    video_annotation = {
        "video_id": video_id,
        "video_path": str(video_path),
        "video_duration_sec": max(s["end_sec"] for s in shots),
        "fps": None,
        "total_frames": max(
            s["end_frame"] for s in shots if s["end_frame"] is not None
        ),
        "shots_cnt": len(shots),
        "groups_cnt": len(groups),
        "group_size": group_size,
        "shots": shots,
        "groups": groups
    }

    # 5. 写盘
    out_path = Path("annotations") / f"{video_id:05d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(video_annotation, f, ensure_ascii=False, indent=2)

    video_cost = time.time() - video_t0
    print(f"[VIDEO DONE] {video_path} | time={video_cost:.1f}s")


        
def main():
    config = Config()

    video_files = get_all_mp4_files(config.video_root_path)
    for video_path in tqdm(video_files, desc="Processing videos"):
        tqdm.write(f"Processing video: {video_path}")
        process_single_video(config, video_path)
        exit()


if __name__ == '__main__':
    main()
