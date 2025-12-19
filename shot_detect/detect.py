"""TransNetV2 shot detection.

Requires `ffmpeg` installed on host.
"""

from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
import os
import subprocess
model_path = "shot_detect/transnetv2-pytorch-weights.pth"

from . import utils
from .transnet import TransNetV2


@lru_cache
def load_model() -> Callable[[torch.Tensor], torch.Tensor]:
    """Singleton of model."""

    model = TransNetV2()
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().cuda()
    return model


def detect(video_path: str, threshold: float = 0.4) -> list[tuple[int, int]]:
    """Shot detection with TransNetV2."""

    model = load_model()
    frames = utils.get_frames(video_path, height=27, width=48)
    print(f"{len(frames)} frames extracted.")
    with torch.no_grad():
        predictions = utils.get_predictions(model, frames, threshold)
    scenes = utils.get_scenes(predictions)
    return scenes.tolist()

def get_fps(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path      
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            numerator, denominator = map(int, fps_str.split('/'))
            return numerator / denominator
        else:
            return float(fps_str)
    except Exception as e:
        print(f"获取 FPS 失败: {e}")
        return None


# def split_video(video_path, frame_seqs):
#     fps = get_fps(video_path)
#     output_path = video_path.split('.')[0]
#     if os.path.exists(output_path) is False:
#         os.mkdir(output_path)
#     pathes = []
#     for idx,frames in enumerate(frame_seqs):
#         start_frame = frames[0]
#         end_frame = frames[1]
#         start_time = start_frame / fps

#         duration = (end_frame - start_frame) / fps
#         cmd = [
#             'ffmpeg',
#             "-hide_banner",
#             "-loglevel", "warning", # 只显示警告和错误
#             '-ss', str(start_time),
#             '-i', video_path,
#             '-t', str(duration),
#             '-c', 'copy',
#             f'{output_path}/{idx}.mp4'
#         ]
#         try:
#             subprocess.run(cmd, check=True)
#             pathes.append(f'{output_path}/{idx}.mp4')
#             print(f'Extracted scene at time {start_time:.2f} seconds')  
#         except Exception as e:
#             print('Error: ',e)
#     return pathes

def split_video(video_path, frame_seqs):
    fps = get_fps(video_path)

    shots = []

    for idx, (start_frame, end_frame) in enumerate(frame_seqs):
        start_time = start_frame / fps
        duration = (end_frame - start_frame) / fps

        shots.append({
            "shot_id": idx,
            "video_path": video_path,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "duration": duration
        })

    return shots

if __name__ == "__main__":
    output=detect("/home/work/pengyimin/datasets/MV合集/8090后怀旧MV（174G）/2-90后怀旧/1-《北半球的孤单》-林依晨 爱情合约 电视原声带-1080P 高清-AVC.mp4")
    split_video("/home/work/pengyimin/datasets/MV合集/8090后怀旧MV（174G）/2-90后怀旧/1-《北半球的孤单》-林依晨 爱情合约 电视原声带-1080P 高清-AVC.mp4",output)