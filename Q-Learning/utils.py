import os
import random
import numpy as np
import torch
import shutil
import subprocess
from PIL import Image
from stable_baselines3.common.utils import set_random_seed

OUT_DIR = "frames"


def save_frame(frame_bgr, step):
    if frame_bgr is not None:
        # BGR to RGB
        frame_rgb = frame_bgr[..., ::-1]
        Image.fromarray(frame_rgb).save(f"{OUT_DIR}/{step:05d}.png")


def cleanup():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)


def make_video_with_suffix(suffix):
    if suffix != "":
        suffix += "-"
    output_video = f"replay-{suffix}001.mp4"
    counter = 0
    while os.path.exists(output_video):
        counter += 1
        output_video = f"replay-{suffix}{counter:03d}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-r",
        "24",
        "-i",
        f"{OUT_DIR}/%05d.png",
        "-pix_fmt",
        "yuv420p",
        output_video,
    ]

    subprocess.run(cmd, check=True)

    print("Video saved to", output_video)


def make_video():
    make_video_with_suffix(suffix="")


def seed_everything(seed: int, deterministic_torch: bool = True):
    # Python / NumPy
    random.seed(seed)
    np.random.seed(seed)

    # SB3 helper (also seeds numpy + python random again; harmless)
    set_random_seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # More reproducible, can be slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Optional (newer torch): may raise on some ops
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # Sometimes helps with hash-based nondeterminism
    os.environ["PYTHONHASHSEED"] = str(seed)
