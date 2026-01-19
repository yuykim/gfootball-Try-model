import os
import shutil
import subprocess
from PIL import Image

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


def make_video():
    output_video = f"replay-001.mp4"
    counter = 0
    while os.path.exists(output_video):
        counter += 1
        output_video = f"replay-{counter:03d}.mp4"

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