import os
import argparse
import subprocess
import inspect

import numpy as np
import gfootball.env as football_env
from stable_baselines3 import DQN


def make_env(env_name):
    # 버전마다 create_environment 인자가 달라서 signature 기반으로 필터링
    base_kwargs = dict(
        env_name=env_name,
        representation="simple115v2",
        rewards="scoring,checkpoints",
        stacked=False,
        render=True,        # ✅ 렌더링 활성화
        write_video=False,  # ✅ 내부 avi 저장은 쓰지 않음(불안정해서)
    )
    sig = inspect.signature(football_env.create_environment)
    kwargs = {k: v for k, v in base_kwargs.items() if k in sig.parameters}
    return football_env.create_environment(**kwargs)


def get_frame(env):
    # 1) rgb_array로 프레임 얻기 시도
    try:
        frame = env.render(mode="rgb_array")
        if frame is not None:
            return frame
    except Exception:
        pass

    # 2) 그냥 render()가 ndarray를 주는 경우도 있음
    try:
        frame = env.render()
        if frame is not None and isinstance(frame, np.ndarray):
            return frame
    except Exception:
        pass

    raise RuntimeError("env.render()에서 RGB 프레임을 가져오지 못했습니다.")


def start_ffmpeg(out_path, width, height, fps):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="academy_run_to_score_with_keeper")
    p.add_argument("--model", required=True)
    p.add_argument("--outdir", default="./videos_mp4")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max_steps", type=int, default=2500)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    env = make_env(args.env)
    model = DQN.load(args.model)

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        # 첫 프레임으로 해상도 결정
        frame = get_frame(env)
        h, w = frame.shape[:2]

        out_path = os.path.join(args.outdir, f"{args.env}_ep{ep}.mp4")
        proc = start_ffmpeg(out_path, w, h, args.fps)

        # 첫 프레임 기록
        proc.stdin.write(frame.astype(np.uint8).tobytes())

        while not done and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            frame = get_frame(env)
            proc.stdin.write(frame.astype(np.uint8).tobytes())

        proc.stdin.close()
        proc.wait()

        print(f"[EP {ep}] reward={total_reward:.2f} saved={out_path}")

    env.close()


if __name__ == "__main__":
    main()