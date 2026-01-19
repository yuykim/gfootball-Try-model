```bash
conda create -n grf_dqn python=3.10 -y
conda activate grf_dqn
pip install -U pip
pip install "stable-baselines3[extra]" numpy
```

```bash
python 01_train.py
```

```bash
# 영상 저장 
python record_mp4_ffmpeg.py \
  —env academy_run_to_score_with_keeper \
  —model ./dqn_keeper_final.zip \
  —outdir ./videos_mp4 \
  —episodes 3 \
  —fps 30 \
  —max_steps 2500
```