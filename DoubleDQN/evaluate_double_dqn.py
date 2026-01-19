import os
import numpy as np
import gfootball.env as football_env
from double_dqn import DoubleDQN


# -----------------------------
# 1) env (scoring only)
# -----------------------------
def make_env(render=False):
    return football_env.create_environment(
        env_name="5_vs_5",
        render=render,
        write_video=False,
        representation="simple115v2",
        rewards="scoring",   # +1 득점, -1 실점, 0 없음
    )


# -----------------------------
# 2) evaluate: global timeline event steps
# -----------------------------
def evaluate_scoring(model, env, episodes=20, max_steps=3000, deterministic=True):
    total_home = 0
    total_away = 0
    home_list = []
    away_list = []

    score_steps_global = []    # reward +1 발생 global step
    concede_steps_global = []  # reward -1 발생 global step

    total_timeline_steps = episodes * max_steps

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        t = 0

        home = 0
        away = 0

        while (not done) and (t < max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)

            g = (ep - 1) * max_steps + t  # global step

            # float 안전 처리
            if reward > 0.5:
                home += 1
                score_steps_global.append(g)
            elif reward < -0.5:
                away += 1
                concede_steps_global.append(g)

            t += 1

        total_home += home
        total_away += away
        home_list.append(home)
        away_list.append(away)

        print(f"[{ep}/{episodes}] home={home} away={away}", end="\r")

    print()

    return {
        "episodes": episodes,
        "max_steps": max_steps,
        "total_timeline_steps": total_timeline_steps,
        "total_home_score": total_home,
        "total_away_score": total_away,
        "home_score_list": np.array(home_list, dtype=int),
        "away_score_list": np.array(away_list, dtype=int),
        "score_steps_global": np.array(score_steps_global, dtype=int),
        "concede_steps_global": np.array(concede_steps_global, dtype=int),
    }


# -----------------------------
# 3) helpers: step stats
# -----------------------------
def step_stats(arr):
    arr = np.asarray(arr, dtype=int)
    if len(arr) == 0:
        return None, None, None
    return int(arr.min()), int(arr.max()), float(arr.mean())


# -----------------------------
# 4) across-games no-concede stats (global timeline)
# -----------------------------
def no_concede_stats(concede_steps_global, total_timeline_steps):
    """
    경기 넘어(전체 타임라인) 기준
    - no-concede span = 실점 이벤트 사이 gap (start~first, between, last~end)
    반환:
      longest_span, avg_span, (best_start, best_end)
    """
    c = np.asarray(concede_steps_global, dtype=int)

    if len(c) == 0:
        # 실점이 없으면 전체가 무실점
        return int(total_timeline_steps), float(total_timeline_steps), (0, int(total_timeline_steps))

    c = np.sort(c)

    spans = []
    starts = []
    ends = []

    # 0 ~ first concede
    spans.append(int(c[0]))
    starts.append(0)
    ends.append(int(c[0]))

    # between concede events
    for i in range(1, len(c)):
        prev_c = int(c[i - 1])
        cur_c = int(c[i])
        spans.append(cur_c - prev_c)
        starts.append(prev_c)
        ends.append(cur_c)

    # last concede ~ end
    last_c = int(c[-1])
    spans.append(int(total_timeline_steps) - last_c)
    starts.append(last_c)
    ends.append(int(total_timeline_steps))

    spans = np.array(spans, dtype=int)
    longest = int(spans.max())
    avg = float(spans.mean())

    idx = int(spans.argmax())
    best_start = int(starts[idx])
    best_end = int(ends[idx])
    return longest, avg, (best_start, best_end)


def global_to_ep_step(global_step, max_steps):
    ep = global_step // max_steps + 1
    step_in = global_step % max_steps
    return int(ep), int(step_in)


# -----------------------------
# 5) print summary
# -----------------------------
def print_summary(stats):
    episodes = stats["episodes"]
    max_steps = stats["max_steps"]
    total_timeline_steps = stats["total_timeline_steps"]

    total_home = stats["total_home_score"]
    total_away = stats["total_away_score"]
    home_list = stats["home_score_list"]
    away_list = stats["away_score_list"]

    score_steps = stats["score_steps_global"]
    concede_steps = stats["concede_steps_global"]

    avg_home = float(home_list.mean()) if episodes else 0.0
    avg_away = float(away_list.mean()) if episodes else 0.0

    # step stats
    h_fast, h_late, h_avg = step_stats(score_steps)
    c_fast, c_late, c_avg = step_stats(concede_steps)

    # no-concede stats (across games)
    longest_nc, avg_nc, (best_s, best_e) = no_concede_stats(concede_steps, total_timeline_steps)

    print("\n==============================")
    print("Evaluation Summary (scoring)")
    print("==============================")
    print(f"Episodes           : {episodes}")
    print(f"Max steps / game   : {max_steps}")
    print(f"Total Home Score   : {total_home}")
    print(f"Total Away Score   : {total_away}")
    print(f"Avg score / game   : {avg_home:.3f} - {avg_away:.3f}")

    # requested lines
    if h_fast is None:
        print("Fastest home goal step : N/A")
        print("Latest home goal step  : N/A")
        print("Avg home goal step     : N/A")
    else:
        print(f"Fastest home goal step : {h_fast}")
        print(f"Latest home goal step  : {h_late}")
        print(f"Avg home goal step     : {h_avg:.1f}")

    if c_fast is None:
        print("Fastest concede step   : N/A")
        print("Latest concede step    : N/A")
        print("Avg concede step       : N/A")
    else:
        print(f"Fastest concede step   : {c_fast}")
        print(f"Latest concede step    : {c_late}")
        print(f"Avg concede step       : {c_avg:.1f}")

    # no-concede spans
    print(f"Longest no-concede span: {longest_nc}")
    print(f"Avg no-concede span    : {avg_nc:.1f}")

    # best span location (ep/step 표시)
    s_ep, s_st = global_to_ep_step(best_s, max_steps)
    e_ep, e_st = global_to_ep_step(best_e, max_steps)
    print(f"Best no-concede range  : global[{best_s} -> {best_e}]  (~Ep{s_ep} step{s_st} -> Ep{e_ep} step{e_st})")

    print("==============================\n")


# -----------------------------
# 6) main
# -----------------------------
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "DoubleDQN_model", "double_dqn_5v5_500k.zip")

    EPISODES = 20
    MAX_STEPS = 3000
    DEVICE = "cpu"
    DETERMINISTIC = True
    RENDER = False

    if not os.path.exists(MODEL_PATH):
        print("model not found:", MODEL_PATH)
        return

    env = make_env(render=RENDER)
    model = DoubleDQN.load(MODEL_PATH, env=env, device=DEVICE)

    stats = evaluate_scoring(
        model,
        env,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        deterministic=DETERMINISTIC,
    )
    env.close()

    print("Model:", MODEL_PATH)
    print_summary(stats)


if __name__ == "__main__":
    main()
