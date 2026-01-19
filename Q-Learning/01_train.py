# -*- coding: utf-8 -*-
"""
GRF Tabular Q-learning (minimal state) + Evaluation + Save/Load + CSV logging

- Environment: Google Research Football (gfootball)
- Representation: simple115v2
- Reward: scoring,checkpoints
- Tabular Q-learning using discretized ball (x,y) only
"""

import argparse
import csv
import os
import pickle
import random
from collections import defaultdict

import gfootball.env as football_env

# ----------------------------
# simple115v2 indices (common convention)
# ball (x,y,z) at 88,89,90
# ----------------------------
BALL_X, BALL_Y = 88, 89


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def to_bin(x, n=9, lo=-1.0, hi=1.0):
    """Map continuous x in [lo,hi] to integer bin in {0..n-1}."""
    x = clamp(x, lo, hi)
    t = (x - lo) / (hi - lo + 1e-12)
    b = int(t * n)
    return max(0, min(n - 1, b))


def make_state(obs, bins=9):
    """Very small discrete state: discretized ball (x,y)."""
    return (to_bin(obs[BALL_X], bins), to_bin(obs[BALL_Y], bins))


def eps_greedy(Q, s, nA, eps):
    """Epsilon-greedy action selection from Q-table."""
    if random.random() < eps:
        return random.randrange(nA)
    best_a, best_q = 0, Q[(s, 0)]
    for a in range(1, nA):
        q = Q[(s, a)]
        if q > best_q:
            best_q, best_a = q, a
    return best_a


def greedy(Q, s, nA):
    """Greedy action (eps=0)."""
    best_a, best_q = 0, Q[(s, 0)]
    for a in range(1, nA):
        q = Q[(s, a)]
        if q > best_q:
            best_q, best_a = q, a
    return best_a


def qlearning_update(Q, s, a, r, s2, done, nA, alpha, gamma):
    """Tabular Q-learning update."""
    # max_a' Q(s',a')
    max_next = Q[(s2, 0)]
    for a2 in range(1, nA):
        v = Q[(s2, a2)]
        if v > max_next:
            max_next = v

    target = r + (0.0 if done else gamma * max_next)
    Q[(s, a)] += alpha * (target - Q[(s, a)])


def evaluate_policy(env_name, Q, bins, n_eval=30, max_steps=1500, render=False, seed=None):
    """Evaluate greedy policy (eps=0)."""
    env = football_env.create_environment(
        env_name=env_name,
        representation="simple115v2",
        rewards="scoring,checkpoints",
        render=render,
    )
    if seed is not None:
        random.seed(seed)

    nA = env.action_space.n
    returns = []

    for _ in range(n_eval):
        obs = env.reset()
        s = make_state(obs, bins=bins)
        total = 0.0
        done = False
        steps = 0

        while (not done) and (steps < max_steps):
            steps += 1
            a = greedy(Q, s, nA)
            obs, r, done, info = env.step(int(a))
            total += float(r)
            s = make_state(obs, bins=bins)

        returns.append(total)

    env.close()

    avg = sum(returns) / len(returns)
    mn = min(returns) if returns else 0.0
    mx = max(returns) if returns else 0.0
    return avg, mn, mx, returns


def save_qtable(Q, path):
    with open(path, "wb") as f:
        # defaultdict는 그대로 pickle 가능하지만, 안전하게 dict로 저장
        pickle.dump(dict(Q), f)


def load_qtable(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    Q = defaultdict(float)
    Q.update(d)
    return Q


def write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="academy_empty_goal_close")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_min", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=0.995)

    parser.add_argument("--bins", type=int, default=9)

    parser.add_argument("--eval_n", type=int, default=30)
    parser.add_argument("--eval_render", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_q", type=str, default="qtable.pkl")
    parser.add_argument("--train_csv", type=str, default="train_log.csv")
    parser.add_argument("--eval_csv", type=str, default="eval_returns.csv")

    parser.add_argument("--load_only", action="store_true",
                        help="Skip training and only load Q-table then evaluate.")
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- Load-only mode ----
    if args.load_only:
        Q = load_qtable(args.save_q)
        avg, mn, mx, eval_returns = evaluate_policy(
            env_name=args.env,
            Q=Q,
            bins=args.bins,
            n_eval=args.eval_n,
            max_steps=args.max_steps,
            render=args.eval_render,
            seed=args.seed,
        )
        print(f"[EVAL greedy] n={args.eval_n} avg_return={avg:.3f} min={mn:.3f} max={mx:.3f}")
        write_csv(args.eval_csv, [[i + 1, r] for i, r in enumerate(eval_returns)], ["episode", "return"])
        print(f"Saved eval returns to: {args.eval_csv}")
        return

    # ---- Training env ----
    env = football_env.create_environment(
        env_name=args.env,
        representation="simple115v2",
        rewards="scoring,checkpoints",
        render=False,
    )

    Q = defaultdict(float)
    nA = env.action_space.n
    eps = args.eps_start

    train_rows = []

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        s = make_state(obs, bins=args.bins)
        total = 0.0
        done = False

        steps = 0
        while (not done) and (steps < args.max_steps):
            steps += 1

            a = eps_greedy(Q, s, nA, eps)
            obs2, r, done, info = env.step(int(a))
            s2 = make_state(obs2, bins=args.bins)

            qlearning_update(Q, s, a, r, s2, done, nA, args.alpha, args.gamma)

            total += float(r)
            s = s2

        eps = max(args.eps_min, eps * args.eps_decay)
        qsize = len(Q)

        # 로그 저장
        train_rows.append([ep, total, steps, eps, qsize])

        if ep % 10 == 0:
            print(f"ep={ep:4d} return={total:8.3f} eps={eps:.3f} Qsize={qsize}")

    env.close()

    # ---- Save Q-table + training CSV ----
    save_qtable(Q, args.save_q)
    print(f"\nSaved Q-table to: {args.save_q} (entries={len(Q)})")

    write_csv(args.train_csv, train_rows, ["episode", "return", "steps", "eps", "Qsize"])
    print(f"Saved training log to: {args.train_csv}")

    # ---- Evaluation (greedy) ----
    print("\n=== Evaluation (greedy, eps=0) ===")
    avg, mn, mx, eval_returns = evaluate_policy(
        env_name=args.env,
        Q=Q,
        bins=args.bins,
        n_eval=args.eval_n,
        max_steps=args.max_steps,
        render=args.eval_render,
        seed=args.seed,
    )
    print(f"[EVAL greedy] n={args.eval_n} avg_return={avg:.3f} min={mn:.3f} max={mx:.3f}")

    write_csv(args.eval_csv, [[i + 1, r] for i, r in enumerate(eval_returns)], ["episode", "return"])
    print(f"Saved eval returns to: {args.eval_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()