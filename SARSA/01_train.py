# SARSA Codes

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gfootball.env as football_env


# 간단한 MLP Q-network
class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def epsilon_greedy(q_values, eps: float):
    if random.random() < eps:
        return random.randrange(q_values.shape[-1])
    return int(torch.argmax(q_values).item())


def main():
    # 1) 환경 생성
    env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        render=False,
        # representation="simple115v2",  # 필요하면 명시 (과제/레포 설정에 따라)
    )

    obs = env.reset()

    # 2) obs/action 크기 파악
    # obs가 numpy array라고 가정
    obs_dim = int(np.prod(obs.shape))
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = QNet(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(q.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    # SARSA 하이퍼파라미터
    gamma = 0.99
    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.995

    num_episodes = 50  # timesteps 대신 에피소드로 관리하는 게 보통 편함
    max_steps = 500

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        # 현재 정책(ε-greedy)으로 A 선택
        obs_t = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            a = epsilon_greedy(q(obs_t), eps)

        ep_return = 0.0

        for t in range(max_steps):
            next_obs, r, done, info = env.step(a)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            ep_return += float(r)

            # 다음 상태에서 같은 정책으로 A' 선택 (on-policy 핵심)
            next_obs_t = torch.from_numpy(next_obs).to(device)
            with torch.no_grad():
                next_a = epsilon_greedy(q(next_obs_t), eps)

            # SARSA 타깃: r + gamma * Q(s', a')
            obs_t = torch.from_numpy(obs).to(device)
            q_sa = q(obs_t)[a]

            with torch.no_grad():
                q_s_next_a_next = q(next_obs_t)[next_a]
                target = torch.tensor(r, dtype=torch.float32, device=device) + (0.0 if done else gamma * q_s_next_a_next)

            loss = loss_fn(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs
            a = next_a

            if done:
                break

        eps = max(eps_min, eps * eps_decay)
        print(f"[EP {ep:03d}] return={ep_return:.2f}, eps={eps:.3f}")

    # 저장
    torch.save(q.state_dict(), "model-sarsa.pt")
    env.close()


if __name__ == "__main__":
    main()