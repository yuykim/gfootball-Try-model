import numpy as np
import torch
import torch.nn as nn
import gfootball.env as football_env
import utils


# 그대로 모델 불러오
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


def select_action_greedy(qnet, obs, device):
    """테스트는 greedy(ε=0)로 행동 선택"""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    obs_t = torch.from_numpy(obs).to(device)

    with torch.no_grad():
        q_values = qnet(obs_t)  # [n_actions]
        action = int(torch.argmax(q_values).item())

    return action


def main():
    env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        render=False
    )

    obs = env.reset()

    obs_dim = int(np.prod(np.asarray(obs).shape))
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SARSA 모델 로드
    qnet = QNet(obs_dim, n_actions).to(device)
    qnet.load_state_dict(torch.load("model-sarsa.pt", map_location=device))
    qnet.eval()

    done = False
    max_steps = 500
    total_reward = 0.0

    t = 0
    while (not done) and (t < max_steps):
        action = select_action_greedy(qnet, obs, device)
        obs, reward, done, info = env.step(action)

        frame = env.render(mode="rgb_array")
        if frame is not None:
            utils.save_frame(frame, t)

        if t % 20 == 0:
            print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)

        total_reward += float(reward)
        t += 1

    print(f"Step: {t}/{max_steps}", "Done:", done)
    print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":
    utils.cleanup()
    main()
    utils.make_video()