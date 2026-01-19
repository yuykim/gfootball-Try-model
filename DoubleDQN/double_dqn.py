import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.dqn.dqn import DQN


class DoubleDQN(DQN):
    """
    Minimal Double DQN patch for older SB3.
    Fix: ReplayBufferSamples has no `discounts`, so we use self.gamma.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # ✅ Double DQN target:
                # a* = argmax_a Q_online(s', a)
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)

                # Q_target(s', a*)
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_target.gather(1, next_actions)

                # 1-step target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # current Q(s, a)
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))
