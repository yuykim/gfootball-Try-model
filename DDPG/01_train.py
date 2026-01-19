import gym
import numpy as np
import gfootball.env as football_env
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise


class ContinuousActionWrapper(gym.Wrapper):
    """
    Map continuous actions to discrete by argmax.
    This lets DDPG output a continuous vector while the env expects discrete.
    """

    def __init__(self, env):
        super().__init__(env)
        self.n_actions = self.env.action_space.n
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actions,),
            dtype=np.float32,
        )

    def step(self, action):
        discrete_action = int(np.argmax(action))
        return self.env.step(discrete_action)


def main():
    env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        representation="simple115v2",
        render=False,
        write_video=False,
    )

    env = ContinuousActionWrapper(env)

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    model = DDPG(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        action_noise=action_noise,
    )

    model.learn(total_timesteps=500_000)
    model.save("model-run-to-score-ddpg.zip")
    env.close()
    print("Training completed! Model saved as model-run-to-score-ddpg.zip")


if __name__ == "__main__":
    main()
