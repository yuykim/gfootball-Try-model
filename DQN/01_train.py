import inspect
import gfootball.env as football_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env(env_name):
    def _init():
        base_kwargs = dict(
            env_name=env_name,
            representation="simple115v2",
            rewards="scoring,checkpoints",
            stacked=False,
            render=False,
        )
        sig = inspect.signature(football_env.create_environment)
        kwargs = {k:v for k,v in base_kwargs.items() if k in sig.parameters}
        env = football_env.create_environment(**kwargs)
        return Monitor(env)
    return _init

env_name = "academy_run_to_score_with_keeper"
env = DummyVecEnv([make_env(env_name)])

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=200_000,
    learning_starts=20_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=2000,
    exploration_fraction=0.40,
    exploration_final_eps=0.05,
    verbose=1
)

model.learn(total_timesteps=500_000)
model.save("dqn_model.zip")