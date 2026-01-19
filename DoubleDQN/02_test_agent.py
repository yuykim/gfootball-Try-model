import gfootball.env as football_env
from double_dqn import DoubleDQN
import utils
import os


def main():

    env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        render=True,
        write_video=False,
        representation="simple115v2",
        rewards="scoring, checkpoint",
    )

    obs = env.reset()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "doubleDQN_runToScore_500000_steps.zip")

    model = DoubleDQN.load(model_path, env=env, device="cpu")

    total_episodes = 100 # 테스트할 에피소드 횟수
    global_t = 0 # 전체 프레임 인덱스

    for ep in range(total_episodes):
        obs = env.reset()
        done = False
        max_steps = 500
        ep_reward = 0
        t = 0

        print(f"--- Episode {ep + 1} Start ---")

        while (not done) and (t < max_steps):
            # 모델 예측
            action, _ = model.predict(obs, deterministic=True) # 테스트시는 deterministic=True
            obs, reward, done, info = env.step(action)

            # 프레임 저장
            #frame = env.render(mode="rgb_array")
            #if frame is not None:
            #    utils.save_frame(frame, global_t)

            if reward != 0:
                print(f"Ep {ep+1} | Step: {t} | Reward: {reward}")
            
            t += 1
            global_t += 1
            ep_reward += reward

        print(f"Episode {ep + 1} Done | Reward: {ep_reward}")

    env.close()
    print("Testing finished. Creating video...")

if __name__ == "__main__":
    # 기존 프레임 삭제
    #utils.cleanup()

    main()

    # 영상 제작
    #utils.make_video()