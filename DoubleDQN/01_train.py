import gfootball.env as football_env
from double_dqn import DoubleDQN
from custom_reward import CustomReward 
from stable_baselines3.common.callbacks import CheckpointCallback
import os

def make_env():
    """구글 풋볼 환경 생성 및 기본 설정"""
    return football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        render=False,              # 학습 가속화를 위해 렌더링은 비활성화
        write_video=False,
        representation="simple115v2",
        rewards="scoring,checkpoint", # 득점 및 전진 보상 활성화
    )

def main():
    # 1. 환경 생성 및 커스텀 리워드 래퍼 적용
    base_env = make_env()
    env = CustomReward(base_env) # 정체 방지 및 패스 유도 로직 포함

    # 2. 체크포인트 콜백 설정 (50만 스텝마다 저장)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path='./models/',
        name_prefix='doubleDQN_runToScore'
    )

    # 3. Double DQN 모델 정의
    model = DoubleDQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,           # 단기 학습을 위해 조금 더 높은 학습률
        buffer_size=100_000,          # 최신 성공 경험 위주로 학습하기 위해 버퍼 최적화
        learning_starts=5_000,        # 아주 빠르게 학습 시작
        batch_size=256,               # 배치 사이즈를 키워 학습 안정성 확보
        gamma=0.98,                   # 단기 보상에 더 집중 (멀리 있는 골보다 당장의 전진)
        exploration_fraction=0.1,     # 전체의 10%(5만 스텝) 지점에서 탐색 종료
        exploration_final_eps=0.02,   # 80% 성공률을 위해 탐색 확률을 아주 낮게 고정(2%)
        policy_kwargs={'net_arch': [256, 256]}, 
        verbose=1
    )

    # 4. 학습 시작
    print("Starting Double DQN training for 5,000,000 steps...")
    print("Checkpoints will be saved every 500,000 steps.")
    
    try:
        model.learn(
            total_timesteps=500_000,
            callback=checkpoint_callback
        )
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving the current model...")

    # 5. 최종 모델 저장 및 종료
    final_model_path = "double_dqn_run_to_score_5M_final"
    model.save(final_model_path)
    print(f"Final model saved as: {final_model_path}")
    
    env.close()

if __name__ == "__main__":
    # 필요한 폴더 자동 생성
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./models/checkpoints/", exist_ok=True)
    
    main()