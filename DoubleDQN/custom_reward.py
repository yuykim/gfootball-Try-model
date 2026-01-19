import gym
import numpy as np

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_x = None
        self.steps_from_kickoff = 0

    def reset(self, **kwargs):
        self.prev_ball_x = None
        self.steps_from_kickoff = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        d = self.obs_to_dict(obs)
        
        # 기본 데이터 추출
        active_idx = np.argmax(d["active_player"])
        my_pos = d["left_team"][active_idx*2 : active_idx*2+2]
        current_ball_owned_by_us = (d["ball_ownership"][1] == 1.0)
        current_ball_x = d["ball"][0]
        goal_pos = np.array([1.0, 0.0])
        dist_to_goal = np.linalg.norm(my_pos - goal_pos)

        if current_ball_owned_by_us:
            # 1. 전진 압박 (백패스 엄단)
            if self.prev_ball_x is not None:
                if current_ball_x < self.prev_ball_x:
                    reward -= (self.prev_ball_x - current_ball_x) * 5.0 # 페널티 강화
                else:
                    reward += (current_ball_x - self.prev_ball_x) * 1.5 # 전진 보상 강화

            # 2. 슈팅 결단력 (Decisiveness)
            # 골대 근처(x > 0.7)에서 슛을 안 쏘고 머뭇거리면 매 스텝 감점
            if current_ball_x > 0.7:
                if action != 12: # 12번이 아닌 다른 행동을 할 때
                    reward -= 0.05 
                else: # 슛을 쐈을 때 (Shot Accuracy)
                    # 골대와 가까울수록, 그리고 중앙에 가까울수록 보너스
                    accuracy_bonus = max(0, 1.0 - dist_to_goal)
                    reward += accuracy_bonus * 2.0 

            # 3. 골키퍼 무력화 (Open Goal)
            keeper_pos = d["right_team"][0:2]
            keeper_dist_from_goal = np.linalg.norm(keeper_pos - goal_pos)
            if keeper_dist_from_goal > 0.15 and action == 12:
                reward += 1.0 # 빈집 털기 성공 시 파격 보상

        # 4. 득점 성공 (80% 도달을 위한 최종 보상)
        self.steps_from_kickoff += 1
        if reward > 0: # 득점 발생
            # 빨리 넣을수록 보너스 (최대 10점 이상)
            time_bonus = max(0, (1500 - self.steps_from_kickoff) * 0.01)
            reward += (10.0 + time_bonus)
            self.steps_from_kickoff = 0

        self.prev_ball_x = current_ball_x
        return obs, reward, done, info

    def obs_to_dict(self, obs):
        return {
            "left_team": obs[0:22],
            "right_team": obs[44:66],
            "ball": obs[88:91],
            "ball_ownership": obs[94:97],
            "active_player": obs[97:108],
            "game_mode": obs[108:115],
        }