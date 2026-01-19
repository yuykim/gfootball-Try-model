import gym
import numpy as np
from gfootball.env import football_action_set

class RuleBasedAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def obs_to_dict(self, obs):
        obs_dict = {}
        obs_dict["left_team"] = obs[0:22]
        obs_dict["left_team_direction"] = obs[22:44]
        obs_dict["right_team"] = obs[44:66]
        obs_dict["right_team_direction"] = obs[66:88]
        obs_dict["ball"] = obs[88:91]
        obs_dict["ball_direction"] = obs[91:94]
        obs_dict["ball_ownership"] = obs[94:97] # [무소속, 내팀, 상대팀]
        obs_dict["active_player"] = obs[97:108]
        obs_dict["game_mode"] = obs[108:115]
        return obs_dict

    def get_action(self, obs):
        d = self.obs_to_dict(obs)
        
        # 1. 내 정보
        active_idx = np.argmax(d["active_player"])
        my_x = d["left_team"][active_idx * 2]
        my_y = d["left_team"][active_idx * 2 + 1]
        
        # 2. 공 정보
        ball_x = d["ball"][0]
        ball_y = d["ball"][1]
        is_my_ball = d["ball_ownership"][1] == 1.0
        
        # 3. 상대 팀 정보 (right_team: 11명의 x, y 좌표가 순서대로 들어있음)
        opponents = d["right_team"].reshape(11, 2)
        
        # --- [추가 조건: 주변 상황 판단] ---
        # 내 앞에 상대가 있는가? (나보다 x값이 크고, y값이 비슷한 적)
        opponents_in_front = [opt for opt in opponents if opt[0] > my_x and abs(opt[1] - my_y) < 0.9]
        
        # 내 뒤에 상대가 있는가? (나보다 x값이 작고, 거리가 가까운 적)
        opponents_behind = [opt for opt in opponents if opt[0] < my_x and (my_x - opt[0]) < 0.2]

        # --- 규칙 설계 ---
        
        if not is_my_ball:
            # [수정] 공을 쫓아갈 때도 내 앞에 아무도 없으면 스프린트하며 접근
            if len(opponents_in_front) != 0:
                # print("Do sprint!")
                return football_action_set.action_sprint
            
            if ball_x > my_x + 0.01: return football_action_set.action_right
            if ball_x < my_x - 0.01: return football_action_set.action_left
            if ball_y > my_y + 0.01: return football_action_set.action_bottom
            if ball_y < my_y - 0.01: return football_action_set.action_top
            return football_action_set.action_idle

        else:
            # [Rule 2] 내가 공을 가졌을 때
            
            # 1. 슈팅 거리면 슛
            if my_x > 0.7:
                return football_action_set.action_shot
            
            # 2. [사용자 요청] 내 앞은 비어있고 뒤에서 누가 따라오면 전력 질주!
            if len(opponents_in_front) == 0 and len(opponents_behind) > 0:
                # 단, 골대 방향으로 가면서 스프린트해야 함
                return football_action_set.action_sprint
            
            # 3. 일반 드리블 전진
            if my_y > 0.05: return football_action_set.action_top_right
            if my_y < -0.05: return football_action_set.action_bottom_right
            return football_action_set.action_right