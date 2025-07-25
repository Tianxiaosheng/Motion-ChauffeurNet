from collections import deque, namedtuple       # 队列类型
import math
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import sys
SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(SCRIPT_PATH)
sys.path.append(os.path.join(SCRIPT_PATH, 'motion_scenario_demo/py_protos'))

from ogm import OccupancyGrid
from motion_scenario_demo.waymo_data_load import WaymoScenarioDataset

from waymo_open_dataset.protos.scenario_pb2 import Scenario
from waymo_open_dataset.protos.scenario_pb2 import Track

mph_to_ms = 0.44704

def normalize_neg(x):
    return math.exp(-x) / (1 + math.exp(-x)) * 2

def normalize_pos(x):
    return (1 - math.exp(-math.pow(x / 2, 3))) /\
            (1 + math.exp(-math.pow(x / 2, 3)))

class Frame_Dataset(Dataset):
    def __init__(self, observation_dim=(8, 100, 200), tf_file_path=None):
        delta_x = 0.6
        delta_y = 0.6
        self.render = False
        self.tf_file_path = tf_file_path
        self.ogm = OccupancyGrid(observation_dim, delta_x=delta_x, delta_y=delta_y,
                                 render=self.render)
        self.frame_len = 0
        self.frame_index_list = []

        # 新增：场景缓存
        self._scenario_cache = {}
        self._max_cache_size = 100  # 可根据内存调整

        self.scenarios = WaymoScenarioDataset(self.tf_file_path)
        self.calc_frame_len()

    def __len__(self):
        return self.frame_len

    def __getitem__(self, idx):
        scenario_idx, frame_idx = self.frame_index_list[idx]
        #print("frame_idx:{}, scenario_idx:{}, frame_idx:{}".format(idx, scenario_idx, frame_idx))
        scenario = self._get_scenario_with_cache(scenario_idx)

        state = self.get_observation_from_scenario(scenario, frame_idx).astype(np.float32)
        next_state = self.get_observation_from_scenario(scenario, frame_idx+1).astype(np.float32)

        action = np.expand_dims(self.get_action_from_scenario(scenario, frame_idx), axis=-1)
        reward = np.expand_dims(self.get_reward_from_scenario(scenario, frame_idx), axis=-1).astype(np.float32)
        done = np.expand_dims(False, axis=-1).astype(np.float32)

        return state, action, reward, next_state, done

    def _get_scenario_with_cache(self, scenario_idx):
        # 优先从缓存取
        if scenario_idx in self._scenario_cache:
            return self._scenario_cache[scenario_idx]
        # 没有则解析并加入缓存
        scenario = self.scenarios.get_scenario(scenario_idx)
        if len(self._scenario_cache) >= self._max_cache_size:
            # 简单FIFO策略，弹出最早加入的
            self._scenario_cache.pop(next(iter(self._scenario_cache)))
        self._scenario_cache[scenario_idx] = scenario
        return scenario

    def calc_frame_len(self):
        for scenario_idx in range(min(len(self.scenarios), 500)):
            scenario = self._get_scenario_with_cache(scenario_idx)
            num_frames = self.scenarios.get_scenario_frame_size(scenario)-1
            for frame_idx in range(num_frames):
                self.frame_index_list.append((scenario_idx, frame_idx))
        self.frame_len = len(self.frame_index_list)

    def calc_risk_value_from_scenario(self, scenario, frame, render=False):
        sdc_track_index = scenario.sdc_track_index
        ego_state = scenario.tracks[sdc_track_index].states[frame]
        ego_length = ego_state.length
        ego_width = ego_state.width
        ego_heading = ego_state.heading

        if render:
            print("ego's pos->x:{}, y:{}, heading:{}, width:{}, length:{}, vel_x:{}, vel_y:{}"\
                  .format(ego_state.center_x, ego_state.center_y,\
                          ego_state.heading, ego_state.width, ego_state.length,\
                          ego_state.velocity_x, ego_state.velocity_y))

        min_ttc = float('inf')
        # 最近点碰撞时间法
        for i, track in enumerate(scenario.tracks):
            # if track.id not in scenario.objects_of_interest:
            #     continue
            if i == scenario.sdc_track_index:
                continue
            obj_state = track.states[frame]
            if not obj_state.valid:
                continue

            obj_length = obj_state.length
            obj_width = obj_state.width
            obj_heading = obj_state.heading

            if render and False:
                print("{}'th obj:[{}] 's pos->x:{}, y:{}, heading:{}, width:{}, length:{}, vel_x:{}, vel_y:{}"\
                      .format(i, track.id, obj_state.center_x, obj_state.center_y,\
                              obj_state.heading, obj_width, obj_length, obj_state.velocity_x, obj_state.velocity_y))

            # 1. 相对位置和速度
            dx = ego_state.center_x - obj_state.center_x
            dy = ego_state.center_y - obj_state.center_y
            dvx = ego_state.velocity_x - obj_state.velocity_x
            dvy = ego_state.velocity_y - obj_state.velocity_y

            d0 = np.array([dx, dy])
            v = np.array([dvx, dvy])
            v_norm_sq = np.dot(v, v)

            # 计算两车包络和
            # 车辆朝向与连线方向的夹角
            dir_x = dx / (np.linalg.norm(d0) + 1e-8)
            dir_y = dy / (np.linalg.norm(d0) + 1e-8)
            ego_theta = np.arctan2(dir_y, dir_x) - ego_heading
            obj_theta = np.arctan2(-dir_y, -dir_x) - obj_heading
            ego_half_extent = 0.5 * abs(ego_length * np.cos(ego_theta)) + 0.5 * abs(ego_width * np.sin(ego_theta))
            obj_half_extent = 0.5 * abs(obj_length * np.cos(obj_theta)) + 0.5 * abs(obj_width * np.sin(obj_theta))
            envelope = ego_half_extent + obj_half_extent

            # 最近点时刻
            if v_norm_sq < 1e-8:
                t_star = 0.0
            else:
                t_star = -np.dot(d0, v) / v_norm_sq

            # 最近点距离
            closest_dist = np.linalg.norm(d0 + v * t_star)

            if render and False:
                print(f"t_star:{t_star}, closest_dist:{closest_dist}, envelope:{envelope}")

            # 判断碰撞条件
            if t_star > 0 and closest_dist <= envelope:
                ttc = t_star
            else:
                ttc = float('inf')

            if render and ttc < float('inf'):
                print("{}'th obj:[{}] 's pos->x:{}, y:{}, heading:{}, width:{}, length:{}, vel_x:{}, vel_y:{}"\
                      .format(i, track.id, obj_state.center_x, obj_state.center_y,\
                              obj_state.heading, obj_width, obj_length, obj_state.velocity_x, obj_state.velocity_y))
                print("TTC:{}, dx:{}, dy:{}, dvx:{}, dvy:{}, v_norm_sq:{}, ego_half_extent:{}, obj_half_extent:{}, envelope:{}".\
                      format(ttc, dx, dy, dvx, dvy, v_norm_sq, ego_half_extent, obj_half_extent, envelope))

            if ttc < min_ttc:
                min_ttc = ttc

        if render:
            print("frame{}'s min_ttc:{}".format(frame, min_ttc))
        return normalize_neg(min_ttc)

    def calc_efficiency_value_from_scenario(self, scenario, frame, render=False):
        sdc_track_index = scenario.sdc_track_index
        ego_state = scenario.tracks[sdc_track_index].states[frame]
        ego_vel = self.get_ego_vel_from_scenario(scenario, frame)
        ego_target_vel = 50.0

        feature_type_map = {
            "lane": 1
            # "road_line": 2,
            # "road_edge": 3
        }

        min_dist = float('inf')
        for feature in scenario.map_features:
            feature_type = feature.WhichOneof("feature_data")
            if feature_type in feature_type_map:
                polyline = getattr(feature, feature_type).polyline
                speed_limit_mph = getattr(feature, feature_type).speed_limit_mph
                for pt in polyline:
                    tmp_dist = math.hypot(pt.x-ego_state.center_x, pt.y-ego_state.center_y)
                    if tmp_dist < min_dist:
                        min_dist = tmp_dist
                        ego_target_vel = speed_limit_mph

        ego_target_vel  = ego_target_vel * mph_to_ms
        if (ego_target_vel > 0.0):
            payoff_of_efficiency = pow(abs(ego_vel - ego_target_vel) / ego_target_vel, 2)
        else:
            payoff_of_efficiency = 0.0
        if render:
            print("frame{}'s ego_vel:{} tareget_vel:{}, normalize:{}"\
                  .format(frame, ego_vel, ego_target_vel, payoff_of_efficiency))
        return payoff_of_efficiency

    def get_reward_from_scenario(self, scenario, frame, render=False):
        k_e = 20.0
        k_c = 1000.0
        reward_risk = -k_c * self.calc_risk_value_from_scenario(scenario, frame, render)
        reward_efficiency = -k_e * self.calc_efficiency_value_from_scenario(scenario, frame, render)
        reward = reward_risk + reward_efficiency

        if render:
            print("reward_risk:{}, reward_efficiency:{}, reward:{}"\
                  .format(reward_risk, reward_efficiency, reward))

        return reward

    # action: acc 
    def get_action_from_scenario(self, scenario, frame):
        actions = [-3.0, -1.0, -0.2, 0.0, 0.2, 1.0]
        delay_time = 0.5
        observe_window = 0.5

        calc_acc = 0.0
        timestamps_seconds = scenario.timestamps_seconds
        sdc_track_index = scenario.sdc_track_index
        states = scenario.tracks[sdc_track_index].states
        times = []
        speeds = []
        curr_time = timestamps_seconds[frame]

        for i, state in enumerate(scenario.tracks[sdc_track_index].states[frame:]):
            if frame+i >= len(scenario.timestamps_seconds):
                break
            if (timestamps_seconds[frame+i] - curr_time) < delay_time:
                continue
            if (timestamps_seconds[frame+i] - curr_time - delay_time > observe_window):
                break
            times.append(timestamps_seconds[frame+i])
            speeds.append(state.velocity_x)
            #print("[{}, {}]".format(timestamps_seconds[frame+i], state.velocity_x))
        if len(times) >= 2:
            calc_acc, intercept = np.polyfit(np.array(times), np.array(speeds), 1)

        # 直接计算最接近的动作索引
        closest_index = min(range(len(actions)), key=lambda i: abs(actions[i] - calc_acc))
        return closest_index

    def get_observation_from_scenario(self, scenario, frame):
        ego_state = self.get_ego_state_from_scenario(scenario, frame)
        ego_pos = [ego_state.center_x, ego_state.center_y]
        ego_heading = self.get_ego_heading_from_scenario(scenario, frame)

        width = self.get_ego_width_from_scenario(scenario, frame)
        length = self.get_ego_length_from_scenario(scenario, frame)
        ego_vel = self.get_ego_vel_from_scenario(scenario, frame)
        ego_bounding_box = [length, width]

        self.ogm.preprocess_occupancy_grid(ego_pos, ego_heading)

        # channel 0, 2
        self.update_ogm_by_map_feature(scenario)

        # channel 1
        self.update_ogm_by_traffic_light(scenario, frame)

        # channel 3
        self.update_ogm_by_ego_path(scenario, frame)

        # channel 4, 5
        self.ogm.update_occupancy_grid_by_obj(ego_pos, ego_heading, ego_vel, \
                                       ego_bounding_box, True)
        for obj_idx, obj_info in enumerate(scenario.tracks):
            if obj_idx == scenario.sdc_track_index:
                continue
            obj_state = self.get_obj_state_from_obj_info(obj_info, frame)
            if not obj_state.valid:
                continue
            obj_pos = [obj_state.center_x, obj_state.center_y]

            obj_vel = self.get_obj_vel_from_obj_info(obj_info, frame)
            obj_bounding_box = [self.get_obj_length_from_obj_info(obj_info, frame), \
                                self.get_obj_width_from_obj_info(obj_info, frame)]
            obj_heading = self.get_obj_heading_from_obj_info(obj_info, frame)

            # channel 6, 7
            self.ogm.update_occupancy_grid_by_obj(obj_pos, obj_heading, obj_vel, \
                                            obj_bounding_box, False)

        return self.ogm.grid

    def update_ogm_by_map_feature(self, scenario):
        feature_type_map = {
            "lane": 1
            # "road_line": 2,
            # "road_edge": 3
        }
        map_point_xyv = []
        for feature in scenario.map_features:
            feature_type = feature.WhichOneof("feature_data")
            if feature_type in feature_type_map:
                polyline = getattr(feature, feature_type).polyline
                speed_limit_mph = getattr(feature, feature_type).speed_limit_mph
                for pt in polyline:
                    map_point_xyv.append([pt.x, pt.y, speed_limit_mph * mph_to_ms])

        self.ogm.update_occupancy_grid_by_route_and_speed_limit(map_point_xyv)

    def update_ogm_by_traffic_light(self, scenario, frame):
        map_point_xyt = []
        if frame >= len(scenario.dynamic_map_states):
            return
        for traffic_signal_lane_state in scenario.dynamic_map_states[frame].lane_states:
            lane_state = traffic_signal_lane_state.state
            stop_point = traffic_signal_lane_state.stop_point
            tmp_xyt = []
            if lane_state == 4: # LANE_STATE_STOP
                tmp_xyt = [stop_point.x, stop_point.y, 2]
            elif lane_state == 5: # LANE_STATE_CAUTION
                tmp_xyt = [stop_point.x, stop_point.y, 1]
            else:
                tmp_xyt = [stop_point.x, stop_point.y, 3]
            map_point_xyt.append(tmp_xyt)

        self.ogm.update_occupancy_grid_by_traffic_light(map_point_xyt)

    def calc_dist(self, p1, p2):
        return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1]-p2[1], 2))

    def update_ogm_by_ego_path(self, scenario, frame):
        ego_track = scenario.tracks[scenario.sdc_track_index]
        ego_xy = []
        accumulate_dist = 0.0
        tmp_xy = []
        for i, ego_state in enumerate(ego_track.states[frame:]):
            if not ego_state.valid:
                break
            if i > 0:
                accumulate_dist += self.calc_dist(tmp_xy, \
                        [ego_state.center_x, ego_state.center_y])
            tmp_xy = [ego_state.center_x, ego_state.center_y]
            ego_xy.append(tmp_xy)
            if accumulate_dist > 60:
                break

        self.ogm.update_occupancy_grid_by_ego_path(ego_xy)

    def get_ego_state_from_scenario(self, scenario, frame):
        sdc_track_index = scenario.sdc_track_index
        return scenario.tracks[sdc_track_index].states[frame]
    
    def get_ego_width_from_scenario(self, scenario, frame):
        ego_state = self.get_ego_state_from_scenario(scenario, frame)
        return ego_state.width

    def get_ego_length_from_scenario(self, scenario, frame):
        ego_state = self.get_ego_state_from_scenario(scenario, frame)
        return ego_state.length

    def get_ego_heading_from_scenario(self, scenario, frame):
        ego_state = self.get_ego_state_from_scenario(scenario, frame)
        return (ego_state.heading  % (math.pi * 2.0))

    def get_ego_vel_from_scenario(self, scenario, frame):
        ego_state = self.get_ego_state_from_scenario(scenario, frame)
        return math.hypot(ego_state.velocity_x, ego_state.velocity_y)

    def get_obj_state_from_obj_info(self, track, frame):
        return track.states[frame]

    def get_obj_vel_from_obj_info(self, track, frame):
        obj_state = self.get_obj_state_from_obj_info(track, frame)
        if not obj_state.valid:
            return 0.0
        #print("heading:{}, velocity_x:{}, velocity_y:{}".format(obj_state.heading, obj_state.velocity_x, obj_state.velocity_y))
        return math.hypot(obj_state.velocity_x, obj_state.velocity_y)

    def get_obj_width_from_obj_info(self, track, frame):
        obj_state = self.get_obj_state_from_obj_info(track, frame)
        if not obj_state.valid:
            return 0.0
        return obj_state.width

    def get_obj_length_from_obj_info(self, track, frame):
        obj_state = self.get_obj_state_from_obj_info(track, frame)
        if not obj_state.valid:
            return 0.0
        return obj_state.length

    def get_obj_heading_from_obj_info(self, track, frame):
        obj_state = self.get_obj_state_from_obj_info(track, frame)
        if not obj_state.valid:
            return 0.0
        return (obj_state.heading % (math.pi * 2.0))

    def dump_observateion(self, frame):
        print("frame_nums:{}".format(self.frame_len))
        if self.frame_len > frame:
            item = self.__getitem__(0)
            self.ogm.dump_ogm_graphs(item[0])

# FILENAME = 'data/training_20s'
# file_path_list = list(iter_tf_files(FILENAME))
# print("file_list_size:{}".format(len(file_path_list)))
# if len(file_path_list) > 0:
#     dataset = Frame_Dataset(observation_dim=(8, 100, 200), tf_file_path = file_path_list[0])
#     train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#     states, actions, rewards, next_states, dones = next(iter(train_dataloader))
#     print(type(states))
#     print("states_size:{}, actions_size:{}, rewards_size:{}, next_reward:{}, Dones_size:{}".\
#           format(states.size(), actions.size(), rewards.size(), next_states.size(), dones.size()))