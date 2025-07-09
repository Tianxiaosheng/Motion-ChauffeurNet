

from collections import deque, namedtuple       # 队列类型
import math
import random
from motion_scenario_demo.waymo_data_load import WaymoScenarioDataset
from ogm import OccupancyGrid

import os
import sys
SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(SCRIPT_PATH)
sys.path.append(os.path.join(SCRIPT_PATH, 'motion_scenario_demo/py_protos'))

from waymo_open_dataset.protos.scenario_pb2 import Scenario
from waymo_open_dataset.protos.scenario_pb2 import Track

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    def sample(self, batch_size):
        batch_data = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch_data)
        return state, action, reward, next_state, done

    def push(self, *args):
        # *args: 把传进来的所有参数都打包起来生成元组形式
        # self.push(1, 2, 3, 4, 5)
        # args = (1, 2, 3, 4, 5)
        self.memory.append(self.Transition(*args))

    def get(self, frame):
        if frame < len(self.memory):
            return self.memory[frame]
        else:
            return -1

    def __len__(self):
        return len(self.memory)

    def print_frame(self, frame_index):
        if frame_index < 0 or frame_index >= len(self.memory):
            print("帧索引超出范围。")
        else:
            transition = list(self.memory)[frame_index]
            print(f"帧 {frame_index}:=========================")
            print(f"动作: {transition.action}")
            print(f"奖励: {transition.reward}")
            print(f"完成: {transition.done}")

    def get_frame(self, frame_index):
        if frame_index < 0 or frame_index >= len(self.memory):
            print("帧索引超出范围。")
        else:
            return list(self.memory)[frame_index]

class Dataset:
    def __init__(self, observation_dim=(6, 300, 600), file_path=None):
        delta_x = 0.2
        delta_y = 0.2
        self.render = False
        self.file_path = file_path
        self.replay_memory = DQNReplayer(capacity=40000)
        self.ogm = OccupancyGrid(observation_dim, delta_x=delta_x, delta_y=delta_y,
                                 render=self.render)

    def load_replay_memory(self):
        # every scenario has 200 records
        self.scenarios = WaymoScenarioDataset(self.file_path)
        #self.fill_replay_memory(self.scenarios.get_scenario(1))

    def fill_replay_memory(self, scenario):
        state = self.get_observation_from_scenario(scenario, 0)
        for frame in range(1, len(scenario), 1):
            next_state = self.get_observation_from_scenario(scenario, frame)
            action = self.get_action_from_scenario(scenario, frame-1)
            reward = self.update_reward(scenario[frame-1])
            done = False

            ego_info = self.deserialization.get_ego_info_from_scenario(\
                scenario[frame-1])
            if ego_info.vel == 0 and ego_info.prev_cmd_acc <= 0:
                continue
            self.replay_memory.push(state, action, reward, next_state, done)
            state = next_state

    def dump_observateion(self, frame):
        scenario = self.scenarios.get_scenario(1)
        state = self.get_observation_from_scenario(scenario, frame)
        self.ogm.dump_ogm_graphs(state)


    def get_action_from_scenario(self, scenario, frame):
        pass


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
                    map_point_xyv.append([pt.x, pt.y, speed_limit_mph])

        self.ogm.update_occupancy_grid_by_route_and_speed_limit(map_point_xyv)

    def update_ogm_by_traffic_light(self, scenario, frame):
        map_point_xyt = []
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
        return ego_state.velocity_x

    def get_obj_state_from_obj_info(self, track, frame):
        return track.states[frame]

    def get_obj_vel_from_obj_info(self, track, frame):
        obj_state = self.get_obj_state_from_obj_info(track, frame)
        return obj_state.velocity_x

    def get_obj_width_from_obj_info(self, track, frame):
        obj_state = self.get_obj_state_from_obj_info(track, frame)
        return obj_state.width

    def get_obj_length_from_obj_info(self, track, frame):
        obj_state = self.get_obj_state_from_obj_info(track, frame)
        return obj_state.length

    def get_obj_heading_from_obj_info(self, track, frame):
        obj_state = self.get_obj_state_from_obj_info(track, frame)
        return (obj_state.heading % (math.pi * 2.0))


FILENAME = '/home/uisee/Downloads/uncompressed_scenario_training_20s_training_20s.tfrecord-00006-of-01000'
dataset = Dataset(observation_dim=(8, 100, 200), file_path=FILENAME)
dataset.load_replay_memory()
dataset.dump_observateion(101)
