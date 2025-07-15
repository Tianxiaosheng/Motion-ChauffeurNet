import tensorflow as tf
import numpy as np
import os
import sys

import torch
from torch.utils.data import Dataset


SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(SCRIPT_PATH)
sys.path.append(os.path.join(SCRIPT_PATH, 'py_protos'))

from waymo_open_dataset.protos.scenario_pb2 import Scenario
from waymo_open_dataset.protos.scenario_pb2 import Track


def handle_one_tfrecord_file(tf_record_file):
    raw_dataset = tf.data.TFRecordDataset(tf_record_file)
    for i, record in enumerate(raw_dataset):
        scenario = Scenario()
        scenario.ParseFromString(record.numpy())

        print(f'{i}st sceanrio id is {scenario.scenario_id}')

        # for obj_track in scenario.tracks:
        #     print(f'obj_id: {obj_track.id}, obj type: {obj_track.object_type}')

class WaymoScenarioDataset(Dataset):
    def __init__(self, tfrecord_path):

        self.tfrecord_path = tfrecord_path
        # self.dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
        print("正在缓存所有TFRecord帧到内存...")
        self._raw_records = list(tf.data.TFRecordDataset(tfrecord_path, compression_type=''))
        print(f"已缓存 {len(self._raw_records)} 帧。")

    def __len__(self):
        return len(self._raw_records)

    def get_scenario(self, idx):
        data = self._raw_records[idx]
        scenario = Scenario()
        scenario.ParseFromString(data.numpy())

        return scenario

    def _parse_example(self, idx):
        data = self._raw_records[idx]
        scenario = Scenario()
        scenario.ParseFromString(data.numpy())

        # 轨迹相关: Agent数量, 时间步数
        num_agents = len(scenario.tracks)
        num_steps = len(scenario.timestamps_seconds)

        print(f"num_agents: {num_agents}, num_steps: {num_steps}, current_time_index: {scenario.current_time_index}")

        # 轨迹状态初始化
        past_x, past_y, past_z = [], [], []
        past_length, past_width, past_height, past_bbox_yaw = [], [], [], []
        past_valid = []

        current_x, current_y, current_z = [], [], []
        current_length, current_width, current_height, current_bbox_yaw = [], [], [], []
        current_valid, current_timestamp_micros = [], []
        
        future_x, future_y, future_z = [], [], []
        future_length, future_width, future_height, future_bbox_yaw = [], [], [], []
        future_valid = []

        is_sdc = []
        agent_type = []
        agent_id = []

        # 在每个agent的所有时刻状态前，增加速度相关list
        past_velocity_x, past_velocity_y = [], []
        current_velocity_x, current_velocity_y = [], []
        future_velocity_x, future_velocity_y = [], []

        # 遍历每个agent
        for agent_idx, track in enumerate(scenario.tracks):
            # agent id
            agent_id.append(track.id)
            # agent type
            agent_type.append(track.object_type)
            # 是否是SDC
            is_sdc.append(1 if agent_idx == scenario.sdc_track_index else 0)

            # 每个agent的所有时刻状态
            px, py, pz = [], [], []
            pl, pw, ph, pyaw = [], [], [], []
            pvalid = []

            cx, cy, cz = [], [], []
            cl, cw, ch, cyaw = [], [], [], []
            cvalid, cts = [], []

            fx, fy, fz = [], [], []
            fl, fw, fh, fyaw = [], [], [], []
            fvalid = []

            pvx, pvy = [], []
            cvx, cvy = [], []
            fvx, fvy = [], []

            for t, state in enumerate(track.states):
                # 这里假设current_time_index前为past，等于为current，后为future
                if t < scenario.current_time_index:
                    px.append(state.center_x)
                    py.append(state.center_y)
                    pz.append(state.center_z)
                    pl.append(state.length)
                    pw.append(state.width)
                    ph.append(state.height)
                    pyaw.append(state.heading)
                    pvalid.append(state.valid)
                    pvx.append(state.velocity_x)
                    pvy.append(state.velocity_y)
                elif t == scenario.current_time_index:
                    cx.append(state.center_x)
                    cy.append(state.center_y)
                    cz.append(state.center_z)
                    cl.append(state.length)
                    cw.append(state.width)
                    ch.append(state.height)
                    cyaw.append(state.heading)
                    cvalid.append(state.valid)
                    cts.append(scenario.timestamps_seconds[t])
                    cvx.append(state.velocity_x)
                    cvy.append(state.velocity_y)
                else:
                    fx.append(state.center_x)
                    fy.append(state.center_y)
                    fz.append(state.center_z)
                    fl.append(state.length)
                    fw.append(state.width)
                    fh.append(state.height)
                    fyaw.append(state.heading)
                    fvalid.append(state.valid)
                    fvx.append(state.velocity_x)
                    fvy.append(state.velocity_y)

            # 填充到总list
            past_x.append(px)
            past_y.append(py)
            past_z.append(pz)
            past_length.append(pl)
            past_width.append(pw)
            past_height.append(ph)
            past_bbox_yaw.append(pyaw)
            past_valid.append(pvalid)
            past_velocity_x.append(pvx)
            past_velocity_y.append(pvy)


            current_x.append(cx)
            current_y.append(cy)
            current_z.append(cz)
            current_length.append(cl)
            current_width.append(cw)
            current_height.append(ch)
            current_bbox_yaw.append(cyaw)
            current_valid.append(cvalid)
            current_timestamp_micros.append(cts)
            current_velocity_x.append(cvx)
            current_velocity_y.append(cvy)


            future_x.append(fx)
            future_y.append(fy)
            future_z.append(fz)
            future_length.append(fl)
            future_width.append(fw)
            future_height.append(fh)
            future_bbox_yaw.append(fyaw)
            future_valid.append(fvalid)
            future_velocity_x.append(fvx)
            future_velocity_y.append(fvy)

        # roadgraph相关（这里只做简单示例，具体字段需查 scenario.map_features）
        feature_type_map = {
            "lane": 1
            # "road_line": 2,
            # "road_edge": 3
        }

        roadgraph_xyz = []
        roadgraph_valid = []
        roadgraph_type = []
        roadgraph_id = []
        roadgraph_dir = []

        for feature in scenario.map_features:
            feature_type = feature.WhichOneof("feature_data")
            if feature_type in feature_type_map:
                polyline = getattr(feature, feature_type).polyline
                for pt in polyline:
                    roadgraph_xyz.append([pt.x, pt.y, pt.z])
                    roadgraph_valid.append(1)
                    roadgraph_type.append(feature_type_map[feature_type])
                    roadgraph_id.append(feature.id)
                    roadgraph_dir.append([0, 0, 0])  # MapPoint 没有方向信息

        # 转为tensor
        to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
        to_int_tensor = lambda x: torch.tensor(x, dtype=torch.int64)
        to_bool_tensor = lambda x: torch.tensor(x, dtype=torch.bool)

        return {
            'roadgraph_xyz': to_tensor(roadgraph_xyz),
            'roadgraph_valid': to_int_tensor(roadgraph_valid),
            'roadgraph_type': to_int_tensor(roadgraph_type),
            'roadgraph_id': to_int_tensor(roadgraph_id),
            'roadgraph_dir': to_tensor(roadgraph_dir),

            'past_x': to_tensor(past_x),
            'past_y': to_tensor(past_y),
            'past_z': to_tensor(past_z),
            'past_valid': to_bool_tensor(past_valid),
            'past_length': to_tensor(past_length),
            'past_width': to_tensor(past_width),
            'past_height': to_tensor(past_height),
            'past_bbox_yaw': to_tensor(past_bbox_yaw),
            'past_velocity_x': to_tensor(past_velocity_x),
            'past_velocity_y': to_tensor(past_velocity_y),

            'current_x': to_tensor(current_x),
            'current_y': to_tensor(current_y),
            'current_z': to_tensor(current_z),
            'current_valid': to_bool_tensor(current_valid),
            'current_length': to_tensor(current_length),
            'current_width': to_tensor(current_width),
            'current_timestamp_micros': to_tensor(current_timestamp_micros),
            'current_height': to_tensor(current_height),
            'current_bbox_yaw': to_tensor(current_bbox_yaw),
            'current_velocity_x': to_tensor(current_velocity_x),
            'current_velocity_y': to_tensor(current_velocity_y),

            'future_x': to_tensor(future_x),
            'future_y': to_tensor(future_y),
            'future_z': to_tensor(future_z),
            'future_valid': to_bool_tensor(future_valid),
            'future_length': to_tensor(future_length),
            'future_width': to_tensor(future_width),
            'future_height': to_tensor(future_height),
            'future_bbox_yaw': to_tensor(future_bbox_yaw),
            'future_velocity_x': to_tensor(future_velocity_x),
            'future_velocity_y': to_tensor(future_velocity_y),

            'is_sdc': to_int_tensor(is_sdc),
            'agent_type': to_int_tensor(agent_type),
            'agent_id': to_int_tensor(agent_id),
        }

    def __getitem__(self, idx):
        return self._parse_example(idx)

    def to_numpy_dict(self, idx):
        """Convert to numpy dictionary for visualization"""
        data = self._parse_example(idx)
        return {k: v.numpy() if torch.is_tensor(v) else v for k, v in data.items()}


# FILENAME = '/home/uisee/Downloads/uncompressed_scenario_training_20s_training_20s.tfrecord-00006-of-01000'
# handle_one_tfrecord_file(FILENAME)
