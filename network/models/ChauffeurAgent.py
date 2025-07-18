import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys

MODEL_PATH = os.path.abspath(os.path.dirname(__file__))
NETWORK_PATH = os.path.dirname(MODEL_PATH)
PROJ_PATH = os.path.dirname(NETWORK_PATH)
DATASET_PATH = os.path.join(PROJ_PATH, 'dataset')

sys.path.append(DATASET_PATH)
from dataset import Dataset

EVA_PATH =  os.path.join(DATASET_PATH, "data/eva_model_net.pth")
TARGET_PATH = os.path.join(DATASET_PATH, "data/target_model_net.pth")

# 定义DQN网络结构
class FMDQNNet(nn.Module):
    def __init__(self, input_dim=(6, 300, 300),
                 conv_param={'filter1_size': 6, 'filter2_size': 16,
                                'filter_width': 5, 'pad': 0, 'stride': 1},
                                hidden1_size=128, hidden2_size=32, output_size=3):
        super(FMDQNNet, self).__init__()
        self.input_dim = input_dim
        filter1_size = conv_param['filter1_size']
        filter2_size = conv_param['filter2_size']
        filter_width = conv_param['filter_width']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        intput_channel_size = input_dim[0]
        input_height = input_dim[1]
        input_width = input_dim[2]
        conv1_height = 1 + (input_height + 2*filter_pad - filter_width) / \
                filter_stride
        conv1_width = 1 + (input_width + 2*filter_pad - filter_width) / \
                filter_stride

        conv2_height = 1 + (int(conv1_height / 2) + 2*filter_pad - filter_width) / \
                filter_stride
        conv2_width = 1 + (int(conv1_width / 2) + 2*filter_pad - filter_width) / \
                filter_stride

        pool2_output_size = int(filter2_size * int(conv2_height / 2) *
                               int(conv2_width / 2))
        # print("conv2_height{}, conv2_width{}, pool2_output_size{}, hidden1_size{}, hidden2_size{}, output_size{}".format(
        #         conv2_height, conv2_width, pool2_output_size, hidden1_size, hidden2_size, output_size))

        self.conv_block = nn.Sequential(
            nn.Conv2d(intput_channel_size, filter1_size, filter_width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=filter_pad),
            nn.Conv2d(filter1_size, filter2_size, filter_width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=filter_pad)
        )

        self.fc = nn.Sequential(
            nn.Linear(pool2_output_size, out_features=hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size)
        )

    def forward(self, x):
        x = self.conv_block(x)
        if x.dim() == len(self.input_dim):
            x = x.view(-1)
        else:
            x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# 定义DQNAgent类
class ChauffeurAgent:
    def __init__(self, net_kwargs={}, gamma=0.9, epsilon=0.05,\
                 batch_size=64, observation_dim=(4, 67, 133), action_size=3,
                 offline_RL_data_path=None):
        # action_size == 3,  0: -1m/s, 1: 0, 2: +1m/s
        super().__init__()
        self.offline_RL_data_path = offline_RL_data_path
        self.observation_dim = observation_dim
        self.action_n = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.count = 0
        delta_x = 1.0
        delta_y = 1.0
        self.render = False

        self.dataset = Dataset(observation_dim=observation_dim, file_path=offline_RL_data_path)

        # 初始化评估网络和目标网络
        self.evaluate_net = nn.DataParallel(FMDQNNet(input_dim=self.observation_dim,
                                     conv_param={'filter1_size': 6, 'filter2_size': 16,
                                                          'filter_width': 5, 'pad': 0, 'stride': 1},
                                     hidden1_size= 120, hidden2_size=84, output_size=self.action_n)).to(self.device)
        self.target_net = nn.DataParallel(FMDQNNet(input_dim=self.observation_dim,
                                     conv_param={'filter1_size': 6, 'filter2_size': 16,
                                                          'filter_width': 5, 'pad': 0, 'stride': 1},
                                     hidden1_size= 120, hidden2_size=84, output_size=self.action_n)).to(self.device)

        # 初始化目标网络的权重与评估网络相同
        self.update_target_net()

        # 定义优化器
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=net_kwargs.get('learning_rate', 0.001))

        # 定义损失函数
        self.criterion = nn.MSELoss()

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                 step_size=2000,
                                                 gamma=0.98)

        self.target_update_freq = 500  # 增加更新间隔
        self.tau = 0.001  # 减小软更新系数

        print("device is {}".format(self.device))

    def load_replay_memory(self):
        self.dataset.load_replay_memory()

    def update_target_net(self):
        self.target_net.load_state_dict(self.evaluate_net.state_dict())

    def learn(self,transition_dict):
        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1) # 扩充维度
        rewards = np.expand_dims(transition_dict.reward, axis=-1) # 扩充维度
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1) # 扩充维度

        # 优化后的代码:
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        predict_q_values = self.evaluate_net(states).gather(1, actions)

        # 计算目标Q值
        with torch.no_grad():
            # max(1) 即 max(dim=1)在行向找最大值，这样的话shape(64, ), 所以再加一个view(-1, 1)扩增至(64, 1)
            max_next_q_values = self.target_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        l = self.criterion(predict_q_values, q_targets)

        self.optimizer.zero_grad()
        l.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.evaluate_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()
 
        if self.count % 200 == 0:
            # copy model parameters
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
 
        self.count += 1

        if self.count % self.target_update_freq == 0:
            self.soft_update(self.tau)

    def decide(self, observation):
        # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)

        # 确保observation是numpy数组
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)

        # 添加batch维度和channel维度(如果需要)
        if observation.ndim == 3:  # 如果已经是[C,H,W]格式
            observation = observation[np.newaxis, ...]  # 变成[1,C,H,W]
        elif observation.ndim == 2:  # 如果是[H,W]格式
            observation = observation[np.newaxis, np.newaxis, ...]  # 变成[1,1,H,W]

        # 转换为tensor
        state = torch.from_numpy(observation).float().to(self.device)

        # 获取动作
        with torch.no_grad():
            q_values = self.evaluate_net(state)
            action = torch.argmax(q_values).item()

        return action

    def dump_ego_state(self, lon_decision_input):
        ego_pos = self.deserialization.get_ego_position_from_lon_decision_input(lon_decision_input)
        ego_pos = [ego_pos.x, ego_pos.y]
        ego_heading = self.deserialization.get_ego_heading_from_lon_decision_input(lon_decision_input)
        width = self.deserialization.get_ego_width_from_lon_decision_input(lon_decision_input)
        length = self.deserialization.get_ego_length_from_lon_decision_input(lon_decision_input)
        ego_vel = self.deserialization.get_ego_vel_from_lon_decision_input(lon_decision_input)

        print(f"ego->x:: {ego_pos[0]}, ego->y:: {ego_pos[1]}, theta: {ego_heading}, vel: {ego_vel}, width: {width}, length: {length}")

    def dump_obj_set(self, lon_decision_input):
        for obj_info in lon_decision_input.obj_set.obj_info:
            obj_pos = self.deserialization.get_obj_position_from_obj_info(obj_info)
            obj_pos = [obj_pos.x, obj_pos.y]
            obj_heading = self.deserialization.get_obj_heading_from_obj_info(obj_info)
            width = self.deserialization.get_obj_width_from_obj_info(obj_info)
            length = self.deserialization.get_obj_length_from_obj_info(obj_info)
            obj_vel = self.deserialization.get_obj_vel_from_obj_info(obj_info)
            obj_dtc = self.deserialization.get_obj_dtc_from_obj_info(obj_info)
            print(f"obj->x:: {obj_pos[0]}, obj->y:: {obj_pos[1]}, theta: {obj_heading}, vel: {obj_vel}, width: {width}, length: {length}, dtc: {obj_dtc}")

    def get_observation_from_lon_decision_input(self, lon_decision_input):
        ego_pos = self.deserialization.get_ego_position_from_lon_decision_input(lon_decision_input)
        ego_pos = [ego_pos.x, ego_pos.y]
        ego_heading = self.deserialization.get_ego_heading_from_lon_decision_input(lon_decision_input)

        width = self.deserialization.get_ego_width_from_lon_decision_input(lon_decision_input)
        length = self.deserialization.get_ego_length_from_lon_decision_input(lon_decision_input)
        ego_vel = self.deserialization.get_ego_vel_from_lon_decision_input(lon_decision_input)
        ego_bounding_box = [length, width]

        self.ogm.preprocess_occupancy_grid(ego_pos, ego_heading)
        self.ogm.update_occupancy_grid(ego_pos, ego_heading, ego_vel, \
                                       ego_bounding_box, 0.0)
        bird_eye_valid_dist = self.ogm.width * self.ogm.delta_x / 2.0

        for obj_idx, obj_info in enumerate(lon_decision_input.obj_set.obj_info):
            dist = self.deserialization.get_obj_dtc_from_obj_info(obj_info)
            if (dist >= 0) and (dist < bird_eye_valid_dist):
                obj_pos = self.deserialization.get_obj_position_from_obj_info(obj_info)
                obj_pos = [obj_pos.x, obj_pos.y]
                obj_vel = self.deserialization.get_obj_vel_from_obj_info(obj_info)
                obj_bounding_box = [self.deserialization.get_obj_length_from_obj_info(obj_info), \
                                    self.deserialization.get_obj_width_from_obj_info(obj_info)]
                obj_heading = self.deserialization.get_obj_heading_from_obj_info(obj_info)
                if obj_heading < 0.0:
                    obj_heading+=math.pi * 2.0

                self.ogm.update_occupancy_grid(obj_pos, obj_heading, obj_vel, \
                                                obj_bounding_box, dist)

        return self.ogm.grid

    def get_action_from_lon_decision_inputs(self, lon_decision_inputs, frame):
        # actions = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.0]
        actions = [-3.0, -1.0, -0.2, 0.0, 0.2, 0.5]

        if frame < len(lon_decision_inputs) - 1:
            ego_info = self.deserialization.get_ego_info_from_lon_decision_input(\
                lon_decision_inputs[frame + 1])
            # 直接计算最接近的动作索引
            closest_index = min(range(len(actions)), key=lambda i: abs(actions[i] - ego_info.prev_cmd_acc))
            return closest_index
        else:
            return 3  # 返回0.0对应的索引，假设0.0是默认动作

    def calc_safe_time_of_cross_obj(self, obj_info, ego_info):
        obj_time_to_cli =\
            self.deserialization.get_obj_time_to_cli_from_obj_info(obj_info)
        obj_time_to_leave_cli = \
            self.deserialization.get_obj_time_to_leave_cli_from_obj_info(obj_info)
        ego_time_to_cli = \
            self.deserialization.get_ego_time_to_cli(obj_info, ego_info)
        ego_time_to_leave_cli = \
            self.deserialization.get_ego_time_to_leave_cli(obj_info, ego_info)

        if (ego_time_to_cli == LON_DECISION_MAX_SAFE_T or\
                obj_time_to_cli == LON_DECISION_MAX_SAFE_T):
            safe_time = LON_DECISION_MAX_SAFE_T
        elif (obj_time_to_cli > ego_time_to_leave_cli or\
                ego_time_to_cli > obj_time_to_leave_cli):
            delta_time1 = abs(ego_time_to_cli - obj_time_to_leave_cli)
            delta_time2 = abs(ego_time_to_leave_cli - obj_time_to_cli)
            safe_time = min(delta_time1, delta_time2)
        else:
            safe_time = 0.0

        return safe_time

    def calc_safe_time_of_merge_obj(self, obj_info, ego_info):
        obj_time_to_cli =\
            self.deserialization.get_obj_time_to_cli_from_obj_info(obj_info)
        obj_time_to_leave_cli = \
            self.deserialization.get_obj_time_to_leave_cli_from_obj_info(obj_info)
        ego_time_to_cli = \
            self.deserialization.get_ego_time_to_cli(obj_info, ego_info)
        ego_time_to_leave_cli = \
            self.deserialization.get_ego_time_to_leave_cli(obj_info, ego_info)

        if (ego_time_to_cli == LON_DECISION_MAX_SAFE_T or\
                obj_time_to_cli == LON_DECISION_MAX_SAFE_T):
            safe_time = LON_DECISION_MAX_SAFE_T
        elif (obj_time_to_cli > ego_time_to_leave_cli or\
                ego_time_to_cli > obj_time_to_leave_cli):
            delta_time1 = abs(ego_time_to_cli - obj_time_to_leave_cli)
            delta_time2 = abs(ego_time_to_leave_cli - obj_time_to_cli)
            safe_time = min(delta_time1, delta_time2)
        else:
            safe_time = 0.0

        return safe_time

    def update_payoff_of_safety_bak(self, lon_decision_input):
        k_c = 2000
        min_safe_time = LON_DECISION_MAX_SAFE_T

        ego_info = self.deserialization.get_ego_info_from_lon_decision_input(lon_decision_input)
        for obj_info in lon_decision_input.obj_set.obj_info:
            if (obj_info.scene == SCENE_CROSS):
                safe_time = self.calc_safe_time_of_cross_obj(obj_info, ego_info)
            elif (obj_info.scene == SCENE_MERGE or\
                    obj_info.scene == SCENE_MERGE_LANE_CHANGE):
                safe_time = self.calc_safe_time_of_merge_obj(obj_info, ego_info)
            else:
                safe_time = self.deserialization.get_ego_time_to_cli(obj_info, ego_info)

            min_safe_time = min(min_safe_time, safe_time)

        normalized_safe_time = -k_c * normalize_neg(min_safe_time)

        return normalized_safe_time

    def update_payoff_of_safety(self, lon_decision_input):
        k_c = 1000
        min_safe_time = LON_DECISION_MAX_SAFE_T
        mass = 4.0
        max_risk_value = 0.0
        risk_value = 0.0

        ego_info = self.deserialization.get_ego_info_from_lon_decision_input(lon_decision_input)
        for obj_info in lon_decision_input.obj_set.obj_info:
            if (self.deserialization.get_obj_dtc_from_obj_info(obj_info) < 0.5):
                safe_time = self.deserialization.get_ego_time_to_cli(obj_info, ego_info)
                obj_vel = self.deserialization.get_obj_vel_from_obj_info(obj_info)
                ego_vel = self.deserialization.get_ego_vel_from_ego_info(ego_info)
                relative_vel = ego_vel - obj_vel
                dist_to_obj = self.deserialization.get_obj_intersection_start_s_from_obj_info(obj_info)
                if (relative_vel <= 0.0):
                    risk_value = 0.0
                elif (dist_to_obj <= 0.0):
                    risk_value = 5.0
                else:
                    risk_value = mass * pow(relative_vel, 3) / (2 * dist_to_obj)
                    risk_value = min(risk_value, 5.0)
                max_risk_value = max(max_risk_value, risk_value)

        normalized_risk_value = -k_c * normalize_pos(max_risk_value)
        return normalized_risk_value

    def update_payoff_of_comfort(self, lon_decision_input):
        k_a = 100
        ego_acc = self.deserialization.get_ego_acc_from_lon_decision_input\
                (lon_decision_input)
        ego_vel = self.deserialization.get_ego_vel_from_lon_decision_input\
                (lon_decision_input)
        if (ego_acc < 0.0 and ego_vel > 0.0):
            normalized_ego_acc = normalize_pos(min(-ego_acc, 5.0))
            payoff_of_comfort = -k_a * normalized_ego_acc
        else:
            payoff_of_comfort = 0.0

        return payoff_of_comfort

    def update_payoff_of_efficiency(self, lon_decision_input):
        k_e = 20

        ego_vel = self.deserialization.get_ego_vel_from_lon_decision_input\
                (lon_decision_input)
        ego_target_vel = self.deserialization.get_ego_max_vel_from_lon_decision_input\
                (lon_decision_input)
        ego_target_vel = 20 / 3.6
        if (ego_target_vel > 0.0):
            payoff_of_efficiency = -k_e * pow(abs(ego_vel - ego_target_vel) / ego_target_vel, 2)
        else:
            payoff_of_efficiency = 0.0

        return payoff_of_efficiency

    def update_reward(self, lon_decision_input):
        payoff_of_safety = self.update_payoff_of_safety(lon_decision_input)
        payoff_of_comfort = 0.0
        payoff_of_efficiency = self.update_payoff_of_efficiency(lon_decision_input)
        reward = payoff_of_safety + payoff_of_comfort + payoff_of_efficiency

        return reward

    def normalize_state(self, state):
        ego_dist_to_cli = state[0]
        ego_vel = state[1]
        ego_acc = state[2]

        obj_dist_to_cli = state[3]
        obj_vel = state[4]
        obj_acc = state[5]

        ego_dist_to_cli_norm = ego_dist_to_cli / 200.0
        ego_vel_norm = max(ego_vel, 30) / 30.0
        ego_acc_norm = max(ego_acc, 10) / 10.0

        obj_dist_to_cli_norm = obj_dist_to_cli / 200.0
        obj_vel_norm = max(obj_vel, 30) / 30.0
        obj_acc_norm = max(obj_acc, 10) / 10.0

        return [ego_dist_to_cli_norm, ego_vel_norm, ego_acc_norm, obj_dist_to_cli_norm, obj_vel_norm, obj_acc_norm]

    def get_steps_from_observation(self, observation):
        return observation["steps_completed"]

    def get_ego_vel_from_observation(self, observation):
        return observation["ego_vehicle_state"]["speed"]

    def get_ego_pose_from_observation(self, observation):
        return observation["ego_vehicle_state"]["position"]

    def get_events_from_observation(self, observation):
        return observation["events"]

    def get_neighborhood_vehicle_states(self, observation):
        return observation["neighborhood_vehicle_states"]

    def calc_reward(self, observation):
        K_e = 10
        K_c = 2000

        truncated = observation["events"]['collisions']

        ego_vel= observation["ego_vehicle_state"]["speed"]
        ego_target_vel = 12.0

        R_e = -K_e * abs(ego_vel - ego_target_vel) / ego_target_vel

        delta_s = observation["ego_vehicle_state"]["speed"]

        if truncated:
            R_c = -K_c
        else:
            R_c = 0.0

        return R_e + R_c

    def plot_reward(self, episode_rewards, ego_vels, gamma, epsilon, batch_size, net_kwargs={}):
        fig = plt.figure(figsize=(10, 6))  # 设置图形大小

        # 添加第一个子图
        episode_rewards_filtered = savgol_filter(episode_rewards, 19, 1, mode='nearest')
        ax1 = fig.add_subplot(111)  # 使用 add_subplot 更简洁
        ax1.plot(episode_rewards, label='Episode_Reward', color='r', linestyle='--', linewidth=1)  # 设置线条宽度和标签
        ax1.plot(episode_rewards_filtered, label='Episode_Reward-filtered', color='r', linewidth=3)
        ax1.set_ylabel('Reward', color='r')  # 设置y轴标签和色
        ax1.tick_params(axis='y', labelcolor='r')  # 设置y轴刻度标签颜色

        # 添加第二个子图，共享x轴
        ax2 = ax1.twinx()
        ax2.plot(ego_vels, label='Ego Velocity', color='b', linewidth=1)  # 设置线条样式和宽度
        ax2.set_ylabel('Ego Velocity(m/s)', color='b')  # 设置y轴标签和颜色
        ax2.tick_params(axis='y', labelcolor='b')  # 设置y轴刻度标签颜色

        # 设置x轴标签
        ax1.set_xlabel('Episode')

        # 设置标题
        fig.suptitle('Params->hidden_sizes:{}, learning_rate:{}, gamma:{}, epsilon:{}, batch_size:{}'.
                format(net_kwargs.get("hidden_sizes"), net_kwargs.get("learning_rate"), gamma, epsilon, batch_size), fontsize=16)

        # 获取两个轴的 handles 和 labels
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        # 合并 handles 和 labels
        handles = handles1 + handles2
        labels = labels1 + labels2

        # 调整子图之间的间距
        fig.tight_layout()

        # 添加图例，并设置其位置
        fig.legend(handles, labels, loc='upper right')

        plt.show()

    def save_model_params(self):
        # 兼容单卡和多卡
        eval_net = self.evaluate_net.module if hasattr(self.evaluate_net, "module") else self.evaluate_net
        target_net = self.target_net.module if hasattr(self.target_net, "module") else self.target_net
        torch.save(eval_net.state_dict(), EVA_PATH)
        torch.save(target_net.state_dict(), TARGET_PATH)

    def load_model_params(self):
        eval_net = self.evaluate_net.module if hasattr(self.evaluate_net, "module") else self.evaluate_net
        target_net = self.target_net.module if hasattr(self.target_net, "module") else self.target_net
        eval_net.load_state_dict(torch.load(EVA_PATH, map_location=self.device))
        eval_net.eval()
        target_net.load_state_dict(torch.load(TARGET_PATH, map_location=self.device))
        target_net.eval()

    def get_q_value_stats(self):
        """获取Q值的统计信息"""
        # 从回放缓冲区采样批数据
        batch = self.dataset.replay_memory.sample(self.batch_size)
        # batch是一个tuple，包含(state, action, reward, next_state, done)
        states = torch.FloatTensor(batch[0]).to(self.device)  # 获取状态数据

        # 计算Q值
        with torch.no_grad():
            q_values = self.evaluate_net(states)

        return {
            'mean': q_values.mean().item(),
            'max': q_values.max().item(),
            'min': q_values.min().item(),
            'std': q_values.std().item()
        }

    def soft_update(self, tau=0.005):
        """软更新target网络"""
        for target_param, eval_param in zip(self.target_net.parameters(),
                                          self.evaluate_net.parameters()):
            target_param.data.copy_(tau * eval_param.data +
                                  (1.0 - tau) * target_param.data)

    def learn(self, transition_dict):
        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1)
        rewards = np.expand_dims(transition_dict.reward, axis=-1)
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1)

        # 转换为tensor
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # 添加奖励裁剪
        rewards = torch.clamp(rewards, min=-2000, max=0)  # 根据实际reward范围调整

        # 使用Huber Loss替代MSE Loss，对异常值更不敏感
        self.criterion = nn.HuberLoss(delta=1.0)

        # 计算常规TD误差
        current_q = self.evaluate_net(states).gather(1, actions)
        with torch.no_grad():
            # Double DQN: 使用evaluate_net选择动作，target_net计算值
            next_q_evaluate = self.evaluate_net(next_states)
            next_actions = next_q_evaluate.max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states)
            max_next_q = next_q.gather(1, next_actions)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        td_loss = self.criterion(current_q, target_q)

        # CQL正则化项
        # 降低所有动作的Q值期望
        q_values = self.evaluate_net(states)

        # 由于reward为负值，我们希望Q值保持在一个合理的负值范围内
        logsumexp_q = torch.logsumexp(q_values, dim=1, keepdim=True)
        current_q = q_values.gather(1, actions)

        # 修改CQL损失计算方式
        cql_loss = (logsumexp_q - current_q).mean()
        if q_values.mean() > self.target_q_magnitude:
            # 如果Q值高于目标值，增加惩罚
            cql_loss = cql_loss * (1 + abs(q_values.mean() - self.target_q_magnitude))

        # 总损失
        loss = td_loss + self.cql_alpha * cql_loss

        self.optimizer.zero_grad()
        td_loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.evaluate_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        if self.count % 200 == 0:
            self.soft_update()  # 使用软更新替代硬更新
        self.count += 1

        # 记录Q值统计信息用于调整alpha
        self.current_q_stats = {
            'mean': q_values.mean().item(),
            'max': q_values.max().item(),
            'min': q_values.min().item(),
            'std': q_values.std().item()
        }
        
        return {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'total_loss': loss.item(),
            'q_values': self.current_q_stats  # 添加Q值统计信息到返回值
        }

    def adjust_cql_alpha(self, q_stats):
        """针对负值reward调整alpha的策略"""
        if isinstance(q_stats, dict) and 'mean' in q_stats:
            current_q_mean = q_stats['mean']
        else:
            current_q_mean = getattr(self, 'current_q_stats', {}).get('mean', self.target_q_magnitude)

        # 避免除零错误
        if abs(current_q_mean) < 1e-6:
            current_q_mean = -1e-6  # 改为负值

        # 对于负值Q值的特殊处理
        ratio = abs(current_q_mean / self.target_q_magnitude)
        
        # 当Q值比目标值更负时，减小alpha以允许Q值上升
        if current_q_mean < self.target_q_magnitude:
            self.cql_alpha = self.cql_alpha / (ratio + 1e-6)
        else:
            # 当Q值高于目标值时，增大alpha以压低Q值
            self.cql_alpha = self.cql_alpha * ratio

        # 限制alpha的范围
        self.cql_alpha = max(0.01, min(1.0, self.cql_alpha))  # 降低alpha的范围

        return self.cql_alpha

def play_qlearning(agent, train=False, record=False):
    if len(agent.dataset.replay_memory) < agent.batch_size:
        print("repalymemory is not enough, please collect more data")
        return

    # 计算训练的轮次
    period = len(agent.dataset.replay_memory) // agent.batch_size

    epoch_stats = {
        'td_errors': [],  # 存储每个batch的TD误差
        'cql_losses': [], # 存储每个batch的CQL损失
        'total_losses': [], # 存储每个batch的总损失
        'q_values': {     # 存储Q值统计信息
            'mean': [],
            'max': [],
            'min': [],
            'std': []
        }
    }

    for _ in range(period):
        if train:
            # start_time = time.time()
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
                agent.dataset.replay_memory.sample(agent.batch_size)

            T_data = agent.dataset.replay_memory.Transition(state_batch, action_batch,
                                                  reward_batch, next_state_batch, done_batch)

            # 对于CQLAgentlearn方法会返回更多的统计信息
            batch_stats = agent.learn(T_data)

            if record:
                # 记录统计信息
                if isinstance(batch_stats, dict):
                    epoch_stats['td_errors'].append(batch_stats.get('td_loss', 0))
                    epoch_stats['cql_losses'].append(batch_stats.get('cql_loss', 0))
                    epoch_stats['total_losses'].append(batch_stats.get('total_loss', 0))

                    # 更新Q值统计信息
                    if 'q_values' in batch_stats:
                        q_stats = batch_stats['q_values']
                        epoch_stats['q_values']['mean'].append(q_stats['mean'])
                        epoch_stats['q_values']['max'].append(q_stats['max'])
                        epoch_stats['q_values']['min'].append(q_stats['min'])
                        epoch_stats['q_values']['std'].append(q_stats['std'])

                # end_time = time.time()
                # print("period {} train time: {}".format(_, end_time - start_time))

    # 计算整个epoch的平均统计信息
    stats = {
        'td_error': np.mean(epoch_stats['td_errors']) if epoch_stats['td_errors'] else 0,
        'cql_loss': np.mean(epoch_stats['cql_losses']) if epoch_stats['cql_losses'] else 0,
        'total_loss': np.mean(epoch_stats['total_losses']) if epoch_stats['total_losses'] else 0,
        'steps': period,
        'q_values': {
            'mean': np.mean(epoch_stats['q_values']['mean']) if epoch_stats['q_values']['mean'] else 0,
            'max': np.max(epoch_stats['q_values']['max']) if epoch_stats['q_values']['max'] else 0,
            'min': np.min(epoch_stats['q_values']['min']) if epoch_stats['q_values']['min'] else 0,
            'std': np.mean(epoch_stats['q_values']['std']) if epoch_stats['q_values']['std'] else 0
        }
    }

    return stats

def decide_action_of_frame(agent, frame):
    state = agent.dataset.replay_memory.get_frame(frame).state
    action = agent.decide(state)
    return action

def replay_from_memory(agent):
    totol_action = len(agent.dataset.replay_memory)
    accuracy_action = 0
    print("memroy_size:{}".format(len(agent.dataset.replay_memory)))
    for frame in range(len(agent.dataset.replay_memory)):
        action = decide_action_of_frame(agent, frame)
        record_action = agent.dataset.replay_memory.get_frame(frame).action
        if (frame % 100 == 0):
            print("frame: {}, action: {}, agent->action: {}".\
                  format(frame, record_action, action))
        if action == record_action:
            accuracy_action += 1
    print("accuracy_action: {} / {} = {}".format(accuracy_action, totol_action, accuracy_action / totol_action))
    return accuracy_action / totol_action


class CQLAgent(ChauffeurAgent):
    def __init__(self, net_kwargs={}, gamma=0.9, epsilon=0.05,
                 batch_size=64, observation_dim=(4, 67, 133), action_size=6,
                 offline_RL_data_path=None, cql_alpha=0.01):
        super().__init__(net_kwargs, gamma, epsilon, batch_size,
                        observation_dim, action_size, offline_RL_data_path)
        # CQL特有参数
        self.cql_alpha = cql_alpha
        # 根据reward范围设置目标Q值量级
        self.target_q_magnitude = -20.0  # 设置为reward范围的10%左右
        self.min_q_value = -2000.0  # reward最小值
        self.max_q_value = 0.0      # reward最大值
        # 初始化Q值统计信息
        self.current_q_stats = {
            'mean': 0.0,
            'max': 0.0,
            'min': 0.0,
            'std': 0.0
        }

    def learn(self, transition_dict):
        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1)
        rewards = np.expand_dims(transition_dict.reward, axis=-1)
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1)

        # 转换为tensor
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # 添加奖励裁剪
        rewards = torch.clamp(rewards, min=self.min_q_value, max=self.max_q_value)

        # 使用Huber Loss替代MSE Loss，对异常值更不敏感
        self.criterion = nn.HuberLoss(delta=1.0)

        # 计算常规TD误差
        current_q = self.evaluate_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_evaluate = self.evaluate_net(next_states)
            next_actions = next_q_evaluate.max(1)[1].unsqueeze(1)
            next_q = self.target_net(next_states)
            max_next_q = next_q.gather(1, next_actions)
            # 确保目标Q值在合理范围内
            target_q = rewards + self.gamma * torch.clamp(max_next_q,
                                                        min=self.min_q_value,
                                                        max=self.max_q_value) * (1 - dones)

        td_loss = self.criterion(current_q, target_q)

        # CQL正则化项
        q_values = self.evaluate_net(states)
        logsumexp_q = torch.logsumexp(q_values, dim=1, keepdim=True)

        # 修改CQL损失计算方式
        cql_loss = (logsumexp_q - current_q).mean()

        # 根据Q值与目标范围的关系调整CQL损失
        q_mean = q_values.mean()
        if q_mean > self.target_q_magnitude:
            # Q值过高时增加惩罚
            scale = 1.0 + torch.abs(q_mean - self.target_q_magnitude) / abs(self.target_q_magnitude)
            cql_loss = cql_loss * scale

        # 总损失
        loss = td_loss + self.cql_alpha * cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.evaluate_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        if self.count % self.target_update_freq == 0:
            self.soft_update(self.tau)
        self.count += 1

        # 更新Q值统计信息
        with torch.no_grad():
            self.current_q_stats = {
                'mean': q_values.mean().item(),
                'max': q_values.max().item(),
                'min': q_values.min().item(),
                'std': q_values.std().item()
            }

        return {
            'td_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'total_loss': loss.item(),
            'q_values': self.current_q_stats
        }

    def adjust_cql_alpha(self, q_stats):
        """根据实际reward范围调整alpha"""
        if isinstance(q_stats, dict) and 'mean' in q_stats:
            current_q_mean = q_stats['mean']
        else:
            current_q_mean = getattr(self, 'current_q_stats', {}).get('mean', self.target_q_magnitude)

        # 避免除零错误
        if abs(current_q_mean) < 1e-6:
            current_q_mean = -1e-6

        # 计算Q值与目标值的比率
        ratio = abs(current_q_mean / self.target_q_magnitude)

        if current_q_mean < self.target_q_magnitude:
            # Q值过低时，减小alpha允许Q值上升
            self.cql_alpha = self.cql_alpha / (2.0 + ratio)
        else:
            # Q值过高时，增大alpha压低Q值
            self.cql_alpha = self.cql_alpha * (2.0 + ratio)

        # 根据实际reward范围调整alpha的限制范围
        alpha_min = 0.01
        alpha_max = 5.0  # 降低上限以避免过度压制
        self.cql_alpha = max(alpha_min, min(alpha_max, self.cql_alpha))

        return self.cql_alpha