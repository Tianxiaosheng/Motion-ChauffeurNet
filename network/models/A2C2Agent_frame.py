import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
from torch.utils.data import DataLoader
from dataset.frame_dataset import Frame_Dataset
import math
import time

MODEL_PATH = os.path.abspath(os.path.dirname(__file__))
NETWORK_PATH = os.path.dirname(MODEL_PATH)
PROJ_PATH = os.path.dirname(NETWORK_PATH)
DATASET_PATH = os.path.join(PROJ_PATH, 'dataset')

sys.path.append(DATASET_PATH)

ACTOR_FEATURE_PATH =  os.path.join(DATASET_PATH, "data/actor_feature_model_net.pth")
CRITIC_FEATURE_PATH = os.path.join(DATASET_PATH, "data/critic_feature_model_net.pth")
ACTOR_PATH =  os.path.join(DATASET_PATH, "data/actor_model_net.pth")
CRITIC_PATH = os.path.join(DATASET_PATH, "data/critic_model_net.pth")
CRITIC_TARGET_PATH = os.path.join(DATASET_PATH, "data/critic_target_model_net.pth")

def iter_tf_files(folder_path, pattern=".tfrecord"):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if pattern in file:
                yield os.path.join(root, file)

def play_a2c2(agent, dataset, record=False):
    if len(dataset) < agent.batch_size:
        print("repalymemory is not enough, please collect more data")
        return

    # 计算训练的轮次
    period = 0

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

    train_dataloader = DataLoader(dataset, batch_size=agent.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    for batch_idx, (states, actions, rewards, next_states, dones) in enumerate(train_dataloader):
        batch_stats = agent.learn(states, actions, rewards, next_states, dones)

        if record:
            # 记录统计信息
            period += 1
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

# 定义DQN网络结构
class FeatureNet(nn.Module):
    def __init__(self, input_dim=(6, 300, 300),
                 conv_param={'filter1_size': 6, 'filter2_size': 16,
                                'filter_width': 5, 'pad': 0, 'stride': 1}):
        super(FeatureNet, self).__init__()
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

        self.pool2_output_size = int(filter2_size * int(conv2_height / 2) *
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

    def forward(self, x):
        x = self.conv_block(x)
        if x.dim() == len(self.input_dim):
            x = x.view(-1)
        else:
            x = x.view(-1, self.num_flat_features(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_output_size(self):
        return self.pool2_output_size

class A2C2Agent:
    def __init__(self, actor_kwargs={'hidden_sizes': [120, 84], 'learning_rate': 0.0005},\
                 cirtic_kwargs={'hidden_sizes': [120, 84], 'learning_rate': 0.0005}, gamma=0.99, alpha=0.1,
                 batch_size=64, observation_dim=(4, 67, 133), action_size=3,
                 offline_RL_data_path=None, cql_alpha=0.01):
        self.observation_dim = observation_dim
        self.action_n = action_size
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.count = 0
        delta_x = 1.0
        delta_y = 1.0
        self.render = False
        # 根据reward范围设置目标Q值量级
        self.target_q_magnitude = -20.0   # 缩小目标Q值范围
        self.min_q_value = -2000.0         # 缩小最小值范围
        self.max_q_value = 0.0            # 保持不变

        # 初始化Q值统计信息
        self.current_q_stats = {
            'mean': 0.0,
            'max': 0.0,
            'min': 0.0,
            'std': 0.0
        }
        self.cql_alpha = cql_alpha
        self.tau = 0.001  # 软更新系数
        self.target_update_freq = 500  # 目标网络更新频率

        # Actor特征提取网络
        self.actor_feature_net = nn.DataParallel(
                    FeatureNet(input_dim=self.observation_dim,
                               conv_param={'filter1_size': 6, 'filter2_size': 16,
                                            'filter_width': 5, 'pad': 0, 'stride': 1})).to(self.device)

        # Critic特征提取网络
        self.critic_feature_net = nn.DataParallel(
                    FeatureNet(input_dim=self.observation_dim,
                               conv_param={'filter1_size': 6, 'filter2_size': 16,
                                            'filter_width': 5, 'pad': 0, 'stride': 1})).to(self.device)

        self.feature_net_output_size = self.actor_feature_net.module.get_output_size()

        # ACTOR-net - 去掉输出层的 Softmax
        self.actor_net = nn.DataParallel(\
                    self._build_network(input_size=self.feature_net_output_size,\
                                        hidden_sizes=actor_kwargs['hidden_sizes'],\
                                        output_size=action_size,\
                                        output_activation=None)).to(self.device)

        # CRITIC-net
        self.critic_net = nn.DataParallel(
                    self._build_network(input_size=self.feature_net_output_size,
                                        hidden_sizes=cirtic_kwargs['hidden_sizes'],
                                        output_size=action_size)).to(self.device)

        # CRITIC-target-net - 复制 critic 网络
        self.critic_target = nn.DataParallel(
                    self._build_network(input_size=self.feature_net_output_size,
                                        hidden_sizes=cirtic_kwargs['hidden_sizes'],
                                        output_size=action_size)).to(self.device)
        # 初始化目标网络参数与主网络相同
        self.critic_target.load_state_dict(self.critic_net.state_dict())

        # 创建分离的优化器
        self.actor_feature_optimizer = optim.Adam(self.actor_feature_net.parameters(), lr=0.0001)
        self.critic_feature_optimizer = optim.Adam(self.critic_feature_net.parameters(), lr=0.0001)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(),
                                        lr=actor_kwargs.get('learning_rate', 0.001))
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(),
                                          lr=cirtic_kwargs.get('learning_rate', 0.001))

        # 添加学习率调度器
        self.actor_feature_scheduler = optim.lr_scheduler.StepLR(self.actor_feature_optimizer, step_size=1000, gamma=0.95)
        self.critic_feature_scheduler = optim.lr_scheduler.StepLR(self.critic_feature_optimizer, step_size=1000, gamma=0.95)
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)

        self.dataset_list = []
        print("agent _init_!")

        # 添加目标熵和温度参数
        self.target_entropy = 1.2 * np.log(action_size)  # 增大目标熵
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-3)  # 增大学习率
        self.temperature = 3.0  # 增大温度系数

        # 调整网络学习率
        actor_lr = actor_kwargs.get('learning_rate', 0.001) * 0.1  # 降低actor学习率
        critic_lr = cirtic_kwargs.get('learning_rate', 0.001) * 0.2  # 降低critic学习率

        # 重新配置优化器
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.actor_feature_optimizer = optim.Adam(self.actor_feature_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.critic_feature_optimizer = optim.Adam(self.critic_feature_net.parameters(), lr=critic_lr)

        # 使用Huber Loss替代MSE Loss，对异常值更不敏感
        self.critic_criterion = nn.HuberLoss(delta=1.0)


    def _build_network(self, input_size, hidden_sizes, output_size,\
                       activation=nn.ReLU, output_activation=None):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        if output_activation is not None:
            if output_activation == nn.Softmax:
                layers.append(nn.Softmax(dim=1))
            else:
                layers.append(output_activation())
        return nn.Sequential(*layers)

    def decide(self, observation, deterministic=False):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        single = False
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
            single = True
        with torch.no_grad():
            feats = self.actor_feature_net(obs)
            logits = self.actor_net(feats)  # [B,A] - 输出是logits

            # 使用温度参数
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=1)

            if deterministic:
                act = torch.argmax(logits, dim=1)
            else:
                # 训练时使用categorical采样
                act = torch.distributions.Categorical(probs).sample()
        return act.item() if single else act

    def soft_update_target(self):
        """软更新目标网络"""
        for target_param, param in zip(self.critic_target.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def learn(self, states, actions, rewards, next_states, dones):
        # 转换为tensor
        states = states.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True)
        next_states = next_states.to(self.device, non_blocking=True)
        dones = dones.to(self.device, non_blocking=True)

        # 添加奖励裁剪和缩放
        rewards = torch.clamp(rewards, min=self.min_q_value, max=self.max_q_value)

        # 训练Critic - 使用目标网络
        self.critic_optimizer.zero_grad()
        self.critic_feature_optimizer.zero_grad()
        states_feature = self.critic_feature_net(states)
        current_q_value = self.critic_net(states_feature).gather(1, actions)

        with torch.no_grad():
            # Double DQN: 使用当前网络选择动作，目标网络计算值
            next_states_feature = self.critic_feature_net(next_states)
            next_states_feature_detached = next_states_feature.detach()

            # 使用当前critic选择动作
            next_q_current = self.critic_net(next_states_feature_detached)
            next_actions = next_q_current.max(1)[1].unsqueeze(1)

            # 使用目标网络计算目标Q值
            next_q_target = self.critic_target(next_states_feature_detached)
            next_q_value = next_q_target.gather(1, next_actions)

            # 计算目标值并裁剪
            target_q_value = rewards + self.gamma * (1 - dones) * next_q_value
            target_q_value = torch.clamp(target_q_value, min=self.min_q_value, max=self.max_q_value)

        # 使用Huber Loss计算TD误差
        td_loss = self.critic_criterion(current_q_value, target_q_value)

        # CQL正则化项
        q_values = self.critic_net(states_feature)
        logsumexp_q = torch.logsumexp(q_values, dim=1, keepdim=True)

        # 修改CQL损失计算方式
        cql_loss = (logsumexp_q - current_q_value).mean()

        # 根据Q值与目标范围的关系调整CQL损失
        q_mean = q_values.mean()
        if q_mean > self.target_q_magnitude:
            # Q值过高时增加惩罚
            scale = 1.0 + torch.abs(q_mean - self.target_q_magnitude) / abs(self.target_q_magnitude)
            cql_loss = cql_loss * scale

        # 总损失
        critic_loss = td_loss + self.cql_alpha * cql_loss
        critic_loss.backward()  # 不需要retain_graph

        # 分别更新Critic和Critic特征网络
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 10.0)
        self.critic_optimizer.step()

        torch.nn.utils.clip_grad_norm_(self.critic_feature_net.parameters(), 10.0)
        self.critic_feature_optimizer.step()

        # 软更新目标网络
        if self.count % self.target_update_freq == 0:
            self.soft_update_target()


        # 训练Actor - 使用独立的特征网络
        self.actor_optimizer.zero_grad()
        self.actor_feature_optimizer.zero_grad()

        # 使用Actor的特征网络
        actor_states_feature = self.actor_feature_net(states)
        action_logits = self.actor_net(actor_states_feature)

        # 使用温度参数
        scaled_logits = action_logits / self.temperature
        log_probs = torch.log_softmax(scaled_logits, dim=1)
        probs = torch.softmax(scaled_logits, dim=1)

        # 计算动作频率权重
        action_counts = torch.bincount(actions.squeeze(), minlength=self.action_n).float()
        action_weights = (1.0 / (action_counts + 1e-6))
        action_weights = action_weights / action_weights.sum()
        sample_weights = action_weights[actions.squeeze()].unsqueeze(1)

        log_prob_action = log_probs.gather(1, actions)

        # 计算advantage并加权
        with torch.no_grad():
            actor_critic_q_values = self.critic_net(self.critic_feature_net(states))
            state_values = (probs * actor_critic_q_values).sum(dim=1, keepdim=True)
            current_q_for_advantage = actor_critic_q_values.gather(1, actions)
            advantage = current_q_for_advantage - state_values

            # 标准化和裁剪advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage = torch.clamp(advantage, -1.0, 1.0)  # 收紧裁剪范围

            # 计算熵和KL散度
            entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
            uniform_probs = torch.ones_like(probs) / self.action_n
            kl_div = (probs * (log_probs - torch.log(uniform_probs + 1e-8))).sum(dim=1, keepdim=True)

        # 自适应调整熵系数
        alpha = torch.exp(self.log_alpha)
        alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 加权的actor loss，包含KL散度惩罚
        weighted_advantage = advantage * sample_weights
        actor_loss = -(weighted_advantage * log_prob_action + \
                      alpha.detach() * entropy - \
                      0.01 * kl_div).mean()  # 添加KL散度惩罚
        actor_loss.backward()

        # 更新Actor和Actor特征网络
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 1.0)
        self.actor_optimizer.step()

        torch.nn.utils.clip_grad_norm_(self.actor_feature_net.parameters(), 1.0)
        self.actor_feature_optimizer.step()

        # 更新学习率调度器（降低频率）
        if self.count % 100 == 0:
            self.actor_feature_scheduler.step()
            self.critic_feature_scheduler.step()
            self.actor_scheduler.step()
            self.critic_scheduler.step()

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
            'total_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item(),
            'entropy': entropy.mean().item(),
            'action_weights': action_weights.cpu().numpy(),
            'action_distribution': probs.mean(0).detach().cpu().numpy(),
            'q_values': self.current_q_stats
        }

    def replay_from_memory(self, testing_tf_folder_path, batch_size=64, view_time=False):
        total_action = 0
        accuracy_action = 0
        time_0 = time.time()
        print("-------start_replay_from_memory-------")
        if len(self.dataset_list) == 0:
            file_path_list = list(iter_tf_files(testing_tf_folder_path))
            print("testing_file_size{}".format(len(file_path_list)))
            for file_path in file_path_list:
                self.dataset_list.append(Frame_Dataset(observation_dim=(8, 100, 200),\
                        tf_file_path=file_path, caching=True))
            time_1 = time.time()
            print("dataset init time:{}".format(time_1-time_0))
        for dataset in self.dataset_list:
            time_2 = time.time()
            num_samples = len(dataset)
            if num_samples == 0:
                continue
            total_action += num_samples
            print("memory_size (current file): {}".format(num_samples))

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
            time_3 = time.time()
            if view_time:
                print("dataloader_init_time:{}".format(time_3-time_2))
            for batch_idx, (states, actions, rewards, next_states, dones) in enumerate(dataloader):
                # states: [batch, ...], actions: [batch, 1] or [batch]
                # agent.decide应支持batch输入
                time_4 = time.time()
                if view_time:
                    print("curr batch_idx:{}, loading time:{}".format(batch_idx, time_4 - time_3))

                with torch.no_grad():
                    # actions shape: [batch, 1] or [batch]
                    if isinstance(actions, torch.Tensor):
                        actions = actions.cpu().numpy()
                    if actions.ndim > 1:
                        actions = actions.squeeze(-1)
                    # agent.decide支持batch输入
                    states = states.to(self.device, non_blocking=True)
                    pred_actions = self.decide(states, deterministic=True)
                    # pred_actions shape: [batch] or [batch, 1]
                    if isinstance(pred_actions, torch.Tensor):
                        pred_actions = pred_actions.cpu().numpy()
                    if isinstance(pred_actions, np.ndarray) and pred_actions.ndim > 1:
                        pred_actions = pred_actions.squeeze(-1)
                    # 统计本batch准确个数
                    match = (pred_actions == actions).sum()
                    accuracy_action += match
                    if batch_idx % 10 == 0:
                        print("batch: {}, match: {}, batch_size: {}".format(batch_idx, match, len(actions)))
                        all_action = np.stack([actions, pred_actions], axis=-1)
                        # 随机截取100个数据
                        if len(all_action) > 100:
                            # 生成随机索引
                            random_indices = np.random.choice(len(all_action), 100, replace=False)
                            # 按索引截取数据
                            sampled_actions = all_action[random_indices]
                            print("compare_actions (sampled):{}".format(sampled_actions))
                        else:
                            print("compare_actions:{}".format(all_action))
                    time_3 = time.time()
                    if view_time:
                        print("pred time:{}".format(time_3-time_4))
            time_5 = time.time()
            if view_time:
                print("file pred time:{}".format(time_5-time_2))
        if total_action == 0:
            print("No data found!")
            return 0.0
        print("Total replay time:{}".format(time.time() - time_0))
        print("accuracy_action: {} / {} = {:.4f}".format(accuracy_action, total_action, accuracy_action / total_action))
        return accuracy_action / total_action

    def save_model_params(self):
        # 兼容单卡和多卡
        actor_feature_net = self.actor_feature_net.module if hasattr(self.actor_feature_net, "module") else self.actor_feature_net
        critic_feature_net = self.critic_feature_net.module if hasattr(self.critic_feature_net, "module") else self.critic_feature_net
        actor_net = self.actor_net.module if hasattr(self.actor_net, "module") else self.actor_net
        critic_net = self.critic_net.module if hasattr(self.critic_net, "module") else self.critic_net
        critic_target_net = self.critic_target.module if hasattr(self.critic_target, "module") else self.critic_target

        torch.save(actor_feature_net.state_dict(), ACTOR_FEATURE_PATH)
        torch.save(critic_feature_net.state_dict(), CRITIC_FEATURE_PATH)
        torch.save(actor_net.state_dict(), ACTOR_PATH)
        torch.save(critic_net.state_dict(), CRITIC_PATH)
        torch.save(critic_target_net.state_dict(), CRITIC_TARGET_PATH)

    def load_model_params(self):
        actor_feature_net = self.actor_feature_net.module if hasattr(self.actor_feature_net, "module") else self.actor_feature_net
        critic_feature_net = self.critic_feature_net.module if hasattr(self.critic_feature_net, "module") else self.critic_feature_net
        actor_net = self.actor_net.module if hasattr(self.actor_net, "module") else self.actor_net
        critic_net = self.critic_net.module if hasattr(self.critic_net, "module") else self.critic_net
        critic_target_net = self.critic_target.module if hasattr(self.critic_target, "module") else self.critic_target

        actor_feature_net.load_state_dict(torch.load(ACTOR_FEATURE_PATH, map_location=self.device))
        critic_feature_net.load_state_dict(torch.load(CRITIC_FEATURE_PATH, map_location=self.device))
        actor_net.load_state_dict(torch.load(ACTOR_PATH, map_location=self.device))
        critic_net.load_state_dict(torch.load(CRITIC_PATH, map_location=self.device))
        critic_target_net.load_state_dict(torch.load(CRITIC_TARGET_PATH, map_location=self.device))

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

    def get_q_value_stats(self, dataset):
        """获取Q值的统计信息"""
        # 从回放缓冲区采样批数据
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        states, actions, rewards, next_states, dones = next(iter(dataloader))

        # 计算Q值
        with torch.no_grad():
            states = states.to(self.device, non_blocking=True)
            features = self.critic_feature_net(states)
            q_values = self.critic_net(features)

        return {
            'mean': q_values.mean().item(),
            'max': q_values.max().item(),
            'min': q_values.min().item(),
            'std': q_values.std().item()
        }