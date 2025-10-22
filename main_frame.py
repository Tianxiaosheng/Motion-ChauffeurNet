import os
import sys
import time
from dataset.frame_dataset import Frame_Dataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from network.models.A2C2Agent_frame import A2C2Agent, play_a2c2
from network.models.ChauffeurAgent_frame import CQLAgent, play_fmcql

def plot_training_curves(stats, save_path='training_curves.png'):
    """绘制训练过程的各项指标"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # TD误差曲线
    ax1.plot(stats['td_errors'])
    ax1.set_title('TD Error over Training')
    ax1.set_xlabel('Epochs (x10)')
    ax1.set_ylabel('TD Error')

    # Q值变化曲线
    q_means = [q['mean'] for q in stats['q_values']]
    q_maxs = [q['max'] for q in stats['q_values']]
    ax2.plot(q_means, label='Mean Q')
    ax2.plot(q_maxs, label='Max Q')
    ax2.set_title('Q Values over Training')
    ax2.set_xlabel('Epochs (x10)')
    ax2.set_ylabel('Q Value')
    ax2.legend()

    # 动作匹配率曲线
    ax3.plot(stats['action_match_rates'])
    ax3.set_title('Action Match Rate over Training')
    ax3.set_xlabel('Epochs (x10)')
    ax3.set_ylabel('Match Rate')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def iter_tf_files(folder_path, pattern=".tfrecord"):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if pattern in file:
                yield os.path.join(root, file)

def main_A2C2(num_epochs_training, train=False, view_time=False):
    # data for training
    # training_tf_folder_path = 'dataset/data/training_min_20s'
    # training_tf_folder_path = '../data/training_20s'
    # testing_tf_folder_path = 'dataset/data/testing_min_20s'
    training_tf_folder_path = 'dataset/data/training_one_20s'
    testing_tf_folder_path = 'dataset/data/testing_one_20s'

    time_0 = time.time()
    # DQN parameters
    net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.0005}
    gamma=0.9
    epsilon=0.00
    batch_size=1000
    agent = A2C2Agent(gamma=gamma, batch_size=batch_size,\
                     observation_dim=(8, 100, 200), action_size=6,
                     offline_RL_data_path=training_tf_folder_path, cql_alpha=0.8)  # 降低CQL强度

    time_1 = time.time()
    if view_time:
        print("Agent init time:{}".format(time_1-time_0))

    print("replay_memory accuracy Before training:")
    agent.replay_from_memory(testing_tf_folder_path, 2000)
    time_2 = time.time()
    if view_time:
        print("Replay_memory time before training:{}".format(time_2-time_1))

    min_td_error = float('inf')
    if train:
        training_stats = {
            'td_errors': [],
            'q_values': [],
            'action_match_rates': []
        }
        file_path_list = list(iter_tf_files(training_tf_folder_path))
        for epoch in range(num_epochs_training):

            time_3 = time.time()
            for tf_file_idx, tf_file_path in enumerate(file_path_list):
                time_3_1 = time.time()
                dataset = Frame_Dataset(observation_dim=(8, 100, 200), tf_file_path = tf_file_path)

                epoch_stats = play_a2c2(agent, dataset, epoch % 5 == 0)
                agent.adjust_cql_alpha(epoch_stats['q_values'])
                time_4 = time.time()
                print("epoch {} file {}'s training time: {}".format(epoch, tf_file_idx, time_4 - time_3_1))

                if epoch % 5 == 0:
                    # 计算Q值统计
                    q_stats = agent.get_q_value_stats(dataset)
                    # 计算动作匹配率
                    action_match_rate = agent.replay_from_memory(testing_tf_folder_path, 2000)

                    training_stats['td_errors'].append(epoch_stats['td_error'])
                    training_stats['q_values'].append(q_stats)
                    training_stats['action_match_rates'].append(action_match_rate)

                    print(f"Epoch {epoch}, File_index {epoch}")
                    print(f"TD Error: {epoch_stats['td_error']:.4f}")
                    print(f"CQL Alpha: {agent.cql_alpha:.4f}")  # 添加CQL alpha监控
                    print(f"Actor LR: {agent.actor_optimizer.param_groups[0]['lr']:.6f}")  # 监控学习率
                    print(f"Entropy: {epoch_stats.get('entropy', 0):.4f}")  # 监控熵值
                    print(f"Alpha: {epoch_stats.get('alpha', 0):.4f}")  # 监控熵系数
                    print(f"Logits Max Diff: {epoch_stats.get('logits_max_diff', 0):.4f}")  # logits差异
                    print(f"Max Prob: {epoch_stats.get('max_prob', 0):.4f}")  # 最大概率
                    print(f"AWR Positive Ratio: {epoch_stats.get('advantage_positive_ratio', 0):.4f}")  # AWR权重比例
                    if 'action_distribution' in epoch_stats:
                        action_dist = epoch_stats['action_distribution']
                        print(f"Action Distribution: {[f'{x:.3f}' for x in action_dist]}")
                    #print(f"Action Match Rate: {action_match_rate:.4f}")
                    print(f"Q Values -> Mean: {q_stats['mean']:.4f}, Max: {q_stats['max']:.4f}")
            time_5 = time.time()
            print("epoch {} training time: {}".format(epoch, time_5 - time_3))
            # save nn model
            # if (epoch_stats['td_error'] < min_td_error):
            #     min_td_error = epoch_stats['td_error']
            #     agent.save_model_params()
        agent.save_model_params()
        # 训练结束后绘制学习曲线
        plot_training_curves(training_stats, save_path=f'training_curves_{time.strftime("%Y%m%d_%H%M%S")}.png')
    else:
        agent.load_model_params()
    time_6 = time.time()

    print("Whole training time:{}".format(time_6-time_2))
    print("replay_memory accuracy After training:")
    agent.replay_from_memory(testing_tf_folder_path, 2000)

def main_FMCQL(num_epochs_training, train=False, view_time=False):
    # data for training
    # training_tf_folder_path = 'dataset/data/training_min_20s'
    # training_tf_folder_path = '../data/training_20s'
    training_tf_folder_path = 'dataset/data/training_one_20s'
    testing_tf_folder_path = 'dataset/data/testing_one_20s'

    time_0 = time.time()
    # DQN parameters
    net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.0005}
    gamma=0.9
    epsilon=0.00
    batch_size=1000

    agent = CQLAgent(net_kwargs, gamma, epsilon, batch_size,\
                     observation_dim=(8, 100, 200), action_size=6,
                     offline_RL_data_path=training_tf_folder_path, cql_alpha=1.0)

    time_1 = time.time()
    if view_time:
        print("Agent init time:{}".format(time_1-time_0))

    print("replay_memory accuracy Before training:")
    agent.replay_from_memory(testing_tf_folder_path, 2000)
    time_2 = time.time()
    if view_time:
        print("Replay_memory time before training:{}".format(time_2-time_1))

    min_td_error = float('inf')
    if train:
        training_stats = {
            'td_errors': [],
            'q_values': [],
            'action_match_rates': []
        }
        file_path_list = list(iter_tf_files(training_tf_folder_path))
        for epoch in range(num_epochs_training):

            time_3 = time.time()
            for tf_file_idx, tf_file_path in enumerate(file_path_list):
                time_3_1 = time.time()
                dataset = Frame_Dataset(observation_dim=(8, 100, 200), tf_file_path = tf_file_path)

                epoch_stats = play_fmcql(agent, dataset, tf_file_idx % 5 == 0)
                agent.adjust_cql_alpha(epoch_stats['q_values'])
                time_4 = time.time()
                print("epoch {} file {}'s training time: {}".format(epoch, tf_file_idx, time_4 - time_3_1))

                if tf_file_idx % 5 == 0:
                    # 计算Q值统计
                    q_stats = agent.get_q_value_stats(dataset)
                    # 计算动作匹配率
                    action_match_rate = agent.replay_from_memory(testing_tf_folder_path, 2000)

                    training_stats['td_errors'].append(epoch_stats['td_error'])
                    training_stats['q_values'].append(q_stats)
                    training_stats['action_match_rates'].append(action_match_rate)

                    print(f"Epoch {epoch}, File_index {tf_file_idx}")
                    print(f"TD Error: {epoch_stats['td_error']:.4f}")
                    #print(f"Action Match Rate: {action_match_rate:.4f}")
                    print(f"Q Values -> Mean: {q_stats['mean']:.4f}, Max: {q_stats['max']:.4f}")
            time_5 = time.time()
            print("epoch {} training time: {}".format(epoch, time_5 - time_3))
            # save nn model
            # if (epoch_stats['td_error'] < min_td_error):
            #     min_td_error = epoch_stats['td_error']
            #     agent.save_model_params()
        agent.save_model_params()
        # 训练结束后绘制学习曲线
        plot_training_curves(training_stats, save_path=f'training_curves_{time.strftime("%Y%m%d_%H%M%S")}.png')
    else:
        agent.load_model_params()
    time_6 = time.time()

    print("Whole training time:{}".format(time_6-time_2))
    print("replay_memory accuracy After training:")
    agent.replay_from_memory(testing_tf_folder_path, 2000)


# algo = 'FMCQL'
# Advantage Actor-Critic with CQL
algo = 'A2C2'

if __name__ == "__main__":
    if algo == 'A2C2':
        main_A2C2(num_epochs_training=20, train=True, view_time=True)
    elif algo == 'FMCQL':
        main_FMCQL(num_epochs_training=1, train=True, view_time=True)
