import sys
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from network.models.ChauffeurAgent import CQLAgent, play_qlearning, replay_from_memory

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

def main(num_epochs_training, train=False):
    # data for training
    file_path = '/home/uisee/Downloads/uncompressed_scenario_training_20s_training_20s.tfrecord-00006-of-01000'
    # data for testing
    #file_path = '/home/uisee/Downloads/uncompressed_scenario_training_20s_training_20s.tfrecord-00012-of-01000'
    # DQN parameters
    net_kwargs = {'hidden_sizes' : [64, 64], 'learning_rate' : 0.0005}
    gamma=0.9
    epsilon=0.00
    batch_size=400
    agent = CQLAgent(net_kwargs, gamma, epsilon, batch_size,\
                     observation_dim=(8, 100, 200), action_size=6,
                     offline_RL_data_path=file_path, cql_alpha=1.0)

    agent.load_replay_memory()

    print("replay_memory accuracy Before training:")
    replay_from_memory(agent)
    min_td_error = float('inf')
    if train:
        training_stats = {
            'td_errors': [],
            'q_values': [],
            'action_match_rates': []
        }
        for epoch in range(num_epochs_training):
            start_time = time.time()
            epoch_stats = play_qlearning(agent, train, epoch % 5 == 0)
            end_time = time.time()
            agent.adjust_cql_alpha(epoch_stats['q_values'])
            print("epoch {} train time: {}, cql_alpha: {}".format(epoch, end_time - start_time, agent.cql_alpha))

            if epoch % 5 == 0:
                # 计算Q值统计
                q_stats = agent.get_q_value_stats()
                # 计算动作匹配率
                action_match_rate = replay_from_memory(agent)

                training_stats['td_errors'].append(epoch_stats['td_error'])
                training_stats['q_values'].append(q_stats)
                training_stats['action_match_rates'].append(action_match_rate)

                print(f"Epoch {epoch}")
                print(f"TD Error: {epoch_stats['td_error']:.4f}")
                print(f"Action Match Rate: {action_match_rate:.4f}")
                print(f"Q Values -> Mean: {q_stats['mean']:.4f}, Max: {q_stats['max']:.4f}")
            # save nn model
            # if (epoch_stats['td_error'] < min_td_error):
            #     min_td_error = epoch_stats['td_error']
            #     agent.save_model_params()
        agent.save_model_params()
        # 训练结束后绘制学习曲线
        #plot_training_curves(training_stats, save_path=f'training_curves_{time.strftime("%Y%m%d_%H%M%S")}.png')
    else:
        agent.load_model_params()

    print("replay_memory accuracy After training:")
    replay_from_memory(agent)

if __name__ == "__main__":
    main(num_epochs_training=11, train=True)
