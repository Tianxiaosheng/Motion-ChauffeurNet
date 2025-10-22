from waymo_data_load import WaymoScenarioDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms
import numpy as np

def plot_frameshot(dataset, scenario_id=1, frame=101, size_pixels=1000):
    example_np = dataset.to_numpy_dict(scenario_id)

    # 拼接所有时刻的状态
    all_x = np.concatenate([example_np['past_x'], example_np['current_x'], example_np['future_x']], axis=1)
    all_y = np.concatenate([example_np['past_y'], example_np['current_y'], example_np['future_y']], axis=1)
    all_valid = np.concatenate([example_np['past_valid'], example_np['current_valid'], example_np['future_valid']], axis=1)
    all_length = np.concatenate([example_np['past_length'], example_np['current_length'], example_np['future_length']], axis=1)
    all_width = np.concatenate([example_np['past_width'], example_np['current_width'], example_np['future_width']], axis=1)
    all_yaw = np.concatenate([example_np['past_bbox_yaw'], example_np['current_bbox_yaw'], example_np['future_bbox_yaw']], axis=1)
    agent_id = example_np['agent_id']
    is_sdc = example_np['is_sdc'].astype(bool)
    roadgraph_xyz = example_np['roadgraph_xyz']
    all_vx = np.concatenate([example_np['past_velocity_x'], example_np['current_velocity_x'], example_np['future_velocity_x']], axis=1)
    all_vy = np.concatenate([example_np['past_velocity_y'], example_np['current_velocity_y'], example_np['future_velocity_y']], axis=1)

    # 当前帧所有agent的状态
    x = all_x[:, frame]
    y = all_y[:, frame]
    valid = all_valid[:, frame]
    length = all_length[:, frame]
    width = all_width[:, frame]
    yaw = all_yaw[:, frame]
    vx = all_vx[:, frame]
    vy = all_vy[:, frame]

    # 只画有效agent
    mask = valid > 0
    x = x[mask]
    y = y[mask]
    length = length[mask]
    width = width[mask]
    yaw = yaw[mask]
    agent_id = agent_id[mask]
    is_sdc = is_sdc[mask]
    vx = vx[mask]
    vy = vy[mask]

    # 为每个agent分配不同颜色
    num_agents = len(x)
    cmap = plt.get_cmap('jet', num_agents)
    colors = cmap(range(num_agents))

    # 画图
    fig, ax = plt.subplots(figsize=(8, 8))
    # 路网点
    ax.plot(roadgraph_xyz[:, 0], roadgraph_xyz[:, 1], 'k.', alpha=1, ms=2)

    # 画每个agent
    for idx, (xi, yi, l, w, ya, aid, sdc_flag, vxi, vyi) in enumerate(zip(x, y, length, width, yaw, agent_id, is_sdc, vx, vy)):
        if l <= 0 or w <= 0:
            continue
        if sdc_flag:
            edgecolor = 'red'
            linewidth = 3
            facecolor = 'none'
        else:
            edgecolor = '#888888'
            linewidth = 1
            facecolor = colors[idx]
        rect = patches.Rectangle(
            (xi - l/2, yi - w/2),
            l, w,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=1.0
        )
        t = matplotlib.transforms.Affine2D().rotate_around(xi, yi, ya) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        # 标注ID
        ax.text(xi, yi + w/2 + 0.5, str(int(aid)), color='black', fontsize=8, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), clip_on=True)
        # SDC中心点特殊标记
        if sdc_flag:
            ax.plot(xi, yi, 'r*', markersize=15, markeredgecolor='yellow', markeredgewidth=2)

        # 画速度箭头
        speed = np.hypot(vxi, vyi)
        if speed > 0.01:  # 速度太小就不画
            arrow_scale = 1.0  # 可调节箭头长度缩放
            ax.arrow(
                xi, yi,
                vxi * arrow_scale, vyi * arrow_scale,
                head_width=0.7, head_length=1.2,
                fc='blue', ec='blue', alpha=0.8, length_includes_head=True
            )

    ax.set_title(f"Scenario {scenario_id+1} Frame {frame}")
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

# FILENAME = '/home/uisee/Downloads/uncompressed_scenario_training_20s_training_20s.tfrecord-00006-of-01000'
FILENAME = '/home/uisee/Documents/my_script/Motion-ChauffeurNet/dataset/data/testing_min_20s/training_20s.tfrecord-00006-of-01000'
dataset = WaymoScenarioDataset(FILENAME)
print(f"数据集大小: {len(dataset)}")
plot_frameshot(dataset, 0, 101)