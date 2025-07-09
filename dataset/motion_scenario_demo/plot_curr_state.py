import math
import os
import uuid

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms

import numpy as np
from waymo_data_load import WaymoScenarioDataset

def create_figure_and_axes(size_pixels):
  """Initializes a unique figure and axes for plotting."""
  fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

  # Sets output image to pixel resolution.
  dpi = 100
  size_inches = size_pixels / dpi
  fig.set_size_inches([size_inches, size_inches])
  fig.set_dpi(dpi)
  fig.set_facecolor('white')
  ax.set_facecolor('white')
  ax.xaxis.label.set_color('black')
  ax.tick_params(axis='x', colors='black')
  ax.yaxis.label.set_color('black')
  ax.tick_params(axis='y', colors='black')
  fig.set_tight_layout(True)
  ax.grid(False)
  return fig, ax


def fig_canvas_image(fig):
  """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
  # Just enough margin in the figure to display xticks and yticks.
  fig.subplots_adjust(
      left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
  fig.canvas.draw()
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents):
  """Compute a color map array of shape [num_agents, 4]."""
  colors = cm.get_cmap('jet', num_agents)
  colors = colors(range(num_agents))
  np.random.shuffle(colors)
  return colors


def get_viewport(all_states, all_states_mask):
  """Gets the region containing the data.

  Args:
    all_states: states of agents as an array of shape [num_agents, num_steps,
      2].
    all_states_mask: binary mask of shape [num_agents, num_steps] for
      `all_states`.

  Returns:
    center_y: float. y coordinate for center of data.
    center_x: float. x coordinate for center of data.
    width: float. Width of data.
  """
  valid_states = all_states[all_states_mask]
  all_y = valid_states[..., 1]
  all_x = valid_states[..., 0]

  center_y = (np.max(all_y) + np.min(all_y)) / 2
  center_x = (np.max(all_x) + np.min(all_x)) / 2

  range_y = np.ptp(all_y)
  range_x = np.ptp(all_x)

  width = max(range_y, range_x)

  return center_y, center_x, width


def visualize_one_step(states,
                       mask,
                       roadgraph,
                       title,
                       center_y,
                       center_x,
                       range_width,
                       color_map,
                       size_pixels=1000,
                       agent_info=None):
  """Generate visualization for a single step."""

  # Create figure and axes.
  fig, ax = create_figure_and_axes(size_pixels=size_pixels)

  # Plot roadgraph
  ax.plot(roadgraph[:,0], roadgraph[:,1], 'k.', alpha=1, ms=2)

  masked_x = states[:, 0][mask]
  masked_y = states[:, 1][mask]
  colors = color_map[mask]
  
  # 获取智能体信息
  if agent_info is not None:
    masked_lengths = agent_info['lengths'][mask]
    masked_widths = agent_info['widths'][mask]
    masked_yaws = agent_info['yaws'][mask]
    masked_is_sdc = agent_info['is_sdc'][mask]
  else:
    # 如果没有提供智能体信息，使用默认值
    masked_lengths = np.ones_like(masked_x) * 4.0  # 默认长度4米
    masked_widths = np.ones_like(masked_x) * 2.0   # 默认宽度2米
    masked_yaws = np.zeros_like(masked_x)          # 默认朝向0度
    masked_is_sdc = np.zeros_like(masked_x, dtype=bool)  # 默认都不是SDC

  # 绘制智能体矩形（支持旋转，跳过无效尺寸）
  import colorsys
  def brighten(color, factor=1.3):
      # color: RGBA, 0~1
      r, g, b, a = color
      h, l, s = colorsys.rgb_to_hls(r, g, b)
      l = min(1.0, l * factor)
      r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
      return (r2, g2, b2, a)

  for i, (x, y, length, width, yaw, color, is_sdc) in enumerate(
      zip(masked_x, masked_y, masked_lengths, masked_widths, masked_yaws, colors, masked_is_sdc)):
    if length <= 0 or width <= 0:
      continue
    # 提亮颜色
    # bright_color = brighten(color, 1.5 if is_sdc else 1.3)
    bright_color = brighten(color, 1.3)

    # 边框和填充色
    if is_sdc:
      edgecolor = 'red'
      linewidth = 3
      bright_color = 'none'

    else:
      edgecolor = '#888888'  # 柔和灰色
      linewidth = 1
    rect = patches.Rectangle(
        (x - length/2, y - width/2),
        length, width,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=bright_color,
        alpha=1.0
    )
    t = matplotlib.transforms.Affine2D().rotate_around(x, y, yaw) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    if is_sdc:
      ax.plot(x, y, 'r*', markersize=15, markeredgecolor='yellow', markeredgewidth=2)

  # Title.
  ax.set_title(title)
  # Set axes.  Should be at least 10m on a side and cover 160% of agents.
  size = max(10, range_width * 1.0)
  ax.axis([
      -size / 2 + center_x, size / 2 + center_x,
      -size / 2 + center_y, size / 2 + center_y
  ])
  ax.set_aspect('equal')

  image = fig_canvas_image(fig)
  plt.close(fig)
  return image

def create_animation(images):
  """ Creates a Matplotlib animation of the given images.

  Args:
    images: A list of numpy arrays representing the images.

  Returns:
    A matplotlib.animation.Animation.

  Usage:
    anim = create_animation(images)
    anim.save('/tmp/animation.avi')
    HTML(anim.to_html5_video())
  """

  plt.ioff()
  fig, ax = plt.subplots()
  dpi = 100
  size_inches = 1000 / dpi
  fig.set_size_inches([size_inches, size_inches])
  plt.ion()

  def animate_func(i):
    ax.imshow(images[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid('off')

  anim = animation.FuncAnimation(
      fig, animate_func, frames=len(images), interval=100)
  plt.close(fig)
  return anim

def visualize_all_frames_of_scenario(example_np, idx, size_pixels=1000):
    """
    绘制指定场景的所有历史、当前和未来帧。
    Args:
        example_np: one scenario of numpy dict from WaymoScenarioDataset
        size_pixels: 输出图片大小
    Returns:
        images: list of np.ndarray
    """
    num_agents = example_np['current_x'].shape[0]
    # 计算总帧数
    num_past = example_np['past_x'].shape[1]
    num_current = 1

    num_future = example_np['future_x'].shape[1]
    total_steps = num_past + num_current + num_future

    # 拼接所有时刻的状态
    all_x = np.concatenate([
        example_np['past_x'], 
        example_np['current_x'], 
        example_np['future_x']
    ], axis=1)  # [num_agents, total_steps]
    all_y = np.concatenate([
        example_np['past_y'], 
        example_np['current_y'], 
        example_np['future_y']
    ], axis=1)
    all_valid = np.concatenate([
        example_np['past_valid'], 
        example_np['current_valid'], 
        example_np['future_valid']
    ], axis=1)
    all_length = np.concatenate([
        example_np['past_length'], 
        example_np['current_length'], 
        example_np['future_length']
    ], axis=1)
    all_width = np.concatenate([
        example_np['past_width'], 
        example_np['current_width'], 
        example_np['future_width']
    ], axis=1)
    all_yaw = np.concatenate([
        example_np['past_bbox_yaw'], 
        example_np['current_bbox_yaw'], 
        example_np['future_bbox_yaw']
    ], axis=1)

    # 计算显示范围
    all_states = np.stack([all_x, all_y], axis=-1)
    center_y, center_x, width = get_viewport(all_states, all_valid)

    roadgraph_xyz = example_np['roadgraph_xyz']
    is_sdc = example_np['is_sdc'].astype(bool)
    color_map = get_colormap(num_agents)

    images = []
    for t in range(total_steps):
        # 当前帧所有agent的状态
        current_states = np.stack([
            all_x[:, t], all_y[:, t]
        ], axis=-1)  # [num_agents, 2]
        current_mask = all_valid[:, t] > 0
        agent_info = {
            'lengths': all_length[:, t],
            'widths': all_width[:, t],
            'yaws': all_yaw[:, t],
            'is_sdc': is_sdc
        }

        # 标题
        title = f"Scenario {idx+1} Step {t+1}/{total_steps}"
        im = visualize_one_step(current_states, current_mask, roadgraph_xyz, title,
                               center_y, center_x, width, color_map, size_pixels, agent_info)
        images.append(im)
    return images

# 主流程演示
if __name__ == "__main__":
    FILENAME = '/home/uisee/Downloads/uncompressed_scenario_training_20s_training_20s.tfrecord-00006-of-01000'
    dataset = WaymoScenarioDataset(FILENAME)
    print(f"数据集大小: {len(dataset)}")

    images = visualize_all_frames_of_scenario(dataset.to_numpy_dict(1), 1)
    print(f"成功生成 {len(images)} 帧图像")

    anim = create_animation(images[::10])
    #anim = create_animation(images)
    anim.save('curr_state_animation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    print("动画已保存为 curr state animation.mp4") 