import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import re
import sys
import math

from dataset.motion_scenario_demo.waymo_data_load import WaymoScenarioDataset
import numpy as np

# --- Placeholder functions for data loading and visualization ---

def update_plot_placeholder(ax, scene_data, scene_idx, frame_idx):
    """Updates the matplotlib plot with simulated data for a specific frame."""
    ax.clear()

    np.random.seed(42 + scene_idx * 100 + frame_idx)

    scene_size = 150
    ax.set_xlim(0, scene_size)
    ax.set_ylim(0, scene_size)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f"Waymo Scene {scene_idx} - Frame {frame_idx + 1}")

    ego_x = scene_size / 2 + np.random.normal(0, 0.5)
    ego_y = scene_size / 2 + np.random.normal(0, 0.5)
    ego_yaw = np.random.uniform(-0.1, 0.1)
    ego_length, ego_width = 4.5, 2.0
    cos_yaw, sin_yaw = np.cos(ego_yaw), np.sin(ego_yaw)
    ego_corners = np.array([
        [-ego_length/2, -ego_width/2],
        [ ego_length/2, -ego_width/2],
        [ ego_length/2,  ego_width/2],
        [-ego_length/2,  ego_width/2],
        [-ego_length/2, -ego_width/2]
    ])
    R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    ego_corners_rotated = ego_corners @ R.T
    ego_corners_translated = ego_corners_rotated + np.array([ego_x, ego_y])
    ax.plot(ego_corners_translated[:, 0], ego_corners_translated[:, 1], 'b-', linewidth=2, label='Ego Vehicle')

    num_objects = np.random.randint(5, 15)
    for i in range(num_objects):
        obj_x = np.random.uniform(10, scene_size - 10)
        obj_y = np.random.uniform(10, scene_size - 10)
        obj_yaw = np.random.uniform(0, 2*np.pi)
        obj_length = np.random.uniform(3.5, 5.0)
        obj_width = np.random.uniform(1.5, 2.2)

        cos_o, sin_o = np.cos(obj_yaw), np.sin(obj_yaw)
        obj_corners = np.array([
            [-obj_length/2, -obj_width/2],
            [ obj_length/2, -obj_width/2],
            [ obj_length/2,  obj_width/2],
            [-obj_length/2,  obj_width/2],
            [-obj_length/2, -obj_width/2]
        ])
        R_o = np.array([[cos_o, -sin_o], [sin_o, cos_o]])
        obj_corners_rot = obj_corners @ R_o.T
        obj_corners_trans = obj_corners_rot + np.array([obj_x, obj_y])

        color = 'g' if i % 3 == 0 else ('m' if i % 3 == 1 else 'c')
        ax.plot(obj_corners_trans[:, 0], obj_corners_trans[:, 1], f'{color}-', linewidth=1.5)

    traj_length = 20
    t = np.linspace(0, 1, traj_length)
    traj_x = ego_x + 20 * np.sin(2 * np.pi * t * 2 + frame_idx * 0.1)
    traj_y = ego_y + 10 * np.cos(2 * np.pi * t * 3 + frame_idx * 0.15)
    ax.plot(traj_x, traj_y, 'b--', alpha=0.7, linewidth=1)

    ax.legend(loc='upper left')
    ax.figure.canvas.draw()


class AccelerationDisplay(tk.Canvas):
    def __init__(self, parent, width=60, height=300, acc_range=(-4, 2), **kwargs):
        super().__init__(parent, width=width, height=height, **kwargs)
        self.width = width
        self.height = height
        self.acc_range = acc_range
        self.current_acc = 0.0 # Initialize to 0

        # --- Grid and Label Configuration ---
        self.num_cells = 20
        self.cell_height = self.height / self.num_cells
        self.cell_width = self.width - 2 # Leave 1px border on each side

        # --- Create labels for min/max and zero ---
        # Pack labels inside the parent frame provided
        self.max_acc_label = tk.Label(parent, text=f"{self.acc_range[1]:+.1f}", font=('Arial', 8))
        self.max_acc_label.pack(side=tk.TOP, anchor='n')

        # Repack the canvas itself inside the parent, below the max label
        self.pack(side=tk.TOP, fill=tk.Y, expand=True) 

        self.zero_label = tk.Label(parent, text=" 0 ", font=('Arial', 8))
        self.zero_label.pack(side=tk.TOP, anchor='n')

        # --- Change: Pack min label inside the canvas frame, anchored to the bottom ---
        # This ensures it's positioned relative to the grid, not the entire pane
        self.min_acc_label = tk.Label(self, text=f"{self.acc_range[0]:+.1f}", font=('Arial', 8), anchor='s')
        # Place it absolutely at the bottom of the canvas
        self.min_acc_label.place(relx=0.5, rely=1.0, anchor='s', y=-2) 

        self.draw_grid()
        self.update_display(self.current_acc)

    def draw_grid(self):
        self.delete("all")
        # Draw background grid cells
        for i in range(self.num_cells):
            y0 = i * self.cell_height
            y1 = y0 + self.cell_height
            self.create_rectangle(1, y0, 1 + self.cell_width, y1, outline='#cccccc', fill='white', width=1)

    def _get_color(self, intensity, is_positive):
        """Calculates a color based on intensity (0.0 to 1.0) and sign."""
        # Clamp intensity to [0, 1]
        intensity = max(0.0, min(1.0, intensity))
        # Scale intensity to 0-255
        color_val = int(255 * (1 - intensity)) # Invert so higher intensity is darker
        # Ensure it doesn't go full black, min at 20 for example
        color_val = max(20, min(255, color_val)) 
        if is_positive:
            # Green for positive: #00XX00
            return f"#00{color_val:02x}00" 
        else:
            # Red for negative: #XX0000
            return f"#{color_val:02x}0000"

    def update_display(self, acc_value):
        self.current_acc = max(self.acc_range[0], min(self.acc_range[1], acc_value))

        # Clear previous state by redrawing the grid
        self.draw_grid()

        # Calculate and fill cells based on acceleration
        mid_index = self.num_cells // 2 
        mid_y_coord = self.height / 2

        range_span = self.acc_range[1] - self.acc_range[0]
        if range_span <= 0:
             return # Cannot display with invalid range

        # Avoid division by zero
        if self.acc_range[1] > 0 and self.current_acc >= 0:
            max_acc = self.acc_range[1]
            prop_covered = self.current_acc / max_acc if max_acc != 0 else 0 
            import math
            num_cells_to_fill = math.ceil(prop_covered * (self.num_cells / 2))

            for i in range(mid_index - num_cells_to_fill, mid_index):
                 if 0 <= i < self.num_cells:
                    y0 = i * self.cell_height
                    y1 = y0 + self.cell_height

                    # Calculate intensity based on distance from zero
                    cell_center_y = (i + 0.5) * self.cell_height
                    dist_from_mid = mid_y_coord - cell_center_y
                    norm_dist = dist_from_mid / (self.height / 2) if self.height > 0 else 0
                    intensity = norm_dist * prop_covered if prop_covered > 0 else 0 

                    color = self._get_color(intensity, is_positive=True)
                    self.create_rectangle(1, y0, 1 + self.cell_width, y1, outline='#cccccc', fill=color, width=1)

        if self.acc_range[0] < 0 and self.current_acc <= 0:
            min_acc = self.acc_range[0]
            prop_covered = abs(self.current_acc) / abs(min_acc) if min_acc != 0 else 0
            import math
            num_cells_to_fill = math.ceil(prop_covered * (self.num_cells / 2))

            for i in range(mid_index, mid_index + num_cells_to_fill):
                if 0 <= i < self.num_cells:
                    y0 = i * self.cell_height
                    y1 = y0 + self.cell_height

                    # Calculate intensity based on distance from zero
                    cell_center_y = (i + 0.5) * self.cell_height
                    dist_from_mid = cell_center_y - mid_y_coord
                    norm_dist = dist_from_mid / (self.height / 2) if self.height > 0 else 0
                    intensity = norm_dist * prop_covered if prop_covered > 0 else 0

                    color = self._get_color(intensity, is_positive=False)
                    self.create_rectangle(1, y0, 1 + self.cell_width, y1, outline='#cccccc', fill=color, width=1)

        # Draw the zero line more distinctly
        self.create_line(0, mid_y_coord, self.width, mid_y_coord, fill='black', width=1)


class WaymoVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Dataset Visualizer")
        # Set a more reasonable default size to accommodate the wider layout
        self.root.geometry("1450x750") # Increased width
        # Ensure the application closes properly on window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.dataset_folder = ""
        self.file_path = ""
        self.files_list = []
        self.current_scene_data = {}
        self.current_scene_idx = None
        self.total_frames = 0
        self.current_frame_idx = 0

        self.simulated_acc = 0.0 # Start at 0
        self.acc_range = (-4.0, 2.0)

        # --- Add this line to store current view limits ---
        self.keep_view_limits = False # Flag to control view limit behavior
        self.current_xlim = None
        self.current_ylim = None

        self.setup_ui()

    def setup_ui(self):
        # --- Title Bar ---
        self.title_frame = tk.Frame(self.root, bg='darkblue', height=50)
        self.title_frame.pack(side=tk.TOP, fill=tk.X)
        self.title_label = tk.Label(self.title_frame, text="Motion Dataset Visualizer", fg='white', bg='darkblue', font=('Arial', 16, 'bold'))
        self.title_label.pack(pady=10)

        # --- Main Content Area ---
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Left: Plot Area ---
        self.plot_pane = tk.Frame(self.main_frame)
        self.plot_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.plot_frame = tk.LabelFrame(self.plot_pane, text="Display Window", padx=5, pady=5)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # --- Center: Acceleration Display ---
        # This is the dedicated column for the acceleration display
        self.acc_pane = tk.Frame(self.main_frame, width=90) # Increased width for labels
        self.acc_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))
        self.acc_pane.pack_propagate(False) # Prevent frame from shrinking

        # Use a LabelFrame with a shorter, multi-line title to fit better
        self.acc_display_frame = tk.LabelFrame(self.acc_pane, text="Acc.\n(m/s²)", padx=5, pady=5)
        self.acc_display_frame.pack(fill=tk.BOTH, expand=True)

        # Create the AccelerationDisplay canvas inside the frame
        self.acc_display = AccelerationDisplay(self.acc_display_frame, acc_range=self.acc_range)
        
        # Label to show the current numerical value, placed below the AccelerationDisplay
        self.acc_value_label = tk.Label(self.acc_display_frame, text=f"Acc: {self.simulated_acc:.2f}")
        self.acc_value_label.pack(side=tk.BOTTOM, pady=(5,0))
        # Link the label to the display canvas for easy updates
        self.acc_display.acc_label = self.acc_value_label

        # --- Right: Control Panel ---
        self.control_frame = tk.LabelFrame(self.main_frame, text="Control Panel", padx=10, pady=10, width=350)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        self.control_frame.pack_propagate(False) # Keep control panel width fixed

        # --- Plot Area Content ---
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            import matplotlib.pyplot as plt

            self.fig, self.ax = plt.subplots()
            self.ax.set_title("Please load a dataset.")
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
            self.toolbar.update()
            self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

            self.plot_initialized = True
        except ImportError as e:
            messagebox.showerror("Import Error", f"Matplotlib is required: {e}")
            self.plot_initialized = False
            self.ax = None
            placeholder_label = tk.Label(self.plot_frame, text="Plot Area\n(Requires matplotlib)")
            placeholder_label.pack(expand=True)

        # --- Control Panel Sections ---
        # 1. Load Dataset
        self.file_load_frame = tk.LabelFrame(self.control_frame, text="Load Dataset", padx=5, pady=5)
        self.file_load_frame.pack(fill=tk.X, pady=(0, 10))
        self.open_button = tk.Button(self.file_load_frame, text="1. Open Dataset Folder", command=self.open_folder)
        self.open_button.pack(pady=5, fill=tk.X)

        # 2. File Selection
        self.file_select_frame = tk.LabelFrame(self.control_frame, text="File Selection", padx=5, pady=5)
        self.file_select_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(self.file_select_frame, text="2. Select File:").pack(anchor='w')
        self.file_combobox = ttk.Combobox(self.file_select_frame, state='disabled', width=40)
        self.file_combobox.bind("<<ComboboxSelected>>", self.on_file_selected)
        self.file_combobox.pack(pady=5, fill=tk.X)

        # 3. Scene Selection
        self.scene_select_frame = tk.LabelFrame(self.control_frame, text="Scene Selection", padx=5, pady=5)
        self.scene_select_frame.pack(fill=tk.X, pady=(0, 10))
        self.scene_count_label = tk.Label(self.scene_select_frame, text="Scenes in file: N/A")
        self.scene_count_label.pack(anchor='w')
        tk.Label(self.scene_select_frame, text="3. Select Scene:").pack(anchor='w')
        self.scene_combobox = ttk.Combobox(self.scene_select_frame, state='disabled', width=40)
        self.scene_combobox.bind("<<ComboboxSelected>>", self.on_scene_selected)
        self.scene_combobox.pack(pady=5, fill=tk.X)

        # 4. Frame Navigation
        self.frame_nav_frame = tk.LabelFrame(self.control_frame, text="Frame Navigation", padx=5, pady=5)
        self.frame_nav_frame.pack(fill=tk.X, pady=(0, 10))
        self.frame_info_label = tk.Label(self.frame_nav_frame, text="Total Frames: N/A | Current Frame: N/A")
        self.frame_info_label.pack(anchor='w')

        self.nav_buttons_frame = tk.Frame(self.frame_nav_frame)
        self.nav_buttons_frame.pack(pady=10, fill=tk.X)
        self.prev_button = tk.Button(self.nav_buttons_frame, text="Previous Frame", command=self.prev_frame, state='disabled')
        self.prev_button.pack(side=tk.LEFT, expand=True, padx=(0, 5))
        self.next_button = tk.Button(self.nav_buttons_frame, text="Next Frame", command=self.next_frame, state='disabled')
        self.next_button.pack(side=tk.RIGHT, expand=True, padx=(5, 0))

        # Initialize demo plot if matplotlib is available
        if self.plot_initialized:
            self.initialize_demo_plot()

    def initialize_demo_plot(self):
        if self.ax:
            self.ax.clear()
            self.ax.set_title("Demo Plot - Load a Dataset to Begin")
            self.ax.text(0.5, 0.5, "Welcome!\n1. Click 'Open Dataset Folder'\n2. Select a .tfrecord file\n3. Choose a scene\n4. Navigate frames", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.axis('off')
            self.canvas.draw()

    def open_folder(self):
        folder_path = filedialog.askdirectory(title="Select Waymo Motion Dataset Folder")
        if folder_path:
            self.dataset_folder = folder_path
            success = self.populate_file_list()
            if success and self.files_list:
                self.file_combobox['state'] = 'readonly' # Enable combobox
                self.file_combobox['values'] = self.files_list
                self.file_combobox.current(0)
                self.on_file_selected(None) # Trigger file load

    def populate_file_list(self):
        if not self.dataset_folder:
            return False
        try:
            # Updated pattern to match Waymo's .tfrecord file naming convention
            # e.g., "some_name.tfrecord-00006-of-01000"
            pattern = re.compile(r".*\.tfrecord-\d{5}-of-\d{5}$")
            # List all files and filter by the pattern
            all_files = os.listdir(self.dataset_folder)
            self.files_list = sorted([f for f in all_files if pattern.match(f)])

            if self.files_list:
                return True
            else:
                messagebox.showinfo("No Files Found", "No files matching the pattern '*.tfrecord-XXXXX-of-XXXXX' were found in the selected folder.")
                self.file_combobox['values'] = []
                self.file_combobox.set("")
                self.disable_scene_controls()
                return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read folder: {e}")
            return False

    def on_file_selected(self, event):
        selected_file = self.file_combobox.get()
        if not selected_file:
            return

        self.file_path = os.path.join(self.dataset_folder, selected_file)
        try:
            # Load file metadata (scene list)
            self.current_scena_data = self.load_file_placeholder(self.file_path)
            scene_ids = sorted(list(self.current_scena_data.keys()))
            if scene_ids:
                self.scene_combobox['values'] = scene_ids
                self.scene_combobox['state'] = 'readonly'
                self.scene_count_label['text'] = f"Scenes in file: {len(scene_ids)}"
                self.scene_combobox.current(0)
                self.on_scene_selected(None) # Trigger scene load
            else:
                self.scene_combobox['values'] = []
                self.scene_combobox.set("No scenes found")
                self.disable_frame_controls()
                self.scene_count_label['text'] = "Scenes in file: 0"
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            self.disable_scene_controls()

    def on_scene_selected(self, event):
        selected_scene_str = self.scene_combobox.get()
        if not selected_scene_str.isdigit():
             return
        self.current_scene_idx = int(selected_scene_str)

        if self.current_scene_idx in self.current_scena_data:
            try:
                # Load scene data (frame count etc.)
                # self.current_scene_data = self.current_scena_data[self.current_scene_idx]
                self.total_frames = self.load_scene_placeholder(self.current_scene_idx)
                self.current_frame_idx = 0 # Reset to first frame

                # Reset simulated acceleration
                self.simulated_acc = 0.0 # Reset to 0 when scene changes
                self.acc_display.update_display(self.simulated_acc)

                # --- Add these lines to reset view on new scene ---
                # This will cause display_frame to NOT restore previous limits
                # and let matplotlib auto-scale or use default limits for the new scene
                self.keep_view_limits = False
                self.current_xlim = None
                self.current_ylim = None

                # Enable frame navigation controls
                self.enable_frame_controls()
                self.update_frame_label()
                self.display_frame() # Show the first frame
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load scene: {e}")
                self.disable_frame_controls()
        else:
            self.disable_frame_controls()

    def load_file_placeholder(self, file_path):
        """Simulates loading a .tfrecord file and returning scene metadata."""
        print(f"[LOG] Loading file: {file_path}")

        self.dataset = WaymoScenarioDataset(file_path)

        scene_count = len(self.dataset)
        scenes_data = {i: {"frame_count": 150 - i * 20} for i in range(scene_count)}
        return scenes_data

    def load_scene_placeholder(self, scene_idx):
        """Simulates loading data for a specific scene."""
        print(f"[LOG] Loading scene {scene_idx}")
        frame_count = self.dataset.get_scenario_frame_size(self.dataset.get_scenario(scene_idx))

        return frame_count

    def display_frame(self):
        if self.plot_initialized and self.ax:
            # Call the placeholder function to update the matplotlib plot
            # Get the current limits from the axes
            if self.keep_view_limits:
                self.current_xlim = self.ax.get_xlim()
                self.current_ylim = self.ax.get_ylim()

            self.update_plot_placeholder(self.ax, self.current_scene_idx, self.current_frame_idx)

            if self.keep_view_limits and self.current_xlim is not None and self.current_ylim is not None:
                 self.ax.set_xlim(self.current_xlim)
                 self.ax.set_ylim(self.current_ylim)
            else:
                # Auto-scale view to fit the newly plotted data
                # This happens on first load or after switching scenes
                self.ax.autoscale_view() # Let matplotlib adjust
                # Update stored limits to the new auto-scaled ones
                self.current_xlim = self.ax.get_xlim()
                self.current_ylim = self.ax.get_ylim()
                # Ensure the keep flag is set for subsequent frame changes
                self.keep_view_limits = True
            self.canvas.draw() # Refresh the canvas
        else:
            print(f"[LOG] Displaying Frame {self.current_frame_idx}")

    def update_frame_label(self):
        self.frame_info_label['text'] = f"Total Frames: {self.total_frames} | Current Frame: {self.current_frame_idx + 1}/{self.total_frames}"
        # Update the numerical value label
        if hasattr(self.acc_display, 'acc_label') and self.acc_display.acc_label:
             self.acc_display.acc_label.config(text=f"Acc: {self.simulated_acc:.2f}")

    def prev_frame(self):
        self.keep_view_limits = True # <<<--- Add this line at the beginning
        # Update simulated acceleration: decrease by 0.5
        # self.simulated_acc = max(self.acc_range[0], self.simulated_acc - 0.5)
        self.simulated_acc = self.get_action_acc_from_scenario(\
                self.dataset.get_scenario(self.current_scene_idx),\
                self.current_frame_idx)

        self.acc_display.update_display(self.simulated_acc)
        self.update_frame_label() # Update label to reflect new acc value

        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_frame_label()
            self.display_frame()

    def next_frame(self):
        self.keep_view_limits = True # <<<--- Add this line at the beginning
        # Update simulated acceleration: increase by 0.5
        # self.simulated_acc = min(self.acc_range[1], self.simulated_acc + 0.5)
        self.simulated_acc = self.get_action_acc_from_scenario(\
                self.dataset.get_scenario(self.current_scene_idx),\
                self.current_frame_idx)
        self.acc_display.update_display(self.simulated_acc)
        self.update_frame_label() # Update label to reflect new acc value

        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.update_frame_label()
            self.display_frame()

    def disable_scene_controls(self):
        self.scene_combobox['state'] = 'disabled'
        self.scene_combobox.set("")
        self.scene_count_label['text'] = "Scenes in file: N/A"
        self.disable_frame_controls()

    def disable_frame_controls(self):
        self.frame_info_label['text'] = "Total Frames: N/A | Current Frame: N/A"
        self.prev_button['state'] = 'disabled'
        self.next_button['state'] = 'disabled'
        self.current_scene_idx = None
        self.total_frames = 0
        self.current_frame_idx = 0

    def enable_frame_controls(self):
        self.prev_button['state'] = 'normal'
        self.next_button['state'] = 'normal'

    def on_closing(self):
        """Handles the window closing event to ensure proper termination."""
        print("Application closing...")
        # It's good practice to clean up resources, though often not strictly necessary for simple Tkinter apps.
        # For matplotlib, closing the figure can help.
        if hasattr(self, 'fig'):
            import matplotlib.pyplot as plt
            plt.close(self.fig)

        # Destroy the root window
        self.root.destroy()
        # Explicitly exit the application
        # This is crucial for ensuring the Python process terminates,
        # especially if there are background threads or resources.
        sys.exit(0)

    def update_plot_placeholder(self, ax, scene_idx, frame):
        """Updates the matplotlib plot with simulated data for a specific frame."""
        ax.clear()

        import numpy as np

        example_np = self.dataset.to_numpy_dict(scene_idx)
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
        import matplotlib.pyplot as plt
        num_agents = len(x)
        cmap = plt.get_cmap('jet', num_agents)
        colors = cmap(range(num_agents))


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
            import matplotlib.patches as patches
            rect = patches.Rectangle(
                (xi - l/2, yi - w/2),
                l, w,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=1.0
            )
            import matplotlib.transforms
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


        # ax.set_xlim(0, scene_size)
        # ax.set_ylim(0, scene_size)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f"Waymo Scene {scene_idx} - Frame {frame + 1}")
        ax.legend(loc='upper left')
        #ax.figure.canvas.draw()

    # action: acc
    def get_action_acc_from_scenario(self, scenario, frame):
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
            speed = math.hypot(state.velocity_x, state.velocity_y);
            speeds.append(speed)
            print("time:{}, speed:{}".format(timestamps_seconds[frame+i], speed))
            #print("[{}, {}]".format(timestamps_seconds[frame+i], state.velocity_x))
        if len(times) >= 2:
            calc_acc, intercept = np.polyfit(np.array(times), np.array(speeds), 1)

        # 直接计算最接近的动作索引
        # closest_index = min(range(len(actions)), key=lambda i: abs(actions[i] - calc_acc))
        # return actions[closest_index]
        return calc_acc

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = WaymoVisualizerApp(root)
        root.mainloop()
        # If mainloop exits normally (e.g., via destroy), also exit
        print("Mainloop exited.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        messagebox.showerror("Fatal Error", f"Application error: {e}")
        sys.exit(1)





