import math
import numpy as np

def is_equal(x, y):
    return abs(x - y) < 1e-6

def is_greater(x, y):
    return x > y and not is_equal(x, y)
def is_less(x, y):
    return x < y and not is_equal(x, y)

def cvt_pose_local_to_global(local_pos_x, local_pos_y, local_pos_theta,
                             base_pos_x, base_pos_y, base_pos_theta):
    global_pos_x = base_pos_x
    global_pos_x += math.sin(base_pos_theta) * local_pos_x + math.cos(base_pos_theta) * local_pos_y
    global_pos_y = base_pos_y
    global_pos_y += math.sin(base_pos_theta) * local_pos_y - math.cos(base_pos_theta) * local_pos_x;

    global_pos_theta = local_pos_theta + base_pos_theta - math.pi / 2.0
    global_pos_theta = global_pos_theta % (math.pi * 2.0)

    return (global_pos_x, global_pos_y, global_pos_theta)

def cvt_pose_global_to_local(global_pos_x, global_pos_y, global_pos_theta,\
                             base_pos_x, base_pos_y, base_pos_theta):
    dx = global_pos_x - base_pos_x
    dy = global_pos_y - base_pos_y
    theta = base_pos_theta

    local_pos_x = math.sin(theta) * dx - math.cos(theta) * dy
    local_pos_y = math.cos(theta) * dx + math.sin(theta) * dy

    local_pos_theta = math.pi / 2.0 + global_pos_theta - theta
    if local_pos_theta < 0.0:
        local_pos_theta += math.pi * 2.0
    local_pos_theta = local_pos_theta % (2.0 * math.pi)
    return (local_pos_x, local_pos_y, local_pos_theta)

def is_point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon.

    :param point: Tuple (x, y) representing the point
    :param polygon: List of tuples representing the polygon's corners
    :return: True if the point is inside the polygon, False otherwise
    """
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if the point is on a boundary
        if ((is_greater(yi, y) != is_greater(yj, y)) and is_less(x, (xj - xi) * (y - yi) / (yj - yi) + xi)):
            inside = not inside  

        j = i

    return inside

class OccupancyGrid:
    def __init__(self, shape_dim, delta_x, delta_y, render):
        self.shape_dim = shape_dim
        self.channels = shape_dim[0]
        self.height = shape_dim[1]  # (channels, hight, width)
        self.width = shape_dim[2]
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.render = render
        #  grid's global pose of coordinate origin
        self.center_x = 0.0
        self.center_y = 0.0
        self.heading = 0.0

        # grid's local max x and local max y
        self.max_x = self.width * self.delta_x
        self.max_y = self.height * self.delta_y

    def cvt_x_to_index_x(self, x):
        return int(x // self.delta_x)
    def cvt_y_to_index_y(self, y):
        return int(y // self.delta_y)

    def preprocess_occupancy_grid(self, ego_center, ego_heading):
        #  grid's global pose of coordinate origin
        dist = (self.width - 1) / 2.0 * self.delta_x
        self.center_x = ego_center[0] - dist * math.sin(ego_heading)
        self.center_y = ego_center[1] + dist * math.cos(ego_heading)
        self.heading = ego_heading % (2.0 * math.pi)

        # self.grid = np.zeros(self.shape_dim, dtype=float)
        self.grid = np.full(self.shape_dim, -1.0, dtype=np.float32)

        if self.render:
            print("ego_x:{}, y:{}, th:{}, grid_global_x:{}, y:{}, th:{}".\
                  format(ego_center[0], ego_center[1], ego_heading, \
                         self.center_x, self.center_y, self.heading))

    def update_occupancy_grid_by_route_and_speed_limit(self, map_point_xyv):
        for point_xyv in map_point_xyv:
            local_x , local_y, local_heading = \
                    cvt_pose_global_to_local(point_xyv[0], point_xyv[1], 0.0,\
                                             self.center_x, self.center_y, self.heading)
            #print("pt->x:{}, y{}".format(point_xyv[0], point_xyv[1]))
            if local_x < 0 or local_x > self.max_x or local_y < 0 or local_y > self.max_y:
                continue
            #print("in range local_pt_x:{}, y{}".format(local_x, local_y))
            y = self.cvt_y_to_index_y(local_y)
            x = self.cvt_x_to_index_x(local_x)

            self.grid[0][y][x] = 1
            self.grid[2][y][x] = point_xyv[2]

    def update_occupancy_grid_by_traffic_light(self, map_point_xyt):
        for point_xyt in map_point_xyt:
            local_x , local_y, local_heading = \
                    cvt_pose_global_to_local(point_xyt[0], point_xyt[1], 0.0,\
                                             self.center_x, self.center_y, self.heading)
            #print("stop_pt->x:{}, y:{}, local_x:{}, y:{}".format(point_xyt[0], point_xyt[1], local_x, local_y))
            if local_x < 0 or local_x > self.max_x or local_y < 0 or local_y > self.max_y:
                continue
            y = self.cvt_y_to_index_y(local_y)
            x = self.cvt_x_to_index_x(local_x)

            self.grid[1][y][x] = point_xyt[2]

    def update_occupancy_grid_by_ego_path(self, ego_xy):
        for point_xyt in ego_xy:
            local_x , local_y, local_heading = \
                    cvt_pose_global_to_local(point_xyt[0], point_xyt[1], 0.0,\
                                             self.center_x, self.center_y, self.heading)
            if local_x < 0 or local_x > self.max_x or local_y < 0 or local_y > self.max_y:
                continue
            y = self.cvt_y_to_index_y(local_y)
            x = self.cvt_x_to_index_x(local_x)

            self.grid[3][y][x] = 1


    def update_occupancy_grid_by_obj(self, obj_center, obj_heading, obj_vel, bounding_box, is_ego):
        # Convert global pose of obj to local pose of grid
        obj_center_x, obj_center_y, heading_local = \
                cvt_pose_global_to_local(obj_center[0], obj_center[1], obj_heading,\
                                         self.center_x, self.center_y, self.heading)
        if self.render:
            print("obj_global->x{}, y{}, th{}, length{}, width{}, local->x{}, y{}, th{}".\
                  format(obj_center[0], obj_center[1], obj_heading, bounding_box[0], bounding_box[1],\
                         obj_center_x, obj_center_y, heading_local))

        feature = [obj_vel, heading_local]

        # Calculate the rotation matrix
        cos_theta = math.cos(heading_local)
        sin_theta = math.sin(heading_local)
        rotation_matrix = [[cos_theta, -sin_theta], [sin_theta, cos_theta]]

        # Calculate the corners of the rectangle in local coordinates (centered at (0, 0))  
        half_height = bounding_box[0] / 2.0
        half_width = bounding_box[1] / 2.0
        corners = [
            (-half_height, -half_width),
            (-half_height, half_width),
            (half_height, half_width),
            (half_height, -half_width)
        ]

        # Rotate and translate the corners to global coordinates
        rotated_corners = []
        for corner in corners:
            # Rotate
            rotated_x, rotated_y = (rotation_matrix[0][0]*corner[0] + rotation_matrix[0][1]*corner[1],
                                    rotation_matrix[1][0]*corner[0] + rotation_matrix[1][1]*corner[1])
            # Translate
            rotated_corners.append((rotated_x + obj_center_x, rotated_y + obj_center_y))

        # Get the minimum and maximum x and y coordinates to define the bounding box
        x_min, y_min = min(c[0] for c in rotated_corners), min(c[1] for c in rotated_corners)
        x_max, y_max = max(c[0] for c in rotated_corners), max(c[1] for c in rotated_corners)

        # Update the occupancy grid
        for x in range(math.floor(x_min / self.delta_x), math.floor(x_max / self.delta_x) + 1):
            for y in range(math.floor(y_min / self.delta_y), math.floor(y_max / self.delta_y) + 1):
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    continue
                # Check if the current cell is within the rotated rectangle
                if is_point_in_polygon((x * self.delta_x, y * self.delta_y), rotated_corners):
                    # feature [occupancy_status, obj_heading, obj_speeding, dist_to_cli]
                    if is_ego:
                        self.grid[4][y][x] = feature[0]
                        self.grid[5][y][x] = feature[1]
                    else:
                        self.grid[6][y][x] = feature[0]
                        self.grid[7][y][x] = feature[1]

    def dump_ogm_graph(self, grid, channel):
        if channel == 0:
            ch = 'route'
        elif channel == 1:
            ch = 'traffic_light'
        elif channel == 2:
            ch = 'speed_limit'
        elif channel == 3:
            ch = 'ego_path'
        elif channel == 4:
            ch = 'ego_speed'
        elif channel == 5:
            ch = 'ego_heading'
        elif channel == 6:
            ch = 'obj_speed'
        elif channel == 7:
            ch = 'obj_heading'
        else:
            ch = 'None'

        print('------------------------{}---------------------------'.format(ch))
        for row in reversed(grid[channel]):
            print(' '.join(str(int(cell)) for cell in row))

    def dump_ogm_graphs(self, grid):
        for channel in range(self.channels):
            self.dump_ogm_graph(grid, channel)

    def save_ogm_graphs(self, grid):
        file_path = "/home/uisee/Documents/my_script/offlineFMDQN/data/state_py.txt"
        with open(file_path, 'w') as file:
            # 遍历每个通道
            for channel in range(self.channels):
                # 遍历网格的每一行
                for row in reversed(grid[channel]):
                    # 将每个单元格的值转换为字符串并用空格分隔，然后写入文件
                    file.write(' '.join(str(cell) for cell in row) + '\n')
