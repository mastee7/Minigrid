import numpy as np

import matplotlib.pyplot as plt

class Lidar:
    """
    A simulated lidar on an NxN map
    """

    def __init__(self, grid_map, fov=360, res=0.05, depth=3):
        self.grid_map = grid_map
        self.map_min = [-0.5, -0.5]
        self.map_max = [grid_map.shape[0] - 0.5, grid_map.shape[1] - 0.5]

        self.fov = fov          # field of view in degrees
        self.res = res
        self.depth = depth
        self.pos = (-1, -1)
        self.dir = -1

        # lidar rays are 2 degrees apart
        self.num_rays = int(fov / 2) + 1
        self.angles = np.linspace(-fov / 2, fov / 2, num=self.num_rays) * np.pi / 180
        self.num_pts_per_ray = int((depth - res) / res) + 2

    def set_pos(self, pos):
        i, j = int(pos[0]), int(pos[1])
        assert isinstance(i, int) and 0 <= i <= self.grid_map.shape[0] - 1
        assert isinstance(j, int) and 0 <= j <= self.grid_map.shape[1] - 1
        self.pos = (i, j)

    def set_dir(self, direction):
        assert isinstance(direction, int) and 0 <= direction <= 3
        self.dir = direction

    def detect(self):
        """
        Detect the lidar points from the current position
        Returns the points which hit obstacles or the farthest
        """ 

        pts_r = np.array([np.linspace(self.res, self.depth, num=self.num_pts_per_ray) 
            for _ in range(self.num_rays)])

        pts_x = pts_r * np.cos(self.angles).reshape(-1, 1) + self.pos[0]
        pts_y = pts_r * np.sin(self.angles).reshape(-1, 1) + self.pos[1]

        # Make sure points are in bounds
        pts_x[pts_x < self.map_min[0]] = self.map_min[0]
        pts_x[pts_x > self.map_max[0]] = self.map_max[0]
        pts_y[pts_y < self.map_min[1]] = self.map_min[1]
        pts_y[pts_y > self.map_max[1]] = self.map_max[1]

        # Convert points from xy to cell index uv
        pts_u = np.ceil(pts_x - 0.5).astype(int)
        pts_v = np.ceil(pts_y - 0.5).astype(int)

        # Find the first lidar point that hits the obstacle
        pts_labels = self.grid_map[pts_u, pts_v]
        pts_obs_idx = np.where(pts_labels == 1)
        idx = [[self.num_pts_per_ray-1] for _ in range(self.num_rays)]
        for u, v in zip(*pts_obs_idx):
            idx[u].append(v)
        idx = [np.amin(row) for row in idx]

        pts_x = pts_x[np.arange(self.num_rays), idx]
        pts_y = pts_y[np.arange(self.num_rays), idx]

        return np.array([pts_x, pts_y])

    def render(self, pts):
        fig, ax = plt.subplots()
        ax.scatter(pts[0], pts[1])
        plt.show()



class SemLidar(Lidar):
    def __init__(self, grid_map, fov=360, res=0.05, depth=3):
        super().__init__(grid_map, fov=fov, res=res, depth=depth)

    def detect(self):
        """
        Detect the lidar points from the current position
        Returns the points which hit obstacles or the farthest 
        """ 

        pts_r = np.array([np.linspace(self.res, self.depth, num=self.num_pts_per_ray) 
            for _ in range(self.num_rays)])

        pts_x = pts_r * np.cos(self.angles).reshape(-1, 1) + self.pos[0]
        pts_y = pts_r * np.sin(self.angles).reshape(-1, 1) + self.pos[1]

        # Make sure points are in bounds
        pts_x[pts_x < self.map_min[0]] = self.map_min[0]
        pts_x[pts_x > self.map_max[0]] = self.map_max[0]
        pts_y[pts_y < self.map_min[1]] = self.map_min[1]
        pts_y[pts_y > self.map_max[1]] = self.map_max[1]

        # Convert points from xy to cell index uv
        pts_u = np.ceil(pts_x - 0.5).astype(int)
        pts_v = np.ceil(pts_y - 0.5).astype(int)

        # Find the first lidar point that hits the obstacle
        # 0 and 1 are unseen and empty in minigrid
        pts_labels = self.grid_map[pts_u, pts_v]
        pts_obs_idx = np.where(pts_labels == 2)
        idx = [[self.num_pts_per_ray-1] for _ in range(self.num_rays)]
        for u, v in zip(*pts_obs_idx):
            idx[u].append(v)
        idx = [np.amin(row) for row in idx]

        pts_x = pts_x[np.arange(self.num_rays), idx]
        pts_y = pts_y[np.arange(self.num_rays), idx]
        labels = pts_labels[np.arange(self.num_rays), idx]
        return np.stack([pts_x, pts_y], axis=-1), labels

    def render(self, pts, labels):
        color_map = np.array(['r', 'g', 'b'])
        # colors = ['red' for _ in range(len(labels))]
        fig, ax = plt.subplots()
        ax.scatter(pts[0], pts[1], color=color_map[labels])
        plt.show()



if __name__ == '__main__':

    grid_map = np.zeros((9, 9), dtype=int)

    grid_map[0, :] = 1
    grid_map[:, 0] = 1
    grid_map[-1, :] = 1
    grid_map[:, -1] = 1

    grid_map[3, 3] = 2

    lidar = SemLidar(grid_map)

    lidar.set_pos((1, 1))
    lidar.set_dir(0)
    pts, labels = lidar.detect()
    lidar.render(pts, labels)

    lidar.set_pos((2, 2))
    lidar.set_dir(1)
    pts, labels = lidar.detect()
    lidar.render(pts, labels)







