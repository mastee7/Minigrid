from __future__ import annotations

import numpy as np
import pygame

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall, Lawn, Asphalt
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.lidar import SemLidar

class LavaLawnAsphaltEnv(MiniGridEnv):
    """
    ## Description
    Randomized environment with Lava, Lawn, Wall, and Asphalt.
    """
    def __init__(
        self, size=16, obstacle_type=Lava, max_steps=None, use_lidar=False, **kwargs
    ):
        self.obstacle_type = obstacle_type
        self.size = size
        self.use_lidar = use_lidar
        self.sem_lidar = None

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "prefer the lawn or asphalt, avoid the lava, and get to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Objects to place
        objects = [Lava, Wall, Lawn, Asphalt]
        
        # Helper to add a line
        def add_line(obj_class):
            # Random position and length
            top = self._rand_int([1, 1], [width / 4 * 3, height / 4 * 3])
            length = self._rand_int(width / 4, width / 2)
            
            # Random orientation
            if np.random.rand() < 0.5:
                # Horizontal
                if top[0] + length < width-1:
                    for x in range(top[0], top[0] + length):
                        self.grid.set(x, top[1], obj_class())
            else:
                # Vertical
                if top[1] + length < height-1:
                    for y in range(top[1], top[1] + length):
                        self.grid.set(top[0], y, obj_class())

        # GUARANTEE more Lawn and Asphalt
        # Add 2-3 lines of each guaranteed
        num_lawn = self._rand_int(2, 4)
        for _ in range(num_lawn):
            add_line(Lawn)
            
        num_asphalt = self._rand_int(2, 4)
        for _ in range(num_asphalt):
            add_line(Asphalt)

        # Number of additional random lines
        # Increased density slightly
        num_objects = self._rand_int(int(width / 3), int(width / 1.5))
        
        # Place random objects
        for _ in range(num_objects):
            obj_class = self._rand_elem(objects)
            add_line(obj_class)

        self.agent_dir = 0
        while True:
            pos = self._rand_int([1, 1], [width-1, height-1])
            cell = self.grid.get(*pos)
            if cell is None:
                self.agent_pos = np.array(pos)
                break

        while True:
            pos = self._rand_int([1, 1], [width-1, height-1])
            cell = self.grid.get(*pos)
            if cell is None or not (pos == self.agent_pos).all():
                self.goal_pos = np.array(pos)
                self.put_obj(Goal(), *self.goal_pos)
                break

        self.mission = "prefer the lawn or asphalt, avoid the lava, and get to the goal"

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.use_lidar:
            self._update_lidar_info(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.use_lidar:
            self._update_lidar_info(info)
        return obs, reward, terminated, truncated, info

    def _update_lidar_info(self, info):
        grid_map = self.grid.encode()[:, :, 0]
        self.sem_lidar = SemLidar(grid_map)
        self.sem_lidar.set_pos(self.agent_pos)
        self.sem_lidar.set_dir(self.agent_dir)
        pts, labels = self.sem_lidar.detect()
        info['lidar_pts'] = pts 
        info['lidar_labels'] = labels

    def render(self):
        return super().render()

    def render_with_lidar(self):
        # ... (same as LavaLawnEnv)
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.use_lidar and self.sem_lidar is not None:
            pts, labels = self.sem_lidar.detect()   
            pixels = ((pts + 0.5) * self.tile_size).astype(int)
            for p, l in zip(pixels, labels):
                img[p[1], p[0], :] = np.array([255, 255, 0])

        if self.render_mode == "human":
            self._render_human(img)
        elif self.render_mode == "rgb_array":
            return img

    def _render_human(self, img):
        img = np.transpose(img, axes=(1, 0, 2))
        if self.render_size is None:
            self.render_size = img.shape[:2]
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_size, self.screen_size)
            )
            pygame.display.set_caption("minigrid")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(img)

        # Create background with mission description
        offset = surf.get_size()[0] * 0.1
        bg = pygame.Surface(
            (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
        )
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(surf, (offset / 2, 0))

        bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

        font_size = 22
        text = self.mission
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text_rect = font.get_rect(text, size=font_size)
        text_rect.center = bg.get_rect().center
        text_rect.y = bg.get_height() - font_size * 1.5
        font.render_to(bg, text_rect, text, size=font_size)

        self.window.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()


class FixedLawnAsphaltEnv(LavaLawnAsphaltEnv):
    """
    ## Description
    Fixed map with two distinct paths: one Lawn, one Asphalt.
    Mimics contrastive/fixed_map_experiment.py
    """
    def __init__(self, size=16, **kwargs):
        # Force size to 16 for this fixed map
        super().__init__(size=16, **kwargs)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Fill everything with Walls first? 
        # In fixed_map_experiment.py: grid = np.ones((size, size)) # Walls
        # Minigrid starts empty. So we should fill with walls then clear paths.
        
        # Fill interior with walls
        for x in range(1, width-1):
            for y in range(1, height-1):
                self.grid.set(x, y, Wall())

        # ====== LEFT PATH (LAWN) ======
        # Start area (top-left) [1:3, 1:4] -> x in [1..3], y in [1..2]
        # x is col index (2nd dim in numpy), y is row index (1st dim in numpy)
        # fixed_map_experiment.py: grid[row, col]
        # So grid[1:3, 1:4] means rows 1,2 and cols 1,2,3.
        # Minigrid set(col, row) -> set(x, y)
        
        # Helper to clear area (set to None/Empty)
        def clear_rect(y_start, y_end, x_start, x_end):
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    self.grid.set(x, y, None)
                    
        # Helper to set terrain
        def set_rect(y_start, y_end, x_start, x_end, obj_class):
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    self.grid.set(x, y, obj_class())

        # Start area (top-left)
        # grid[1:3, 1:4] = 0 (Empty)
        clear_rect(1, 3, 1, 4)
        
        # Lawn corridor going down
        # grid[3:13, 2:4] = 2 (Lawn)
        set_rect(3, 13, 2, 4, Lawn)
        
        # End area (bottom-left)
        # grid[13:15, 1:4] = 0 (Empty)
        clear_rect(13, 15, 1, 4)
        
        # ====== RIGHT PATH (ASPHALT) ======
        # Start area (top-right)
        # grid[1:3, 12:15] = 0 (Empty)
        clear_rect(1, 3, 12, 15)
        
        # Asphalt corridor going down
        # grid[3:13, 12:14] = 4 (Asphalt)
        set_rect(3, 13, 12, 14, Asphalt)
        
        # End area (bottom-right)
        # grid[13:15, 12:15] = 0 (Empty)
        clear_rect(13, 15, 12, 15)
        
        # ====== CONNECTIONS ======
        # Top connection
        # grid[1:3, 4:12] = 0 (Empty)
        clear_rect(1, 3, 4, 12)
        
        # Bottom connection
        # grid[13:15, 4:12] = 0 (Empty)
        clear_rect(13, 15, 4, 12)
        
        # ====== WALL BARRIER IN MIDDLE ======
        # grid[6:10, 6:10] = 1 (Wall)
        # Already walls by default, but to be sure
        set_rect(6, 10, 6, 10, Wall)

        # Place Agent
        # In fixed map exp, random start/goal. Here we place agent randomly in valid cells.
        self.agent_dir = 0
        self._place_agent_and_goal()
        
        self.mission = "prefer the lawn or asphalt, avoid the lava, and get to the goal"

    def _place_agent_and_goal(self):
        # Define Zones
        # Top Zone (y < 3): Start area
        # Bottom Zone (y > 12): Goal area
        
        # Valid positions in Top Zone (Empty cells only)
        # We exclude Lawn/Asphalt cells from start/goal to force traversal
        start_candidates = []
        goal_candidates = []
        
        for x in range(1, self.width-1):
            for y in range(1, self.height-1):
                cell = self.grid.get(x, y)
                # Only empty cells (None)
                if cell is None:
                    if y < 3: # Top
                        start_candidates.append((x, y))
                    elif y > 12: # Bottom
                        goal_candidates.append((x, y))
                        
        if not start_candidates or not goal_candidates:
            # Fallback to any empty cell
            # Should not happen with current map gen
            pass
            
        # Sample Start
        idx = self._rand_int(0, len(start_candidates))
        self.agent_pos = np.array(start_candidates[idx])
        
        # Sample Goal
        idx = self._rand_int(0, len(goal_candidates))
        self.goal_pos = np.array(goal_candidates[idx])
        
        self.put_obj(Goal(), *self.goal_pos)
