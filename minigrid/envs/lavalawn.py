from __future__ import annotations

import numpy as np
import pygame

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall, Lawn
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.lidar import SemLidar

class LavaLawnEnv(MiniGridEnv):
    """
    
    """
    """
    ## Description
    Environment with one wall one lawn and one lava

    The agent has to reach the green goal square at the opposite corner of the
    room, and is encouraged to travel on lawn.
    Touching the lava terminate the episode with a zero reward. This environment
    is useful for studying safety and safe exploration.

    ## Mission Space

    "prefer the lawn, avoid the lava, and get to the goal"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of map SxS.

    - `MiniGrid-LavaLawnS16-v0`
    - `MiniGrid-LavaLawnS64-v0`

    """
    def __init__(
        self, size, obstacle_type=Lava, max_steps=None, use_lidar=False, **kwargs
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
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "prefer the lawn, avoid the lava, and get to the goal"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        objects = [Lava, Wall, Lawn]
        num_objects = self._rand_int(width / 4, width / 2)
        for _ in range(num_objects):
            obj = self._rand_elem(objects)
            top = self._rand_int([1, 1], [width / 4 * 3, height / 4 * 3])
            length = self._rand_int(width / 4, width / 2)
            if np.random.rand() < 0.5:
                if top[0] + length < width-1:
                    self.grid.horz_wall(top[0], top[1], length, obj)
            else:
                if top[1] + length < height-1:
                    self.grid.vert_wall(top[0], top[1], length, obj)

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

        self.mission = (
            "prefer the lawn, avoid the lava, and get to the goal"
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.use_lidar:
            grid_map = self.grid.encode()[:, :, 0]
            self.sem_lidar = SemLidar(grid_map)
            self.sem_lidar.set_pos(self.agent_pos)
            self.sem_lidar.set_dir(self.agent_dir)
            pts, labels = self.sem_lidar.detect()
            info['lidar_pts'] = pts 
            info['lidar_labels'] = labels
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.use_lidar:
            self.sem_lidar.set_pos(self.agent_pos)
            self.sem_lidar.set_dir(self.agent_dir)
            pts, labels = self.sem_lidar.detect()
            info['lidar_pts'] = pts 
            info['lidar_labels'] = labels
        return obs, reward, terminated, truncated, info

    def render_with_lidar(self):

        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.use_lidar and self.sem_lidar is not None:
            pts, labels = self.sem_lidar.detect()   
            pixels = ((pts + 0.5) * self.tile_size).astype(int)
            for p, l in zip(pixels, labels):
                img[p[1], p[0], :] = np.array([255, 255, 0])

        if self.render_mode == "human":
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
            # offset = 32 if self.agent_pov else 64
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

        elif self.render_mode == "rgb_array":
            return img

    def render_with_path(self, states):

        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        for state in states:
            state = ((state + 0.5) * self.tile_size).astype(int)
            img[state[1], state[0], :] = np.array([255, 255, 0])

        if self.render_mode == "human":
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
            # offset = 32 if self.agent_pov else 64
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

        elif self.render_mode == "rgb_array":
            return img