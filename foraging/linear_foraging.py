import gymnasium as gym
from gymnasium import spaces, logger, spaces, utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np
from gymnasium.error import DependencyNotInstalled
# debugger
import pdb

STAY = 0
LEFT = 1
RIGHT = 2

MAPS = {
    "short": [
        "LALLLBL"
    ],
    "long": [
        "LALLLLLBL"
    ]
}


class foraging_playground_linear(gym.Env):
    """
    ### Task Setup
    Foraging playground involves agent crossing a grid world from Start(S) to find rewards(A or B), by walking on the 
    tundra(L) that has no food or resources. Reward location and time are scheduled in a periodic pattern, ABAB, which
    means that reward patches A and B alternately get refreshed every T timesteps, and decays exponentially to 0 every
    T timesteps. A and B never have reward available at the same time. 
    
    Hence, the agent has to learn the reward pattern and leave the patch at appropriate times to arrive at the next rewarding
    patch. The task has a limited duration (timesteps) and the goal of agent is to maximize the total amount of reward
    within this fixed time.


    ### Action Space
    The agent takes a 1-element vector for actions.
    The action space is `(dir)`, where `dir` decides direction to move in which can be:
    - 0: STAY
    - 1: LEFT
    - 2: RIGHT

    ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    
    ### Rewards
    Reward schedule: e.g. reward period T = 10
    A has base reward amount r0 at time 0, decays exponentially to 0 at time 10,
    then B has r0 at time 10, decays exponentially to 0 at time 20, 
    then A gets refreshed to r0 at time 20, ..., etc.
    - Reach reward patches (A or B): reward = r0 * decay rate ^ (elapsed timesteps since this patch's refresh time)
    - Reach Tundra(L): no reward 

    ### Arguments
    env = foraging_playground_linear(
                    render_mode='human', 
                    map_name="short",
                    base_reward = 10.0,
                    decay_rate = 0.6,
                    reward_period = 10,
                    session_duration = 100
                    )

    `desc`: Used to specify custom map for frozen lake. For example,

        desc=["ALLLLLLB"]

    `map_name`: ID to use any of the preloaded maps.
"""

    metadata = {
            "render_modes": ["human", "ansi","rgb_array"],
            "render_fps": 4,
        }

    def __init__(self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="short",
        base_reward = 10.0,
        decay_rate = 0.6,
        reward_period = 10,
        session_duration = 100
        ):

        super(foraging_playground_linear, self).__init__()
        if desc is None and map_name is None:
            logger.warn(
                "You are calling the environment without specifying any map. "
                "You can use the pre-loaded or specify customed map at initialization, "
            )
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        # pdb.set_trace()
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 3 #number of actions
        nS = ncol * nrow #number of states
        
        self.s = np.where(np.array(desc == b"A").astype("float64").ravel() == 1.0)[0][0]
        # state transition dict
        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        # total reward 
        self.tol_reward = 0
        # game render mode
        self.render_mode = render_mode

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.lastaction = None
        self.newreward = 0.0
        self.time = -1
        self._populate_state_transitions()


        # ADJUSTABLE VARIABLE
        self.base_reward = base_reward
        self.decay_rate = decay_rate
        self.reward_period = reward_period
        self.session_duration = session_duration

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 768))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.clock = None
        self.window_surface = None
        self.grass_img = None
        self.elf_images = None
        self.food_img = None
        self.start_img = None


                                ###### Class functions #######

    # An action leads to change of state    
    def to_s(self, row, col):
        return row * self.ncol + col
    # An action leads to change of location in the grid
    def inc(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == STAY:
            pass
        return (row, col)
    
    # update the state based on last state and action 
    def state_update(self, row, col, action):
            newrow, newcol = self.inc(row, col, action)
            newstate = self.to_s(newrow, newcol)
            newletter = self.desc[newrow, newcol]
            return newstate,newletter
    
    # Reward functions
    def get_reward(self, location):
        cur_avail_patch_id = self.time // self.reward_period  # e.g. t=8 means patch 0 (A), t= 15 means patch 1 (B), t=21 means patch 2 (A)
        cur_reward_location = b"A" if cur_avail_patch_id % 2 == 0 else b"B"

        # if the patch is the current rewarded location
        if cur_reward_location == location:
            curr_patch_start_time = self.reward_period * cur_avail_patch_id
            patch_stay = self.time - curr_patch_start_time
            reward = self.base_reward * self.decay_rate**(patch_stay)
        # if the path is not available
        else:   
            reward = 0

        return reward

    def get_state(self):
        state = [self.s, self.tol_reward, self.time]
        return state

    def return_state(self,state):
        self.s = state[0]
        self.tol_reward = state[1]
        self.time = state[2]

    def harvest_outcome(self, newletter, action):
        # action: current action
        # self.lastaction
        cur_position = (self.s // self.ncol, self.s % self.ncol)
        if newletter == b'A' or newletter ==b'B':
            newreward = self.get_reward(newletter)
        else:
            newreward = 0.0
        return newreward

    # populate state transition outcomes given curr state & action
    def _populate_state_transitions(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self.to_s(row, col)
                for a in range(3):
                    li = self.P[s][a]
                    newstate, newletter  = self.state_update(row,col,a)
                    li.append((1.0, newstate, 0, False)) # reward and termination are dynamic
    
        
    def step(self, a):
        # pdb.set_trace()
        row, col = self.s // self.ncol, self.s % self.ncol
        newstate, newletter = self.state_update(row, col, a)
        self.s = newstate
        self.time += 1
        self.newreward = self.harvest_outcome(newletter,a)
        self.lastaction = a
        self.tol_reward += self.newreward

        # check if game over
        termination = bool(self.time >= self.session_duration-1)
        truncated = False

        if self.render_mode == "human":
            self.render(self.render_mode)
        return (int(newstate), self.newreward, termination, truncated, {"total_reward": self.tol_reward, "time": self.time})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = np.where(np.array(self.desc == b"A").astype("float64").ravel() == 1.0)[0][0]
        self.lastaction = None
        self.newreward = 0.0
        self.tol_reward = 0.0
        self.time = -1
        if self.render_mode == "human":
            self.render(self.render_mode)
        return int(self.s),{}
    
    def render(self, mode = "rgb_array"):
        if mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        elif mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self,mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        if self.window_surface is None:
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Foraging Playground")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)
        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.grass_img is None:
            self.grass_img = pygame.transform.scale(
                pygame.image.load("./img/ice.png"), self.cell_size)
            
        if self.food_img is None:
            self.food_img = pygame.transform.scale(
                    pygame.image.load("./img/food.png"), self.cell_size
                )
        
        if self.start_img is None:
            self.start_img = pygame.transform.scale(
                    pygame.image.load("./img/stool.png"), self.cell_size
                )
        if self.elf_images is None:
            elfs = [
                    "./img/elf_stay.png",
                    "./img/elf_left.png",
                    "./img/elf_up.png",
                    "./img/elf_right.png",
                    "./img/elf_down.png",
                ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]
        
        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.grass_img, pos) #grass 

                if desc[y][x] == b"A" or desc[y][x] ==b"B":
                    self.window_surface.blit(self.food_img, pos) #food patch
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos) #start
                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        # pdb.set_trace()
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 0
        elf_img = self.elf_images[last_action]
        self.window_surface.blit(elf_img, cell_rect)
        
        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Up', 'Right', 'DOWN',][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame
            
            pygame.display.quit()
            pygame.quit()