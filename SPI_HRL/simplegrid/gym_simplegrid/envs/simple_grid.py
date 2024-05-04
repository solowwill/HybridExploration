from __future__ import annotations
import logging
import numpy as np
import gym_simplegrid.rendering as r
from gym_simplegrid.window import Window
from gymnasium import spaces, Env
import sys

MAPS = {
    "4x4": ["0000", "0101", "0001", "1000"],
    "8x8": [
        "00000000",
        "00000000",
        "00010000",
        "00000100",
        "00010000",
        "01100010",
        "01001010",
        "00010000",
    ],
}

class SimpleGridEnv(Env):
    """
    Simple Grid Environment

    The environment is a grid with obstacles (walls) and agents. The agents can move in one of the four cardinal directions. If they try to move over an obstacle or out of the grid bounds, they stay in place. Each agent has a unique color and a goal state of the same color. The environment is episodic, i.e. the episode ends when the agents reaches its goal.

    To initialise the grid, the user must decide where to put the walls on the grid. This can be done by either selecting an existing map or by passing a custom map. To load an existing map, the name of the map must be passed to the `obstacle_map` argument. Available pre-existing map names are "4x4" and "8x8". Conversely, if to load custom map, the user must provide a map correctly formatted. The map must be passed as a list of strings, where each string denotes a row of the grid and it is composed by a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell. An example of a 4x4 map is the following:
    ["0000", 
     "0101", 
     "0001", 
     "1000"]

    Assume the environment is a grid of size (nrow, ncol). A state s of the environment is an elemente of gym.spaces.Discete(nrow*ncol), i.e. an integer between 0 and nrow * ncol - 1. Assume nrow=ncol=5 and s=10, to compute the (x,y) coordinates of s on the grid the following formula are used: x = s // ncol  and y = s % ncol.
     
    The user can also decide the starting and goal positions of the agent. This can be done by through the `options` dictionary in the `reset` method. The user can specify the starting and goal positions by adding the key-value pairs(`starts_xy`, v1) and `goals_xy`, v2), where v1 and v2 are both of type int (s) or tuple (x,y) and represent the agent starting and goal positions respectively. 
    """
    metadata = {"render_modes": ["human", "rgb_array"], 'render_fps': 5}
    FREE: int = 0
    WALL: int = 1
    LAVA: int = 2
    GOAL: int = 3
    MOVES: dict[int,tuple] = {
        0: (-1, 0), #UP
        1: (1, 0),  #DOWN
        2: (0, -1), #LEFT
        3: (0, 1)   #RIGHT
    }

    def __init__(self,     
        obstacle_map: str | list[str],
        start: tuple,
        render_mode: str | None = None,
    ):
        """
        Initialise the environment.

        Parameters
        ----------
        agent_color: str
            Color of the agent. The available colors are: red, green, blue, purple, yellow, grey and black. Note that the goal cell will have the same color.
        obstacle_map: str | list[str]
            Map to be loaded. If a string is passed, the map is loaded from a set of pre-existing maps. The names of the available pre-existing maps are "4x4" and "8x8". If a list of strings is passed, the map provided by the user is parsed and loaded. The map must be a list of strings, where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell. 
            An example of a 4x4 map is the following:
            ["0000",
             "0101", 
             "0001",
             "1000"]
        """

        # Env confinguration
        self.map = self.parse_obstacle_map(obstacle_map) #walls
        self.nrow, self.ncol = self.map.shape

        # Set the starting state
        self.start_xy = start
        # Set the goal state
        self.goal_xy = tuple(np.argwhere(self.map==self.GOAL).flatten())

        self.action_space = spaces.Discrete(len(self.MOVES))
        self.observation_space = spaces.Discrete(n=self.nrow*self.ncol)

        # Rendering configuration
        self.render_mode = render_mode
        self.window = None
        self.agent_color = 'yellow'
        self.tile_cache = {}
        self.fps = self.metadata['render_fps']

    def reset(
            self, 
            seed: int | None = None,
            options: dict = dict()
        ) -> tuple:
        """
        Reset the environment.

        Parameters
        ----------
        seed: int | None
            Random seed.
        options: dict
            Optional dict that allows you to define the start (`start_loc` key) and goal (`goal_loc`key) position when resetting the env. By default options={}, i.e. no preference is expressed for the start and goal states and they are randomly sampled.
        """

        # Set seed
        super().reset(seed=seed)

        # Set the goal state
        self.goal_xy = tuple(np.argwhere(self.map==self.GOAL).flatten())

        # initialise internal vars
        self.agent_xy = self.start_xy

        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()

        # Check integrity

        self.integrity_checks()

        #if self.render_mode == "human":
        self.render()

        return self.get_obs(), self.get_info()
    
    def step(self, action: int):
        """
        Take a step in the environment.
        """
        #assert action in self.action_space

        # Get the current position of the agent
        row, col = self.agent_xy
        dx, dy = self.MOVES[action]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy

        # Compute the reward
        self.reward = self.get_reward(target_row, target_col)
        
        # Check if the move is valid
        # NOTE: No longer check if the agent is free
        if self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):

            self.agent_xy = (target_row, target_col)
            self.done = self.on_goal()

        # if self.render_mode == "human":
        self.render()

        return self.get_obs(), self.reward, self.done, False, self.get_info()
    
    
    # Parse the obstacle map
    def parse_obstacle_map(self, obstacle_map) -> np.ndarray:

        map_str = np.asarray(obstacle_map, dtype='c')
        map_int = np.asarray(map_str, dtype=int)
        return map_int
    
    def to_s(self, s: tuple) -> int:
        """
        Transform a (row, col) point to a state in the observation space.
        """
        return s[0] * self.ncol + s[1]

    def to_xy(self, s: int) -> tuple[int, int]:
        """
        Transform a state in the observation space to a (row, col) point.
        """
        return (s // self.ncol, s % self.ncol)

    
    def integrity_checks(self) -> None:
        # check that goals do not overlap with walls

        assert self.map[self.start_xy] == self.FREE, \
            f"Start position {self.start_xy} overlaps with a wall."
        assert self.map[self.goal_xy] == self.GOAL, \
            f"Goal position {self.goal_xy} overlaps with a wall."
        assert self.is_in_bounds(*self.start_xy), \
            f"Start position {self.start_xy} is out of bounds."
        assert self.is_in_bounds(*self.goal_xy), \
            f"Goal position {self.goal_xy} is out of bounds."

    def is_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free.
        """
        return self.map[row, col] != self.WALL
    
    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check if a target cell is in the grid bounds.
        """
        return 0 <= row < self.nrow and 0 <= col < self.ncol
    
    def is_lava(self, row: int, col: int) -> bool:
        """
        Check if a cell is lava.
        """
        return self.map[row, col] == self.LAVA
    
    def on_goal(self) -> bool:
        """
        Check if the agent is on its own goal.
        """
        return self.agent_xy == self.goal_xy
    
    def get_reward(self, x: int, y: int) -> float:
        """
        Get the reward of a given cell.
        """
        if not self.is_in_bounds(x,y):
            return (-1.0, 0)
        if (x, y) == self.goal_xy:
            return (10.0, 0.0)
        elif self.is_lava(x,y):
            return (-1.0, -10.0)
        else:
            return (-1.0,0)

    def get_obs(self) -> int:
        return self.agent_xy
    
    def get_info(self) -> dict:
        return {'agent_xy': self.agent_xy}

    def close(self):
        """
        Close the environment.
        """
        if self.window:
            self.window.close()
        return None

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "human":
            img = self.render_frame()
            if not self.window:
                self.window = Window()
                self.window.show(block=False)
            caption = ''
            self.window.show_img(img, caption, self.fps)
            return None
        elif self.render_mode == "rgb_array":
            return self.render_frame()
        else:
            raise ValueError(f"Unsupported rendering mode {self.render_mode}")
    
    def render_frame(self, tile_size=r.TILE_PIXELS, highlight_mask=None):
        """
        @NOTE: Once again, if agent position is (x,y) then, to properly 
        render it, we have to pass (y,x) to the grid.render method.

        tile_size: tile size in pixels
        """
        width = self.ncol
        height = self.nrow

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(width, height), dtype=bool)

        # Compute the total grid size
        width_px = width * tile_size
        height_px = height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render grid with obstacles
        for x in range(self.nrow):
            for y in range(self.ncol):
                if self.map[x,y] == self.WALL:
                    cell = r.Wall(color='black')
                elif self.map[x,y] == self.LAVA:
                    cell = r.Lava(color='orange')

                elif self.map[x,y] == self.GOAL:
                    cell = r.Goal(color='green')
                else:
                    cell = r.Empty(color='white')

                img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        '''        # Render start
        x, y = self.start_xy
        cell = r.ColoredTile(color="red")
        img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        # Render goal
        x, y = self.goal_xy
        cell = r.ColoredTile(color="green")
        img = self.update_cell_in_frame(img, x, y, cell, tile_size)'''

        # Render agent
        x, y = self.agent_xy
        cell = r.Agent(color=self.agent_color)
        img = self.update_cell_in_frame(img, x, y, cell, tile_size)

        return img
        
    def render_cell(
        self,
        obj: r.WorldObj,
        img: np.ndarray,
        highlight=False,
        tile_size=r.TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        if not isinstance(obj, r.Agent):
            key = (None, highlight, tile_size)
            key = obj.encode() + key if obj else key

            if key in self.tile_cache:
                return self.tile_cache[key]

        if obj != None:
            obj.render(img)


        # Draw the grid lines (top and left edges)
        r.fill_coords(img, r.point_in_rect(0, 0.031, 0, 1), (170, 170, 170))
        r.fill_coords(img, r.point_in_rect(0, 1, 0, 0.031), (170, 170, 170))

        # Downsample the image to perform supersampling/anti-aliasing
        #img = r.downsample(img, subdivs)

        # Cache the rendered tile
        if not isinstance(obj, r.Agent):
            self.tile_cache[key] = img

        return img

    def update_cell_in_frame(self, img, x, y, cell, tile_size):
        """
        Parameters
        ----------
        img : np.ndarray
            Image to update.
        x : int
            x-coordinate of the cell to update.
        y : int
            y-coordinate of the cell to update.
        cell : r.WorldObj
            New cell to render.
        tile_size : int
            Size of the cell in pixels.
        """

        height_min = x * tile_size
        height_max = (x+1) * tile_size
        width_min = y * tile_size
        width_max = (y+1) * tile_size

        small_img = np.copy(img[height_min:height_max, width_min:width_max, :])

        tile_img = self.render_cell(cell, small_img, tile_size=tile_size)

        img[height_min:height_max, width_min:width_max, :] = tile_img
        return img


# Define a stochastic SimpleGrid where the agent has a .75 chance of moving 
# correctly, a .1 chance of moving to each side, and a .05 chance of moving 
# in the opposite direction
class Stochastic_SimpleGridEnv(SimpleGridEnv):

    def __init__(self,     
            obstacle_map: str | list[str],
            start: tuple,
            render_mode: str | None = None,
        ):
        SimpleGridEnv.__init__(self, obstacle_map, start, render_mode)

        self.success_probs = np.array([.75, .1, .1, .05])
    

    # Subclass the step 
    def step(self, targ_act: int):
        """
        Take a step in the environment.
        """
        #assert action in self.action_space

        # Get the current position of the agent
        row, col = self.agent_xy

        # Create the array of possible actions
        act_arr = np.array([targ_act, (targ_act-1)%4, (targ_act+1)%4, (targ_act+2)%4])
        # Choose the action that actually happens 
        action = np.random.choice(act_arr, 1, p=self.success_probs)[0]

        dx, dy = self.MOVES[action]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy

        # Compute the reward
        self.reward = self.get_reward(target_row, target_col)
        
        # Check if the move is valid
        # NOTE: No longer check if the agent is free
        if self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):

            self.agent_xy = (target_row, target_col)
            self.done = self.on_goal()

        # if self.render_mode == "human":
        self.render()

        return self.get_obs(), self.reward, self.done, False, self.get_info()


    