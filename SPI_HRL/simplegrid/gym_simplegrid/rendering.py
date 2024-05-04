import math
import numpy as np

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'white' : np.array([255, 255, 255]),
    'black' : np.array([0, 0, 0]),
    'orange' : np.array([255, 165, 0]),
    'green' : np.array([0, 200, 0]),
    'yellow': np.array([255, 205, 0]),
    'red'   : np.array([255, 0, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'grey'  : np.array([100, 100, 100])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'white' : 0,
    'black' : 1,
    'orange': 2,
    'green' : 3,
    'yellow': 4,
    'red'   : 5,
    'blue'  : 6,
    'purple': 7,
    'grey'  : 8
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'empty'         : 0,
    'wall'          : 1,
    'lava'          : 2,
    'goal'          : 3,
    'agent'         : 4,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color
    return img


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x-cx)*(x-cx) + (y-cy)*(y-cy) <= r * r
    return fn

def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax
    return fn

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None


    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)


    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError
    
class Lava(WorldObj):
    def __init__(self, color):
        super().__init__('lava', color)

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Goal(WorldObj):
    def __init__(self, color):
        super().__init__('goal', color)

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Agent(WorldObj):
    def __init__(self, color):
        super().__init__('agent', color)
    
    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Empty(WorldObj):
    def __init__(self, color='white'):
        super().__init__('empty', color)

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Wall(WorldObj):
    def __init__(self, color='grey '):
        super().__init__('wall', color)

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])