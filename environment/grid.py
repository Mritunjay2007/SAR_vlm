import numpy as np
import random

class GridEnvironment:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size))
        self.victim_pos = self._place_victim()

    def _place_victim(self):
        x = random.randint(0, self.size - 1)
        y = random.randint(0, self.size - 1)
        self.grid[x][y] = 1
        return (x, y)

    def is_victim(self, pos):
        return self.grid[pos[0]][pos[1]] == 1

    def get_image(self, pos):
        x, y = pos
        return f"dataset/grid/{x}_{y}.jpg"