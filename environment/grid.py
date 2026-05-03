import os
import numpy as np
import random


class GridEnvironment:

    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size))

        # randomly place victim (simulation ground truth)
        self.victim_pos = self._place_victim()

    def _place_victim(self):
        x = random.randint(0, self.size - 1)
        y = random.randint(0, self.size - 1)
        self.grid[x][y] = 1
        return (x, y)

    def is_victim(self, pos):
        return self.grid[pos[0]][pos[1]] == 1

    def get_image(self, pos):
        """
        Returns image path for given grid cell
        """
        x, y = pos
        path = f"dataset/grid/{x}_{y}.jpg"

        if not os.path.exists(path):
            # fallback safety
            return None

        return path