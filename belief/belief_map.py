import numpy as np
from config import DECAY

class BeliefMap:

    def __init__(self, size):
        self.size = size
        self.belief = np.ones((size, size)) / (size * size)

    def update(self, pos, observation_prob):

        new_belief = np.zeros_like(self.belief)

        for i in range(self.size):
            for j in range(self.size):

                prior = self.belief[i][j]

                if (i, j) == pos:
                    likelihood = observation_prob
                else:
                    likelihood = DECAY

                new_belief[i][j] = prior * likelihood

        new_belief /= np.sum(new_belief)

        self.belief = self.apply_spatial_smoothing(new_belief)

    def apply_spatial_smoothing(self, belief):
        smoothed = np.copy(belief)

        for i in range(self.size):
            for j in range(self.size):
                for ni, nj in self.get_neighbors(i, j):
                    smoothed[ni][nj] += 0.1 * belief[i][j]

        smoothed /= np.sum(smoothed)
        return smoothed

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors

    def compute_entropy(self):
        epsilon = 1e-9
        return -np.sum(self.belief * np.log(self.belief + epsilon))