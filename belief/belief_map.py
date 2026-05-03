import numpy as np

class BeliefMap:
    def __init__(self, size):
        self.size = size
        self.belief = np.ones((size, size), dtype=float) / (size * size)
        self.explored = np.zeros((size, size), dtype=float)
        self.eps = 1e-9

    def update(self, pos, observation_prob):
        """
        Update only the observed cell.
        This keeps the belief stable and prevents global distortion.
        """
        x, y = pos

        self.explored[x][y] = 1.0

        # observation_prob should be in [0, 1]
        obs = float(np.clip(observation_prob, 0.01, 0.99))

        # local Bayesian-style update
        self.belief[x][y] *= (0.05 + 0.95 * obs)

        # slight decay for still-unexplored cells
        self.belief[self.explored == 0] *= 0.995

        # numerical safety
        self.belief = np.maximum(self.belief, self.eps)
        self.belief /= self.belief.sum()

    def compute_entropy(self):
        p = np.maximum(self.belief, self.eps)
        return -np.sum(p * np.log(p))

    def get_uncertainty_map(self):
        uniform = 1.0 / (self.size * self.size)
        return np.abs(self.belief - uniform)