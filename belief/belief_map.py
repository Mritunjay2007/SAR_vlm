import numpy as np
from config import DECAY

class BeliefMap:
    """
    Maintains probability distribution over grid.
    Implements Bayesian update.
    """

    def __init__(self, size):
        self.size = size
        self.belief = np.ones((size, size)) / (size * size)

    def update(self, pos, observation_prob):
        """
        Bayesian Update:
        P_new(c) ∝ P_old(c) * P(observation | c)
        """

        new_belief = np.zeros_like(self.belief)

        for i in range(self.size):
            for j in range(self.size):

                prior = self.belief[i][j]

                # Likelihood model
                if (i, j) == pos:
                    likelihood = observation_prob
                else:
                    likelihood = DECAY  # slight decay

                new_belief[i][j] = prior * likelihood

        # Normalize (very important step)
        self.belief = new_belief / np.sum(new_belief)

    def compute_entropy(self):
        """
        Entropy = measure of uncertainty
        H = -∑ P log P
        """
        epsilon = 1e-9
        return -np.sum(self.belief * np.log(self.belief + epsilon))