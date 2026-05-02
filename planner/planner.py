import numpy as np

class Planner:
    """
    Chooses next action based on belief.
    Currently: greedy (max probability)
    """

    def get_next_move(self, belief, visited):
        scores = np.copy(belief)

        for i in range(belief.shape[0]):
            for j in range(belief.shape[1]):
                if (i, j) in visited:
                    scores[i][j] = -1  # avoid revisiting

        next_pos = np.unravel_index(np.argmax(scores), scores.shape)
        return next_pos