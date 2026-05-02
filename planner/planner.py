import numpy as np

class Planner:

    def __init__(self, size):
        self.visit_count = np.zeros((size, size))

        # weights (tunable)
        # self.alpha = 1.0    # probability
        # self.beta = 0.8     # uncertainty
        # self.gamma = 1.5    # revisit penalty
        # self.delta = 0.3    # distance penalty

        self.alpha = 1.0
        self.beta = 0.7   # ↑ exploration
        self.gamma = 2.2  # ↑ revisit penalty
        self.delta = 0.4

    def update_visit(self, pos):
        self.visit_count[pos[0]][pos[1]] += 1

    def get_next_move(self, belief_map, uncertainty_map, drone_pos):

        best_score = -1e9
        best_pos = drone_pos

        for i in range(belief_map.shape[0]):
            for j in range(belief_map.shape[1]):

                prob = belief_map[i][j]
                uncertainty = uncertainty_map[i][j]
                visits = self.visit_count[i][j]

                dist = abs(drone_pos[0] - i) + abs(drone_pos[1] - j)

                score = (
                    self.alpha * prob +
                    self.beta * uncertainty -
                    self.gamma * visits -
                    self.delta * dist
                )

                if score > best_score:
                    best_score = score
                    best_pos = (i, j)

        return best_pos