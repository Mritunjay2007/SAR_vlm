import numpy as np

class Planner:

    def get_next_move(self, belief, visited, drone_pos):
        best_score = -1
        best_pos = drone_pos

        for i in range(belief.shape[0]):
            for j in range(belief.shape[1]):

                if (i, j) in visited:
                    continue

                prob = belief[i][j]
                dist = abs(drone_pos[0] - i) + abs(drone_pos[1] - j)

                score = prob / (dist + 1)

                if score > best_score:
                    best_score = score
                    best_pos = (i, j)

        return best_pos