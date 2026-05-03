import numpy as np
from planner.astar import astar
from planner.frontier import get_frontiers

class Planner:
    def __init__(self, size):
        self.size = size
        self.visit_count = np.zeros((size, size), dtype=int)

        # weights
        self.w_frontier = 2.5
        self.w_prob = 1.5
        self.w_novelty = 2.0
        self.w_path = 1.0
        self.w_visit = 3.0

    def update_visit(self, pos):
        self.visit_count[pos] += 1

    def _frontier_bonus(self, explored, cell):
        x, y = cell
        bonus = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if explored[nx][ny] == 0:
                    bonus += 1
        return bonus / 4.0

    def _best_from_candidates(self, belief, explored, drone_pos, candidates):
        best_score = -1e18
        best_goal = drone_pos
        best_path = []

        for c in candidates:
            if c == drone_pos:
                continue

            visits = self.visit_count[c]
            if visits > 2:
                continue

            path = astar(drone_pos, c, self.size)
            if len(path) == 0:
                continue

            frontier_bonus = self._frontier_bonus(explored, c)
            prob = float(belief[c])
            novelty = 1.0 if visits == 0 else 0.0
            path_cost = len(path)

            score = (
                self.w_frontier * frontier_bonus +
                self.w_prob * prob +
                self.w_novelty * novelty -
                self.w_path * path_cost -
                self.w_visit * visits
            )

            if score > best_score:
                best_score = score
                best_goal = c
                best_path = path

        return best_goal, best_path, best_score

    def select_goal(self, belief, explored, drone_pos):
        frontiers = get_frontiers(explored)

        # main candidate set
        candidates = [c for c in frontiers if self.visit_count[c] <= 2]

        # fallback if no frontier exists
        if len(candidates) == 0:
            unexplored = np.argwhere(explored == 0)
            if len(unexplored) == 0:
                return drone_pos, [], 0.0
            candidates = [tuple(x) for x in unexplored]

        best_goal, best_path, best_score = self._best_from_candidates(
            belief, explored, drone_pos, candidates
        )

        # absolute fallback if everything was filtered out
        if len(best_path) == 0:
            x, y = drone_pos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if explored[nx][ny] == 0:
                        return (nx, ny), [(nx, ny)], -1.0

            # if no unexplored neighbor exists, move to any valid neighbor
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    return (nx, ny), [(nx, ny)], -2.0

        print(f"[Planner] Goal={best_goal}, Score={best_score:.4f}, PathLen={len(best_path)}")
        return best_goal, best_path, best_score

    def escape_move(self, explored, drone_pos):
        """
        Safety fallback when the planner would otherwise stall.
        """
        x, y = drone_pos
        best = drone_pos
        best_score = -1e18

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                visit_penalty = self.visit_count[nx][ny]
                unexplored_bonus = 1.0 if explored[nx][ny] == 0 else 0.0
                score = unexplored_bonus - 0.5 * visit_penalty

                if score > best_score:
                    best_score = score
                    best = (nx, ny)

        return best

    def get_next_move(self, belief, explored, drone_pos):
        goal, path, _ = self.select_goal(belief, explored, drone_pos)

        # if a valid shortest path exists, take only the first step
        if len(path) > 0:
            next_step = path[0]
            if next_step != drone_pos:
                return next_step

        # hard safety fallback
        return self.escape_move(explored, drone_pos)