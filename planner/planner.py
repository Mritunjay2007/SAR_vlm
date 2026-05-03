import numpy as np
import heapq


def astar(start, goal, size):
    if start == goal:
        return []

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = [(0, start)]
    came_from = {}
    g = {start: 0}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path[1:]

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                nxt = (nx, ny)
                tentative = g[current] + 1
                if nxt not in g or tentative < g[nxt]:
                    came_from[nxt] = current
                    g[nxt] = tentative
                    heapq.heappush(open_heap, (tentative + h(nxt, goal), nxt))

    return []


class Planner:
    """
    Frontier-guided, revisit-penalized, receding-horizon planner.
    """

    def __init__(self, size):
        self.size = size
        self.visit_count = np.zeros((size, size), dtype=np.int32)
        self.last_visit = np.full((size, size), -10_000, dtype=np.int32)
        self.step_idx = 0

    def tick(self):
        self.step_idx += 1

    def update_visit(self, pos):
        self.visit_count[pos] += 1
        self.last_visit[pos] = self.step_idx

    def get_frontiers(self, explored):
        frontiers = []
        for i in range(self.size):
            for j in range(self.size):
                if explored[i, j] == 0:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < self.size and 0 <= nj < self.size and explored[ni, nj] == 0:
                        frontiers.append((i, j))
                        break
        return frontiers

    def score_cell(self, cell, belief, explored, drone_pos, battery):
        r, c = cell
        prob = float(belief[r, c])
        visits = int(self.visit_count[r, c])

        age = self.step_idx - int(self.last_visit[r, c])
        recency_penalty = np.exp(-max(age, 0) / 5.0)

        dist = abs(drone_pos[0] - r) + abs(drone_pos[1] - c)

        frontier_bonus = 0.0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and explored[nr, nc] == 0:
                frontier_bonus += 0.25

        novelty = 1.0 if explored[r, c] == 0 else 0.0
        urgency = 1.0 + (1.0 - battery)

        score = (
            2.5 * frontier_bonus +
            2.0 * prob +
            1.5 * novelty +
            0.5 * urgency -
            0.9 * dist -
            3.5 * visits -
            1.0 * recency_penalty
        )
        return score

    def select_goal(self, belief, explored, drone_pos, battery):
        frontiers = self.get_frontiers(explored)

        if len(frontiers) == 0:
            unexplored = np.argwhere(explored == 0)
            if len(unexplored) > 0:
                frontiers = [tuple(x) for x in unexplored]
            else:
                return drone_pos, 0.0

        best_goal = drone_pos
        best_score = -1e18

        for cell in frontiers:
            if self.visit_count[cell] > 3:
                continue
            s = self.score_cell(cell, belief, explored, drone_pos, battery)
            if s > best_score:
                best_score = s
                best_goal = cell

        return best_goal, float(best_score)

    def escape_move(self, explored, drone_pos):
        x, y = drone_pos
        best = drone_pos
        best_score = -1e18

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                unexplored_bonus = 1.0 if explored[nx, ny] == 0 else 0.0
                visit_penalty = self.visit_count[nx, ny]
                s = unexplored_bonus - 0.5 * visit_penalty
                if s > best_score:
                    best_score = s
                    best = (nx, ny)

        return best

    def get_next_move(self, belief, explored, drone_pos, battery):
        goal, _ = self.select_goal(belief, explored, drone_pos, battery)
        path = astar(drone_pos, goal, self.size)

        if len(path) > 0 and path[0] != drone_pos:
            return path[0], goal

        return self.escape_move(explored, drone_pos), goal