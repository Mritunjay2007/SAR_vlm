import random
from config import TRUE_POS_RANGE, FALSE_POS_RANGE

class Sensor:

    def observe(self, env, pos):
        if env.is_victim(pos):
            return random.uniform(*TRUE_POS_RANGE)
        else:
            return random.uniform(*FALSE_POS_RANGE)

    def get_visible_cells(self, pos, grid_size, radius=1):
        visible = []
        x, y = pos

        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if 0 <= i < grid_size and 0 <= j < grid_size:
                    visible.append((i, j))

        return visible