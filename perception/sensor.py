# perception/sensor.py
from perception.vlm_clip import CLIPVLM


class Sensor:
    def __init__(self, prompts=None):
        self.vlm = CLIPVLM(prompts)

    def observe(self, env, pos):
        image_path = env.get_image(pos)

        if image_path is None:
            return 0.1

        probs = self.vlm.predict(image_path)

        p_human = probs[0]
        p_hidden = probs[1]
        p_foot = probs[2]
        p_cloth = probs[3]
        p_empty = probs[4]

        p_victim = (
            0.5 * p_human +
            0.2 * p_hidden +
            0.15 * p_foot +
            0.10 * p_cloth +
            0.05 * (1 - p_empty)
        )

        return float(max(0.01, min(0.99, p_victim)))

    def get_visible_cells(self, pos, grid_size, radius=1):
        visible = []
        x, y = pos

        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if 0 <= i < grid_size and 0 <= j < grid_size:
                    visible.append((i, j))

        return visible