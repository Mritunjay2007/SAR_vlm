import numpy as np
from perception.vlm_clip import CLIPVLM


class Sensor:

    def __init__(self):
        self.vlm = CLIPVLM()

    def observe(self, env, pos):
        """
        Uses CLIP to analyze the image in this cell
        Returns probability of victim presence
        """

        image_path = env.get_image(pos)

        if image_path is None:
            return 0.1  # safe fallback

        probs = self.vlm.predict(image_path)

        # unpack predictions
        p_human = probs[0]
        p_hidden = probs[1]
        p_foot = probs[2]
        p_cloth = probs[3]
        p_empty = probs[4]

        # 🔥 YOUR RESEARCH CONTRIBUTION:
        # semantic fusion of multiple evidence types
        p_victim = (
            0.5 * p_human +
            0.2 * p_hidden +
            0.15 * p_foot +
            0.10 * p_cloth +
            0.05 * (1 - p_empty)
        )

        # clamp for safety
        return float(np.clip(p_victim, 0.01, 0.99))

    def get_visible_cells(self, pos, grid_size, radius=1):
        """
        Simulate drone camera footprint (FOV)
        radius=1 → 3x3
        radius=2 → 5x5
        """

        visible = []
        x, y = pos

        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if 0 <= i < grid_size and 0 <= j < grid_size:
                    visible.append((i, j))

        return visible