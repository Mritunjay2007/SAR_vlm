import random
from config import TRUE_POS_RANGE, FALSE_POS_RANGE

class Sensor:
    """
    Simulates VLM-like probabilistic perception.
    Returns a probability instead of binary output.
    """

    def observe(self, env, pos):
        if env.is_victim(pos):
            return random.uniform(*TRUE_POS_RANGE)
        else:
            return random.uniform(*FALSE_POS_RANGE)