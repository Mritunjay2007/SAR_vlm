class Drone:
    """
    Represents the agent exploring the environment.
    """
    def __init__(self, start=(0, 0)):
        self.position = start
        self.path = [start]

    def move(self, next_pos):
        self.position = next_pos
        self.path.append(next_pos)