class Drone:
    def __init__(self, start=(0, 0)):
        self.position = start
        self.path = [start]

    def move(self, next_pos):
        """
        Move exactly one grid cell.
        """
        self.position = next_pos
        self.path.append(next_pos)

    def move_towards(self, target):
        """
        Backward-compatible one-step move toward a target.
        """
        x, y = self.position
        tx, ty = target

        if x < tx:
            x += 1
        elif x > tx:
            x -= 1
        elif y < ty:
            y += 1
        elif y > ty:
            y -= 1

        self.move((x, y))