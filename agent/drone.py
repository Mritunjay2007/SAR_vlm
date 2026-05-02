class Drone:
    def __init__(self, start=(0, 0)):
        self.position = start
        self.path = [start]

    def move_towards(self, target):
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

        self.position = (x, y)
        self.path.append(self.position)