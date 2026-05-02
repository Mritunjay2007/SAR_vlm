import matplotlib.pyplot as plt

from environment.grid import GridEnvironment
from agent.drone import Drone
from belief.belief_map import BeliefMap
from perception.sensor import Sensor
from planner.planner import Planner
from utils.visualization import plot_state
from utils.image_grid import show_image_grid
from config import GRID_SIZE, NUM_STEPS, USE_IMAGE_GRID


def main():

    env = GridEnvironment(GRID_SIZE)
    drone = Drone()
    belief = BeliefMap(GRID_SIZE)
    sensor = Sensor()
    planner = Planner()

    visited = set()

    plt.figure()

    print(f"True victim location: {env.victim_pos}")

    for step in range(NUM_STEPS):

        pos = drone.position
        visited.add(pos)

        visible_cells = sensor.get_visible_cells(pos, GRID_SIZE, radius=1)

        for cell in visible_cells:
            obs = sensor.observe(env, cell)
            belief.update(cell, obs)

        entropy = belief.compute_entropy()

        print(f"Step {step}, Pos={pos}, Entropy={entropy:.4f}")

        if USE_IMAGE_GRID:
            show_image_grid(GRID_SIZE, drone.position)
        else:
            plot_state(belief.belief, pos, drone.path, step)

        if env.is_victim(pos):
            print(f"✅ Found at {pos} in {step} steps")
            break

        target = planner.get_next_move(belief.belief, visited, drone.position)
        drone.move_towards(target)

    plt.show()


if __name__ == "__main__":
    main()