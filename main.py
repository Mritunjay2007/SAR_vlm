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
    planner = Planner(GRID_SIZE)

    plt.figure()
    print(f"True victim location: {env.victim_pos}")

    for step in range(NUM_STEPS):
        pos = drone.position

        # track visit
        planner.update_visit(pos)

        # observe the local camera footprint
        visible_cells = sensor.get_visible_cells(pos, GRID_SIZE, radius=1)

        for cell in visible_cells:
            obs = sensor.observe(env, cell)
            belief.update(cell, obs)

        entropy = belief.compute_entropy()
        explored_count = int(belief.explored.sum())

        print(
            f"Step {step}, Pos={pos}, "
            f"Entropy={entropy:.4f}, "
            f"Explored={explored_count}/{GRID_SIZE * GRID_SIZE}"
        )

        # visualize
        if USE_IMAGE_GRID:
            show_image_grid(GRID_SIZE, drone.position)
        else:
            plot_state(belief.belief, pos, drone.path, step)

        # victim check
        if env.is_victim(pos):
            print(f"\n✅ Victim found at {pos} in {step} steps!")
            break

        # choose next move
        next_step = planner.get_next_move(
            belief.belief,
            belief.explored,
            drone.position
        )

        print(f"[Move] {pos} -> {next_step}")

        # if planner somehow returns same cell, force escape
        if next_step == pos:
            next_step = planner.escape_move(belief.explored, pos)
            print(f"[Escape] forced move -> {next_step}")

        drone.move(next_step)

    plt.show()


if __name__ == "__main__":
    main()