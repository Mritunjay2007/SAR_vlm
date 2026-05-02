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

        # ✅ Update visit count (IMPORTANT)
        planner.update_visit(pos)

        # -----------------------------
        # Camera observation (multi-cell)
        # -----------------------------
        visible_cells = sensor.get_visible_cells(pos, GRID_SIZE, radius=1)

        for cell in visible_cells:
            obs = sensor.observe(env, cell)
            belief.update(cell, obs)

        # -----------------------------
        # Compute uncertainty
        # -----------------------------
        entropy = belief.compute_entropy()
        uncertainty_map = belief.get_uncertainty_map()

        print(f"Step {step}, Pos={pos}, Entropy={entropy:.4f}")

        # -----------------------------
        # Visualization
        # -----------------------------
        if USE_IMAGE_GRID:
            show_image_grid(GRID_SIZE, drone.position)
        else:
            plot_state(belief.belief, pos, drone.path, step)

        # -----------------------------
        # Check if victim found
        # -----------------------------
        if env.is_victim(pos):
            print(f"\n✅ Victim found at {pos} in {step} steps!")
            break

        # -----------------------------
        # NEW ADVANCED PLANNER
        # -----------------------------
        target = planner.get_next_move(
            belief.belief,
            uncertainty_map,
            drone.position
        )

        # -----------------------------
        # Move step-by-step
        # -----------------------------
        drone.move_towards(target)

    plt.show()


if __name__ == "__main__":
    main()