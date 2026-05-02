import matplotlib.pyplot as plt

from environment.grid import GridEnvironment
from agent.drone import Drone
from belief.belief_map import BeliefMap
from perception.sensor import Sensor
from planner.planner import Planner
from utils.visualization import plot_state
from config import GRID_SIZE, NUM_STEPS

def main():

    # Initialize components
    env = GridEnvironment(GRID_SIZE)
    drone = Drone()
    belief = BeliefMap(GRID_SIZE)
    sensor = Sensor()
    planner = Planner()

    visited = set()

    plt.figure()

    print(f"True victim location (hidden): {env.victim_pos}")

    for step in range(NUM_STEPS):

        pos = drone.position
        visited.add(pos)

        # 1. Observe (perception)
        obs_prob = sensor.observe(env, pos)

        # 2. Update belief (Bayesian update)
        belief.update(pos, obs_prob)

        # 3. Compute uncertainty
        entropy = belief.compute_entropy()

        # 4. Visualization
        plot_state(belief.belief, pos, drone.path, step)

        print(f"Step {step}: Position={pos}, Obs={obs_prob:.2f}, Entropy={entropy:.4f}")

        # 5. Check if found
        if env.is_victim(pos):
            print(f"\n✅ Victim found at {pos} in {step} steps!")
            break

        # 6. Plan next move
        next_pos = planner.get_next_move(belief.belief, visited)

        # 7. Move drone
        drone.move(next_pos)

    plt.show()


if __name__ == "__main__":
    main()