import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from environment.grid import GridEnvironment
from agent.drone import Drone
from belief.belief_map import BeliefMap
from perception.sensor import Sensor
from planner.planner import Planner
from utils.visualization import plot_state
from utils.image_grid import show_image_grid

# Optional config values with safe fallbacks
try:
    from config import (
        GRID_SIZE,
        NUM_STEPS,
        USE_IMAGE_GRID,
        CAMERA_RADIUS,
        SOURCE_IMAGE_PATH,
        GRID_IMAGE_DIR,
    )
except ImportError:
    GRID_SIZE = 17
    NUM_STEPS = 200
    USE_IMAGE_GRID = False
    CAMERA_RADIUS = 1
    SOURCE_IMAGE_PATH = None
    GRID_IMAGE_DIR = "dataset/grid"


def fragment_image_to_grid(source_image_path: str, output_dir: str, grid_size: int) -> bool:
    """
    Split one large jungle image into grid_size x grid_size crops.

    Each crop becomes one cell image:
        output_dir/0_0.jpg
        output_dir/0_1.jpg
        ...
        output_dir/(grid_size-1)_(grid_size-1).jpg

    Returns True if fragmentation was done or already exists.
    Returns False if source image is missing.
    """
    if not source_image_path or not os.path.exists(source_image_path):
        return False

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = grid_size * grid_size
    existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpeg"))
    if len(existing) >= expected:
        return True

    img = Image.open(source_image_path).convert("RGB")
    width, height = img.size

    # Safe boundaries even when width/height are not divisible by grid_size
    xs = np.linspace(0, width, grid_size + 1).astype(int)
    ys = np.linspace(0, height, grid_size + 1).astype(int)

    for r in range(grid_size):
        for c in range(grid_size):
            left = xs[c]
            right = xs[c + 1]
            top = ys[r]
            bottom = ys[r + 1]

            crop = img.crop((left, top, right, bottom))
            crop = crop.resize((224, 224))  # CLIP-friendly fixed size
            crop.save(out_dir / f"{r}_{c}.jpg", quality=95)

    return True


def ensure_grid_dataset():
    """
    Make sure the grid dataset exists.
    If a large source image is available, fragment it into the grid.
    Otherwise assume the grid folder already exists.
    """
    os.makedirs(GRID_IMAGE_DIR, exist_ok=True)

    # If a source image is provided, build the grid from it once.
    if SOURCE_IMAGE_PATH:
        fragment_image_to_grid(SOURCE_IMAGE_PATH, GRID_IMAGE_DIR, GRID_SIZE)


def compute_metrics(drone, visited_cells, found, found_step, env):
    """
    Final run statistics for reporting.
    """
    path_len = max(0, len(drone.path) - 1)
    coverage = len(visited_cells) / float(GRID_SIZE * GRID_SIZE)
    revisit_ratio = 0.0 if path_len == 0 else 1.0 - (len(visited_cells) / float(path_len + 1))

    start = (0, 0)
    manhattan_lower_bound = abs(env.victim_pos[0] - start[0]) + abs(env.victim_pos[1] - start[1])
    efficiency = None
    if found and path_len > 0:
        efficiency = manhattan_lower_bound / float(path_len)

    return {
        "path_len": path_len,
        "coverage": coverage,
        "revisit_ratio": revisit_ratio,
        "found": found,
        "found_step": found_step,
        "efficiency": efficiency,
    }


def main():
    ensure_grid_dataset()

    # Environment = hidden ground truth
    env = GridEnvironment(GRID_SIZE)

    # Search agent
    drone = Drone(start=(0, 0))

    # Belief map + explored memory
    belief = BeliefMap(GRID_SIZE)

    # Perception module (CLIP-based)
    sensor = Sensor()

    # Frontier + A* planner
    planner = Planner(GRID_SIZE)

    # For metrics / debugging
    visited_cells = set()
    found = False
    found_step = None

    plt.figure(figsize=(8, 8))
    print(f"True victim location: {env.victim_pos}")

    # If image-grid mode is enabled, showing it every step is expensive.
    # We show the heavy grid view only occasionally.
    IMAGE_GRID_EVERY = 15

    for step in range(NUM_STEPS):
        pos = drone.position
        visited_cells.add(pos)

        # This is the planner's memory of how often a cell was visited
        planner.update_visit(pos)

        # Camera footprint: nearby cells visible at this step
        visible_cells = sensor.get_visible_cells(pos, GRID_SIZE, radius=CAMERA_RADIUS)

        # Perception + belief update for all visible cells
        # In the final CLIP version, each visible crop is passed to the VLM.
        for cell in visible_cells:
            obs_prob = sensor.observe(env, cell)
            belief.update(cell, obs_prob)

        entropy = belief.compute_entropy()
        explored_count = int(belief.explored.sum())

        print(
            f"Step {step}, Pos={pos}, "
            f"Entropy={entropy:.4f}, "
            f"Explored={explored_count}/{GRID_SIZE * GRID_SIZE}"
        )

        # Visualization
        if USE_IMAGE_GRID and step % IMAGE_GRID_EVERY == 0:
            show_image_grid(GRID_SIZE, drone.position)
        else:
            plot_state(belief.belief, pos, drone.path, step)

        # Simulation ground-truth success check
        if env.is_victim(pos):
            found = True
            found_step = step
            print(f"\n✅ Victim found at {pos} in {step} steps!")
            break

        # Planner decides the next step only
        next_step = planner.get_next_move(
            belief.belief,
            belief.explored,
            drone.position
        )

        print(f"[Move] {pos} -> {next_step}")

        # Hard safety escape if planner returns the same cell
        if next_step == pos:
            next_step = planner.escape_move(belief.explored, pos)
            print(f"[Escape] forced move -> {next_step}")

        drone.move(next_step)

    # Final metrics
    metrics = compute_metrics(drone, visited_cells, found, found_step, env)

    print("\n===== FINAL SUMMARY =====")
    print(f"Found victim: {metrics['found']}")
    print(f"Victim location: {env.victim_pos}")
    print(f"Path length: {metrics['path_len']}")
    print(f"Coverage: {metrics['coverage']:.2%}")
    print(f"Revisit ratio: {metrics['revisit_ratio']:.2%}")
    print(f"Final entropy: {belief.compute_entropy():.4f}")

    if metrics["efficiency"] is not None:
        print(f"Path efficiency (lower is better): {metrics['efficiency']:.3f}")

    plt.show()


if __name__ == "__main__":
    main()