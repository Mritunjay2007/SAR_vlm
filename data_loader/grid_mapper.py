import os
import sys
import random
import shutil

# FIX: add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GRID_SIZE

MIXED_PATH = "dataset/mixed"
GRID_PATH = "dataset/grid"

os.makedirs(GRID_PATH, exist_ok=True)


def create_grid():
    files = os.listdir(MIXED_PATH)

    total_cells = GRID_SIZE * GRID_SIZE

    selected = random.sample(files, total_cells)

    idx = 0

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):

            src = os.path.join(MIXED_PATH, selected[idx])
            dst = os.path.join(GRID_PATH, f"{i}_{j}.jpg")

            shutil.copy(src, dst)

            idx += 1

    print("✅ Grid dataset created!")


if __name__ == "__main__":
    create_grid()