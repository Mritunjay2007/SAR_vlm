import os
import random
from PIL import Image

GRID_SIZE = 10

FOREST_DIR = "dataset/forest"
HUMAN_DIR = "dataset/human"
OUTPUT_DIR = "dataset/mixed"


def load_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)]


def generate_grid():
    forest_images = load_images(FOREST_DIR)
    human_images = load_images(HUMAN_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):

            # Decide randomly if this cell has human
            if random.random() < 0.15:
                img_path = random.choice(human_images)
            else:
                img_path = random.choice(forest_images)

            img = Image.open(img_path).resize((224, 224))

            img.save(f"{OUTPUT_DIR}/{i}_{j}.jpg")

    print("✅ Grid dataset generated!")


if __name__ == "__main__":
    generate_grid()