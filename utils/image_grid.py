import matplotlib.pyplot as plt
from PIL import Image
import os

def show_image_grid(grid_size, drone_pos):
    plt.clf()

    for i in range(grid_size):
        for j in range(grid_size):

            path = f"dataset/grid/{i}_{j}.jpg"

            if os.path.exists(path):
                img = Image.open(path)

                plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
                plt.imshow(img)
                plt.axis('off')

                if (i, j) == drone_pos:
                    plt.title("D", color='blue', fontsize=6)

    plt.pause(0.1)