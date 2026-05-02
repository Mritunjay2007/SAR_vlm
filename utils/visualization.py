import matplotlib.pyplot as plt

def plot_state(belief, drone_pos, path, step):
    plt.clf()

    # Belief heatmap
    plt.imshow(belief, cmap='hot')
    plt.colorbar(label='Probability')

    # Drone position
    plt.scatter(drone_pos[1], drone_pos[0], c='blue', label='Drone')

    # Path
    if len(path) > 1:
        ys = [p[1] for p in path]
        xs = [p[0] for p in path]
        plt.plot(ys, xs, c='cyan', linewidth=1)

    plt.title(f"Step {step}")
    plt.legend()
    plt.pause(0.1)