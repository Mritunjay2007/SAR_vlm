import matplotlib.pyplot as plt

def plot_state(belief, drone_pos, path, step):
    plt.clf()

    plt.imshow(belief, cmap='hot')
    plt.colorbar()

    plt.scatter(drone_pos[1], drone_pos[0], c='blue')

    if len(path) > 1:
        ys = [p[1] for p in path]
        xs = [p[0] for p in path]
        plt.plot(ys, xs, c='cyan')

    plt.title(f"Step {step}")
    plt.pause(0.1)