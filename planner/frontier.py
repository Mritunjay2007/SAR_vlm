def get_frontiers(explored):
    """
    Frontier = explored cell that touches at least one unexplored neighbor.
    """
    size = explored.shape[0]
    frontiers = []

    for i in range(size):
        for j in range(size):
            if explored[i][j] == 0:
                continue

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < size and 0 <= nj < size:
                    if explored[ni][nj] == 0:
                        frontiers.append((i, j))
                        break

    return frontiers