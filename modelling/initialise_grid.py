import numpy as np


def random_start(locations, num_classes):
    shape = np.shape(locations)
    for x in range(shape[0]):
        for y in range(shape[1]):
            locations[x][y] = np.random.randint(0, num_classes)


def centre_start(locations, grid_size, num_seeds):
    centre_x = grid_size[0] // 2
    range_x = grid_size[1] // 4
    centre_y = grid_size[1] // 2
    range_y = grid_size[1] // 4

    for i in range(num_seeds):
        seed_x = np.random.randint(centre_x - range_x, centre_x + range_x)
        seed_y = np.random.randint(centre_y - range_y, centre_y + range_y)
        locations[seed_x, seed_y] = 1


class InitialiseGrid:
    pass
