import numpy as np


def create(progress):
    num_clusters = np.max(progress) + 1
    transition_matrix = np.zeros((num_clusters, num_clusters), dtype=int)
    start_classes = progress[0].flatten()
    end_classes = progress[-1].flatten()

    for i in range(len(start_classes)):
        transition_matrix[start_classes[i]][end_classes[i]] += 1

    return normalise(transition_matrix)


def normalise(matrix):
    dim = len(np.shape(matrix))
    if dim == 2:
        norm_matrix = []
        for i in range(np.shape(matrix)[0]):
            norm_matrix.append(matrix[i]/np.sum(matrix[i]))
        return norm_matrix
    else:
        raise ValueError('Transition matrix should have 2 dimensions, instead it has {dim}'.format(dim=dim))


class TransitionMatrix:
    pass
