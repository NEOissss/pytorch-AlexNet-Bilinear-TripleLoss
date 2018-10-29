import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_distance(filename):
    if os.path.isdir(filename):
        path = filename
        for j in os.listdir(i):
            if 'test_result' in j:
                metric = np.load(j)
                name = j.split('.')[0]
                break
    else:
        path = '.'
        metric = np.load(filename)
        name = filename.split('.')[0]

    x = np.arange(metric.shape[0])
    pos = metric[:, 0]
    neg = metric[:, 1:]
    mask = pos < neg.min()
    x1 = [i for i, j in enumerate(mask) if j]
    x0 = [i for i, j in enumerate(mask) if not j]
    pos1 = [pos[i] for i, j in enumerate(mask) if j]
    pos0 = [pos[i] for i, j in enumerate(mask) if not j]

    plt.clf()
    for i in range(9):
        if i == 0:
            plt.scatter(x, neg[:, i], c='b', alpha=0.2, label='Negative')
        else:
            plt.scatter(x, neg[:, i], c='b', alpha=0.2)
    plt.scatter(x0, pos0, c='r', alpha=0.2, label='False Negative')
    plt.scatter(x1, pos1, c='g', alpha=1.0, label='Positive')
    plt.legend()
    plt.xlabel('#Case')
    plt.ylabel('Distance')
    plt.savefig('{:s}/{:s}'.format(path, name))


if __name__ == '__main__':
    for i in sys.argv[1:]:
        plot_distance(i)