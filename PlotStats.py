import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from Utils import analyze_log


def plot_stats(log_lists):
    labels = []
    stats = []
    for i in log_lists:
        if os.path.isdir(i):
            res = None
            for j in os.listdir(i):
                if 'slurm-' in j:
                    res = analyze_log(i + '/' + j)
                    break
            if not res:
                raise ValueError('Missing log file!')
            y = i + '/' + res['train']
        else:
            res = analyze_log(i)
            y = res['train']

        x = '{:s}-{:s}-{:s}-{:s}'.format(res['freeze'], res['margin'], res['batch'], res['lr'])
        labels.append(x)
        stats.append(np.load(y))

    path = 'plot'
    if not os.path.exists(path):
        os.mkdir(path)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Batch accuracy
    plt.clf()
    for i, stat in enumerate(stats):
        x = np.arange(0, stat.shape[0])
        y = stat[:, 2]
        plt.plot(x, y, label=labels[i])
    plt.xlabel('Number of Update')
    plt.ylabel('Batch Accuracy')
    plt.legend()
    plt.savefig('{:s}/ba-{:s}'.format(path, timestamp))

    # Val accuracy
    plt.clf()
    for i, stat in enumerate(stats):
        x = np.arange(0, stat.shape[0])
        y = stat[:, 3]
        plt.plot(x, y, label=labels[i])
    plt.xlabel('Number of Update')
    plt.ylabel('Val Accuracy')
    plt.legend()
    plt.savefig('{:s}/va-{:s}'.format(path, timestamp))

    # Triplet loss
    plt.clf()
    for i, stat in enumerate(stats):
        x = np.arange(0, stat.shape[0])
        y = stat[:, 4]
        plt.plot(x, y, label=labels[i])
    plt.xlabel('Number of Update')
    plt.ylabel('Triplet Loss')
    plt.legend()
    plt.savefig('{:s}/ls-{:s}'.format(path, timestamp))


if __name__ == '__main__':
    plot_stats(sys.argv[1:])
