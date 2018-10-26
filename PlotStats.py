import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(log_files):
    labels = []
    stats = []
    for i in log_files:
        x, y = analyze_log(i)
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


def analyze_log(filename):
    with open(filename, 'r') as fp:
        for line in fp:
            match_file = re.search(r'train_stats_\d+.npy', line)
            match_batch = re.search(r'Batch:\s\d+', line)
            match_lr = re.search(r'rate:\s\d+.\d+', line)
            match_margin = re.search(r'Margin:\s\d+.\d+', line)
    label = '{:s}-{:s}-{:s}'.format(match_margin.group().split()[-1],
                                    match_batch.group().split()[-1],
                                    match_lr.group().split()[-1])
    return label, match_file.group()


if __name__ == '__main__':
    plot_stats(['slurm-4272548.out '])
