import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_distance(filename):
    if os.path.isdir(filename):
        path = filename
        for j in os.listdir(path):
            if 'test_result' in j:
                metric = np.load(path + '/' + j)
                name = j.split('.')[0]
                break
    else:
        path = '.'
        metric = np.load(filename)
        name = filename.split('.')[0]

    x = np.arange(metric.shape[0])
    cut = 100
    pos = metric[:, 0]
    neg = metric[:, 1:]
    mask = pos < neg.min()
    x1 = [i+cut for i, j in enumerate(mask) if j]
    x0 = [i+cut for i, j in enumerate(mask) if not j]
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


def plot_distance_imporovement(filename):
    if os.path.isdir(filename):
        path = filename
        for j in os.listdir(path):
            if 'test_result' in j:
                metric = np.load(path + '/' + j)
                name = j.split('.')[0]
                break
    else:
        path = '.'
        metric = np.load(filename)
        name = filename.split('.')[0]

    cut = 100
    baseline = np.load('baseline/baseline_result_eval_trans_test.npy')[cut:]
    baseline_rank = (baseline[:, :1] < baseline[:, 1:]).sum(1)
    metric_rank = (metric[:, :1] < metric[:, 1:]).sum(1)

    y = metric_rank - baseline_rank
    k = metric.shape[0]
    # 1: False->True, 2: True->False, 3: Increase 4: Decrease
    x1 = [i+cut for i in range(k) if metric_rank[i]==9 and y[i]>0]
    y1 = [y[i] for i in range(k) if metric_rank[i]==9 and y[i]>0]
    x2 = [i+cut for i in range(k) if baseline_rank[i]==9 and y[i]<0]
    y2 = [y[i] for i in range(k) if baseline_rank[i]==9 and y[i]<0]
    x3 = [i+cut for i in range(k) if metric_rank[i]!=9 and y[i]>0]
    y3 = [y[i] for i in range(k) if metric_rank[i]!=9 and y[i]>0]
    x4 = [i+cut for i in range(k) if baseline_rank[i]!=9 and y[i]<0]
    y4 = [y[i] for i in range(k) if baseline_rank[i]!=9 and y[i]<0]

    plt.bar(x1, y1, color='g', linewidth=0)
    plt.bar(x2, y2, color='r', linewidth=0)
    plt.bar(x3, y3, color='b', linewidth=0)
    plt.bar(x4, y4, color='m', linewidth=0)
    plt.xlabel('#Case')
    plt.ylabel('Rank')
    plt.savefig('{:s}/{:s}_bar'.format(path, name))


if __name__ == '__main__':
    for i in sys.argv[1:]:
        plot_distance(i)
        plot_distance_imporovement(i)