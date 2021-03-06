import sys
import re
import os
import shutil
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def analyze_log(filename):
    with open(filename, 'r') as fp:
        content = fp.read()
        match_n_param = re.search(r'Net\smodel\sparameters\ssaved:.+', content)
        match_m_param = re.search(r'Metric\smodel\sparameters\ssaved:.+', content)
        match_train = re.search(r'train_stats_\d+\.npy', content)
        match_test = re.search(r'test_result_\d+\.npy', content)
        match_test_accu = re.search(r'Test\saccuracy:.+', content)
        match_net = re.search(r'Net:.+', content)
        match_metric = re.search(r'Metric:.+', content)
        match_ver = re.search(r'Benchmark\sver:.+', content)
        match_weight = re.search(r'Pretrained\snet:.+', content)
        match_margin = re.search(r'Margin:\s\d+\.\d+', content)
        match_freeze = re.search(r'Freeze\smode:.+', content)
        match_epoch = re.search(r'#Epoch:\s\d+', content)
        match_batch = re.search(r'#Batch:\s\d+', content)
        match_lr = re.search(r'rate:.+', content)
        match_decay = re.search(r'decay:.+', content)
        match_dim = re.search(r'dimension:.+', content)

    try:
        res = {'net': match_net.group().split()[-1],
               'metric': match_metric.group().split()[-1],
               'dim': match_dim.group().split()[-1] if match_dim else '1',
               'weight': match_weight.group().split()[-1] if match_weight else 'official',
               'ver': match_ver.group().split()[-1] if match_ver else '0',
               'freeze': match_freeze.group().split()[-1] if match_freeze else 'None',
               'margin': match_margin.group().split()[-1],
               'epoch': match_epoch.group().split()[-1],
               'batch': match_batch.group().split()[-1],
               'lr': match_lr.group().split()[-1],
               'decay': match_decay.group().split()[-1] if match_decay else 0,
               'test_accu': match_test_accu.group().split()[-1][:5],
               'n_param': match_n_param.group().split()[-1] if match_n_param else None,
               'm_param': match_m_param.group().split()[-1] if match_m_param else None,
               'train': match_train.group() if match_train else None,
               'test': match_test.group() if match_test else None
               }
        return res
    except AttributeError:
        return None


def pack_results(path='./'):
    files = os.listdir(path)
    dir_list = []
    for fname in files:
        if 'slurm-' in fname:
            res = analyze_log(fname)
            if not res:
                print('Unknown log content: {:s}'.format(fname))
                continue
            # Version|Net|Metric|Dim|Weight|Freeze|Margin|Epoch|Batch|LR|Decay|Accuracy
            dir_path = 'V{:s}|{:s}|{:s}|{:s}|{:s}|{:s}|{:s}|{:s}|{:s}|{:.0e}|{:.0e}|{:s}'.format(
                        res['ver'], res['net'], res['metric'], res['dim'], res['weight'], res['freeze'], res['margin'],
                        res['epoch'], res['batch'], float(res['lr']), float(res['decay']), res['test_accu'])
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
                dir_list.append([dir_path, int(res['ver']), res['net'], res['weight']])
            else:
                print(dir_path + ' already exists!')
                continue

            shutil.move(fname, '{:s}/{:s}'.format(dir_path, fname))
            if res['n_param'] and os.path.exists(res['n_param']):
                shutil.move(res['n_param'], '{:s}/{:s}'.format(dir_path, res['n_param']))
            if res['m_param'] and os.path.exists(res['m_param']):
                shutil.move(res['m_param'], '{:s}/{:s}'.format(dir_path, res['m_param']))
            if res['train'] and os.path.exists(res['train']):
                shutil.move(res['train'], '{:s}/{:s}'.format(dir_path, res['train']))
            if res['test'] and os.path.exists(res['test']):
                shutil.move(res['test'], '{:s}/{:s}'.format(dir_path, res['test']))

    return dir_list


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

        x = '{:s}-{:s}-{:s}-{:s}-{:s}-{:s}-{:s}-{:.0e}-{:.0e}'.format(
            res['net'], res['metric'], res['dim'], res['weight'], res['freeze'], res['margin'],
            res['batch'], float(res['lr']), float(res['decay']))
        labels.append(x)
        stats.append(np.load(y))

    path = 'plot'
    if not os.path.exists(path):
        os.mkdir(path)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Batch accuracy
    plt.clf()
    for i, stat in enumerate(stats):
        x = np.arange(3, stat.shape[0])
        y = stat[3:, 2]
        plt.plot(x, y, label=labels[i])
    plt.xlabel('Number of Update')
    plt.ylabel('Batch Accuracy')
    plt.legend()
    plt.savefig('{:s}/ba-{:s}'.format(path, timestamp))

    # Val accuracy
    plt.clf()
    for i, stat in enumerate(stats):
        x = np.arange(3, stat.shape[0])
        y = stat[3:, 3]
        plt.plot(x, y, label=labels[i])
    plt.xlabel('Number of Update')
    plt.ylabel('Val Accuracy')
    plt.legend()
    plt.savefig('{:s}/va-{:s}'.format(path, timestamp))

    # Triplet loss
    plt.clf()
    for i, stat in enumerate(stats):
        x = np.arange(3, stat.shape[0])
        y = stat[3:, 4]
        plt.plot(x, y, label=labels[i])
    plt.xlabel('Number of Update')
    plt.ylabel('Triplet Loss')
    plt.legend()
    plt.savefig('{:s}/ls-{:s}'.format(path, timestamp))


def plot_distance(filename):
    if os.path.isdir(filename):
        metric = None
        path = filename
        for j in os.listdir(path):
            if 'test_result' in j and '.npy' in j:
                metric = np.load(path + '/' + j)
                name = j.split('.')[0]
                break
        if metric is None:
            return
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


def plot_distance_improvement(filename, ver=0, net='AlexFC7', weight='official'):
    if os.path.isdir(filename):
        metric = None
        path = filename
        for j in os.listdir(path):
            if 'test_result' in j and '.npy' in j:
                metric = np.load(path + '/' + j)
                name = j.split('.')[0]
                break
        if metric is None:
            return
    else:
        path = '.'
        metric = np.load(filename)
        name = filename.split('.')[0]

    r = re.compile(r'baseline_v{:d}_test_{:s}_{:s}.+\.npy'.format(ver, net, weight))
    for npy_name in os.listdir('baseline'):
        if r.match(npy_name):
            break
    baseline = np.load('baseline/' + npy_name)
    baseline_rank = (baseline[:, :1] < baseline[:, 1:]).sum(1)
    metric_rank = (metric[:, :1] < metric[:, 1:]).sum(1)

    y = metric_rank - baseline_rank
    k = metric.shape[0]
    j = 0
    # 1: False->True, 2: Increase, 3: Decrease, 4: True->False
    y1 = sorted([y[i] for i in range(k) if metric_rank[i] == 9 and y[i] > 0], reverse=True)
    x1 = list(range(j, j + len(y1)))
    j += len(y1)

    y2 = sorted([y[i] for i in range(k) if metric_rank[i] != 9 and y[i] > 0], reverse=True)
    x2 = list(range(j, j + len(y2)))
    j += len(y2)

    y3 = sorted([y[i] for i in range(k) if baseline_rank[i] != 9 and y[i] < 0], reverse=True)
    x3 = list(range(j, j + len(y3)))
    j += len(y3)

    y4 = sorted([y[i] for i in range(k) if baseline_rank[i] == 9 and y[i] < 0], reverse=True)
    x4 = list(range(j, j + len(y4)))

    plt.clf()
    plt.bar(x1, y1, color='g', linewidth=0)
    plt.bar(x2, y2, color='b', linewidth=0)
    plt.bar(x3, y3, color='m', linewidth=0)
    plt.bar(x4, y4, color='r', linewidth=0)
    plt.xlabel('Case')
    plt.ylabel('Rank Change')
    plt.savefig('{:s}/{:s}_bar'.format(path, name))

    x_pos = [i for i in range(k) if metric_rank[i] == 9 and y[i] > 0]
    y_pos = [y[i] for i in range(k) if metric_rank[i] == 9 and y[i] > 0]

    x_neg = [i for i in range(k) if baseline_rank[i] == 9 and y[i] < 0]
    y_neg = [y[i] for i in range(k) if baseline_rank[i] == 9 and y[i] < 0]

    positive_list, negative_list = [], []
    for i in np.array(y_pos).argsort()[::-1]:
        positive_list.append([x_pos[i], baseline[x_pos[i]].argmin(), y_pos[i]])
    for i in np.array(y_neg).argsort():
        negative_list.append([x_neg[i], metric[x_neg[i]].argmin(), y_neg[i]])

    with open(path + '/positive_list.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(positive_list)

    with open(path + '/negative_list.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(negative_list)


# Pack and plots
def opt_all(path='./'):
    for dir_path, ver, net, weight in pack_results(path):
        plot_distance(dir_path)
        plot_distance_improvement(dir_path, ver, net, weight)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Usage: python utils.py pack [path] OR python utils.py plot <dir1, dir2, ...>')
    else:
        if sys.argv[1] == 'pack':
            if len(sys.argv) == 2:
                opt_all()
            elif len(sys.argv) == 3:
                opt_all(sys.argv[2])
            else:
                print('Usage: python utils.py pack [path] OR python3 utils.py plot <dir1, dir2, ...>')
        elif sys.argv[1] == 'plot' and len(sys.argv) > 2:
            plot_stats(sys.argv[2:])
        else:
            print('Usage: python utils.py pack [path] OR python3 utils.py plot <dir1, dir2, ...>')
