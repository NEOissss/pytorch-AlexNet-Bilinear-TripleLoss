import csv
import json
import numpy as np

def sun360h_data_load(task='train', data='train', ver=0, batch=1, cut=None):
    root_path = '/mnt/nfs/scratch1/gluo/SUN360/HalfHalf/'
    imgs_path = '/IMGs/'

    if data=='train' or data=='test':
        task_path = 'task_' + data
        gt_path = 'gt_' + data
    else:
        raise ValueError('Unavailable dataset part!')

    if ver==0:
        task_path += '/'
        gt_path += '.csv'
    elif ver==1:
        task_path += '_v1/'
        gt_path += '_v1.csv'
    elif ver==2:
        task_path += '_v2/'
        gt_path += '_v2.csv'
    else:
        raise ValueError('Unavailable dataset version!')

    with open(root_path + gt_path, 'r') as csv_file:
        if cut:
            gt_list = list(csv.reader(csv_file, delimiter=','))[cut[0]:cut[1]]
        else:
           gt_list = list(csv.reader(csv_file, delimiter=','))
        gt_len = len(gt_list)

    result = []
    idx = np.random.permutation(gt_len)

    if task == 'train':
        for i in range(0, gt_len, batch):
            a_bacth, p_batch, n_batch = [], [], []
            for j in idx[i:min(i+batch, gt_len)]:
                with open(root_path + task_path + gt_list[j][0] + '.json', 'r') as f:
                    names = json.load(f)
                    a_bacth.append(root_path + imgs_path + names[0])
                    p_batch.append(root_path + imgs_path + names[1][int(gt_list[j][1])])
                    n_batch.append(root_path + imgs_path + names[1][[k for k in range(10) if k!=int(gt_list[j][1])][np.random.randint(9)]])
            result.append([a_bacth, p_batch, n_batch])

    if task == 'test':
        for i in range(0, gt_len):
            a_bacth, p_batch, n_batch = [], [], []
            for j in idx[i:i+1]:
                with open(root_path + task_path + gt_list[j][0] + '.json', 'r') as f:
                    names = json.load(f)
                    a_bacth.append(root_path + imgs_path + names[0])
                    p_batch.append(root_path + imgs_path + names[1][int(gt_list[j][1])])
                    for k in range(10):
                        if k != int(gt_list[j][1]):
                            n_batch.append(root_path + imgs_path + names[1][k])
            result.append([a_bacth, p_batch, n_batch])
    return result
