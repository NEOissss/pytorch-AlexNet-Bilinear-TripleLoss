import os
import sys
import shutil
from cnn import AlexManager


def main(test=True, ver=0, weight='official', net=net):
    root = '/mnt/nfs/scratch1/gluo/SUN360/HalfHalf/'
    if test:
        dataset = 'test'
    else:
        dataset = 'train'
    data_opts = {'train': {'set': 'train', 'cut': [None, None]},
                 'test': {'set': dataset, 'cut': [None, None]},
                 'val': {'set': 'test', 'cut': [None, None]},
                 'ver': ver}

    if not os.path.exists('baseline'):
        os.mkdir('baseline')

    print('Baseline L2 results:')
    alex = AlexManager(root=root, data_opts=data_opts, batch=128, net=net, weight=weight)
    path, accu = alex.test()
    new_path = 'baseline/baseline_v{:d}_{:s}_{:s}_{:s}_{:.1f}.npy'.format(ver, dataset, net, weight, accu)
    shutil.move(path, new_path)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        main(ver=int(sys.argv[1]), net=sys.argv[2], weight=sys.argv[3])
    else:
        print('Usage: python {:s} Ver[0|1|2] Net[Triplet|TripletConv5] Weight[official|places365].'.format(sys.argv[0]))
