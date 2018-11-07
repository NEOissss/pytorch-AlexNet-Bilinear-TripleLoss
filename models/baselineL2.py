import os
import sys
import shutil
from cnn import AlexManager


def main(test=True, ver=0):
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

    alex = AlexManager(root=root, data_opts=data_opts, batch=128, net='TripletConv5')
    path = alex.test()
    new_path = 'baseline/baseline_{:s}_v{:d}.npy'.format(dataset, ver)
    shutil.move(path, new_path)
    print('Baseline L2 results for {:s} dataset version {:d} saved at {:s}'.format(dataset, ver, new_path))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(ver=int(sys.argv[1]))
    else:
        main()
