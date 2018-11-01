import os
import shutil
from cnn import AlexManager


def main(test=True):
    root = '/mnt/nfs/scratch1/gluo/SUN360/HalfHalf/'
    if test:
        dataset = 'test'
    else:
        dataset = 'train'
    data_opts = {'train': {'set': 'train', 'cut': [None, None]},
                 'test': {'set': dataset, 'cut': [None, None]},
                 'val': {'set': 'test', 'cut': [None, None]},
                 'ver': 0}

    if not os.path.exists('baseline'):
        os.mkdir('baseline')

    alex = AlexManager(root=root, data_opts=data_opts)
    path = alex.test()
    new_path = 'baseline/baseline_{:s}.npy'.format(dataset)
    shutil.move(path, new_path)
    print('Baseline L2 results for {:s} dataset saved at {:s}'.format(dataset, new_path))


if __name__ == '__main__':
    main()
