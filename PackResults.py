import os
import shutil
from Utils import analyze_log


def pack_results(path):
    files = os.listdir(path)
    for fname in files:
        if 'slurm-' in fname:
            res = analyze_log(fname)
            # Net-Freeze-Margin-Epoch-Batch-LR-k-Accu
            for k in range(100):
                dir_path = '{:s}-{:s}-{:s}-{:s}-{:s}-{:s}-{:s}-{:s}'.format(
                    res['net'], res['freeze'], res['margin'], res['epoch'],
                    res['batch'], res['lr'], str(k), res['test_accu'])
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                    break
            shutil.move(fname, '{:s}/{:s}'.format(dir_path, fname))
            if res['param']:
                shutil.move(res['param'], '{:s}/{:s}'.format(dir_path, res['param']))
            if res['train']:
                shutil.move(res['train'], '{:s}/{:s}'.format(dir_path, res['train']))
            if res['test']:
                shutil.move(res['test'], '{:s}/{:s}'.format(dir_path, res['test']))


if __name__ == '__main__':
    pack_results('./')
