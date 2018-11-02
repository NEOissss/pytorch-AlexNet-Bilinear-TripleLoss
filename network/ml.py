import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from SUN360Dataset import Sun360Dataset


class MetricTrplet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        weight = torch.zeros(1, 4096, requires_grad=True)
        bias = torch.zeros(4096, requires_grad=True)
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        torch.nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        x = self.weight * x + self.bias
        return x


class FullMetricTriplet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        weight = torch.zeros(4096, 4096, requires_grad=True)
        bias = torch.zeros(4096, requires_grad=True)
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        torch.nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        x = x.matmul(self.weight).mul(x) + self.bias
        return x


class MetricTripletManager(object):
    def __init__(self, root, data_opts, val=True, batch=1, lr=1e-3, decay=0, margin=1.0, param_path=None, net='Metric'):
        if net == 'FullMetric':
            self._net = torch.nn.DataParallel(FullMetricTriplet()).cuda()
        elif net == 'Metric':
            self._net = torch.nn.DataParallel(MetricTrplet()).cuda()
        else:
            raise ValueError('Unavailable net option.')
        # print(self._net)

        # Load pre-trained parameters
        if param_path:
            self._load(param_path)

        self._batch = batch
        self._val = val
        self._net_name = net
        self._stats = []
        self._criterion = torch.nn.TripletMarginLoss(margin=margin).cuda()
        self._solver = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=decay).cuda()

        # Load data
        self.data_opts = data_opts
        self.train_data_loader, self.test_data_loader, self.val_data_loader = self._data_loader(root=root)

    def train(self, epoch=1, verbose=None):
        """Train the network."""
        print('Training.')
        self._stats = []
        best_iter = [0, 0, 0, 0, 0]
        for t in range(epoch):
            print("\nEpoch: {:d}".format(t+1))
            iter_num = 0
            for data in iter(self.train_data_loader):
                # Data.
                data = data.reshape(-1, 4096)
                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                feat = self._net(data)
                dist_p = ((feat[0::3, :]-feat[1::3, :])**2).sum(1)
                dist_n = ((feat[0::3, :]-feat[2::3, :])**2).sum(1)
                accu = torch.sum(dist_p < dist_n).item()/(feat.size(0)/3)
                loss = self._criterion(feat[0::3, :], feat[1::3, :], feat[2::3, :])
                # Backward pass.
                loss.backward()
                self._solver.step()
                iter_num += 1

                if self._val:
                    val_accu = self.test(val=True)
                    if val_accu > best_iter[3]:
                        best_iter = [t+1, iter_num, accu, val_accu, loss.item()]
                    self._stats.append([t+1, iter_num, accu, val_accu, loss.item()])
                else:
                    if accu > best_iter[2]:
                        best_iter = [t+1, iter_num, accu, loss.item()]
                    self._stats.append([t+1, iter_num, accu, 0, loss.item()])

                if verbose and iter_num % verbose == 0:
                    print('Batch: {:d}, Triplet loss: {:.4f}, Batch accuracy: {:.2f}, Valid accuracy: {:.2f}'.format(
                        iter_num, loss.item(), self._stats[-1][2], self._stats[-1][3]))

        self._stats = np.array(self._stats)
        print('\nBest iteration stats: ' + str(best_iter) + '\n')
        return self._save()

    def test(self, param_path=None, val=False):
        if param_path:
            self._load(param_path)
        if val:
            data_loader = self.val_data_loader
        else:
            data_loader = self.test_data_loader
            print('Testing.')

        self._net.eval()
        dist_mat = np.zeros((len(data_loader.dataset), 10))
        batch = self._batch // 4

        for i, data in enumerate(data_loader):
            data = data.reshape(-1, 4096)
            feat = self._net(data)
            feat = feat.reshape(feat.size(0)//11, 11, -1)
            dist_p = torch.sqrt(((feat[:, 0, :] - feat[:, 1, :])**2).sum(1))
            dist_n = torch.sqrt(((feat[:, :1, :] - feat[:, 2:, :])**2).sum(2))
            dist_mat[i*batch:min((i+1)*batch, dist_mat.shape[0]), 0] = dist_p.cpu().detach().numpy()
            dist_mat[i*batch:min((i+1)*batch, dist_mat.shape[0]), 1:] = dist_n.cpu().detach().numpy()

        num_correct = np.sum(np.sum(dist_mat[:, 1:] > dist_mat[:, :1], axis=1) == 9)
        num_total = dist_mat.shape[0]

        if not val:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            np.save('test_result_' + timestamp, dist_mat)
            print('Test accuracy saved: test_result_' + timestamp + '.npy')
            print('Test accuracy: {:f}'.format(num_correct/num_total))
        self._net.train()
        return num_correct/num_total

    def _data_loader(self, root):
        train_data = self.data_opts['train']['set']
        test_data = self.data_opts['test']['set']
        val_data = self.data_opts['val']['set']
        train_cut = self.data_opts['train']['cut']
        test_cut = self.data_opts['test']['cut']
        val_cut = self.data_opts['val']['cut']
        ver = self.data_opts['ver']
        print('Train dataset: {:s}, test dataset: {:s}{:s}, val dataset: {:s}{:s}'
              .format(train_data, test_data, str(test_cut), val_data, str(val_cut)))
        train_dataset = Sun360Dataset(root=root, train=True, dataset=train_data, cut=train_cut, opt='fc7', version=ver)
        test_dataset = Sun360Dataset(root=root, train=False, dataset=test_data, cut=test_cut, opt='fc7', version=ver)
        val_dataset = Sun360Dataset(root=root, train=False, dataset=val_data, cut=val_cut, opt='fc7', version=ver)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self._batch, shuffle=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self._batch//4)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=self._batch//4)
        return train_data_loader, test_data_loader, val_data_loader

    def _save(self):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        path = self._net_name + '-param-' + timestamp
        path_stats = 'train_stats_' + timestamp
        torch.save(self._net.state_dict(), path)
        np.save(path_stats, self._stats)
        print('Model parameters saved: ' + path)
        print('Training stats saved: ' + path_stats + '.npy\n')
        return path

    def _load(self, path):
        self._net.load_state_dict(torch.load(path))
        print('Model parameters loaded: ' + path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', dest='net', type=str, default='Metric', help='Choose the network.')
    parser.add_argument('--param', dest='param', type=str, default=None, help='Initialize model parameters.')

    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Base learning rate for training.')
    parser.add_argument('--decay', dest='decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--margin', dest='margin', type=float, default=5.0, help='Margin for triplet loss.')

    parser.add_argument('--batch', dest='batch', type=int, default=256, help='Batch size.')
    parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='Epochs for training.')
    parser.add_argument('--verbose', dest='verbose', type=int, default=1, help='Printing frequency setting.')

    parser.add_argument('--valid', dest='valid', action='store_true', help=' Use validation.')
    parser.add_argument('--no-valid', dest='valid', action='store_false', help='Do not use validation.')
    parser.set_defaults(valid=True)

    args = parser.parse_args()

    if args.net not in ['Metric', 'FullMetric']:
        raise AttributeError('--net parameter must be \'Metric\' or \'FullMetric\'.')
    if args.lr <= 0:
        raise AttributeError('--lr parameter must > 0.')
    if args.decay < 0:
        raise AttributeError('--decay parameter must >= 0.')
    if args.margin <= 0:
        raise AttributeError('--margin parameter must > 0.')
    if args.batch <= 0:
        raise AttributeError('--batch parameter must > 0.')
    if args.epoch <= 0:
        raise AttributeError('--epoch parameter must > 0.')

    root = '/mnt/nfs/scratch1/gluo/SUN360/HalfHalf/'
    data_opts = {'train': {'set': 'train', 'cut': [None, None]},
                 'test': {'set': 'test', 'cut': [100, None]},
                 'val': {'set': 'test', 'cut': [0, 100]},
                 'ver': 0}

    print('====Exp details====')
    print('Net: ' + args.net)
    print('Margin: {:.1f}'.format(args.margin))
    print('Validation: ' + str(args.valid))
    print('Pretrained parameters: ' + str(args.param))
    print('#Epoch: {:d}, #Batch: {:d}'.format(args.epoch, args.batch))
    print('Learning rate: {:.0e}'.format(args.lr))
    print('Weight decay: {:.0e}\n'.format(args.decay))

    ml = MetricTripletManager(root=root, data_opts=data_opts, net=args.net, val=args.valid, margin=args.margin,
                              lr=args.lr, decay=args.decay, batch=args.batch, param_path=args.param)
    ml.train(epoch=args.epoch, verbose=args.verbose)
    ml.test()


if __name__ == '__main__':
    main()
