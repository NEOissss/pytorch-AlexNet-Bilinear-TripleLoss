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
    def __init__(self, val=True, batch=1, lr=1e-3, margin=1.0, param_path=None, net='Metric'):
        if net == 'FullMetric':
            self._net = torch.nn.DataParallel(FullMetricTriplet()).cuda()
        elif net == 'Metric':
            self._net = torch.nn.DataParallel(MetricTrplet()).cuda()
        else:
            raise ValueError('Unavailable net option.')
        # Load pre-trained parameters
        if param_path:
            self._load(param_path)

        self._batch = batch
        self._val = val
        self._net_name = net
        self._stats = []
        # print(self._net)
        self._criterion = torch.nn.TripletMarginLoss(margin=margin).cuda()

        # If not test
        self._solver = torch.optim.Adam(self._net.parameters(), lr=lr)

        # Load data
        root = '/mnt/nfs/scratch1/gluo/SUN360/HalfHalf/'
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

        for i, data in enumerate(data_loader):
            data = data.reshape(-1, 4096)
            feat = self._net(data)
            feat = feat.reshape(feat.size(0)//11, 11, -1)
            dist_p = torch.sqrt(((feat[:, 0, :] - feat[:, 1, :])**2).sum(1))
            dist_n = torch.sqrt(((feat[:, :1, :] - feat[:, 2:, :])**2).sum(2))
            dist_mat[i*self._batch:min((i+1)*self._batch, dist_mat.shape[0]), 0] = dist_p.cpu().detach().numpy()
            dist_mat[i*self._batch:min((i+1)*self._batch, dist_mat.shape[0]), 1:] = dist_n.cpu().detach().numpy()

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
        train_data = 'train'
        test_data = 'test'
        val_data = 'test'
        test_cut = [100, None]
        val_cut = [0, 100]
        print('Train dataset: {:s}, test dataset: {:s}{:s}, val dataset: {:s}{:s}'
              .format(train_data, test_data, str(test_cut), val_data, str(val_cut)))
        train_dataset = Sun360Dataset(root=root, train=True, dataset=train_data, opt='fc7')
        test_dataset = Sun360Dataset(root=root, train=False, dataset=test_data, cut=test_cut, opt='fc7')
        val_dataset = Sun360Dataset(root=root, train=False, dataset=val_data, cut=val_cut, opt='fc7')
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self._batch, shuffle=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self._batch)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=self._batch)
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
    ini_param = None
    val = True
    batch_size = 128
    epoch_num = 1
    learning_rate = 0.001
    net_name = 'Metric'
    verbose = 2
    margin = 5

    print('\n====Exp details====')
    print('Net: ' + net_name)
    print('Margin: {:.1f}'.format(margin))
    print('Validation: ' + str(val))
    print('Pretrained parameters: ' + str(ini_param))
    print('Freeze mode: ' + str(freeze))
    print('#Epoch: {:d}, #Batch: {:d}'.format(epoch_num, batch_size))
    print('Learning rate: {:.4f}\n'.format(learning_rate))

    metric = MetricTripletManager(val=val, margin=margin, lr=learning_rate, batch=batch_size, param_path=ini_param, net=net_name)
    path = metric.train(epoch=epoch_num, verbose=verbose)
    metric.test(param_path=path)
    # bcnn.test()



if __name__ == '__main__':
    main()
