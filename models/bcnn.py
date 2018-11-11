import os
from itertools import chain
import argparse
from datetime import datetime
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from SUN360Dataset import Sun360Dataset


# BACKBONE
def get_alexnet(pretrained='official'):
    if pretrained == 'official':
        return models.alexnet(pretrained=True)
    elif pretrained == 'places365':
        model_file = 'alexnet_places365.pth.tar'
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)
        model = models.alexnet(num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        return model
    else:
        raise ValueError('Unknown pretrained model!')


class AlexFC7(torch.nn.Module):
    def __init__(self, freeze=False, pretrained='official'):
        super(AlexFC7, self).__init__()
        alexnet = get_alexnet(pretrained=pretrained)
        self.features = alexnet.features
        self.fc = alexnet.classifier[:-2]
        if freeze:
            self._freeze()

    def forward(self, x):
        x = x.float()
        n = x.size()[0]
        assert x.size() == (n, 3, 227, 227)
        x = self.features(x)
        x = x.view(n, 256 * 6 * 6)
        x = self.fc(x)
        assert x.size() == (n, 4096)
        return x

    def _freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = False


class AlexConv5(torch.nn.Module):
    def __init__(self, freeze=False, pretrained='official'):
        super(AlexConv5, self).__init__()
        alexnet = get_alexnet(pretrained=pretrained)
        self.features = alexnet.features
        if freeze:
            self._freeze()

    def forward(self, x):
        x = x.float()
        n = x.size()[0]
        assert x.size() == (n, 3, 227, 227)
        x = self.features(x)
        x = x.view(n, 256, -1)
        x = x.mean(2)
        assert x.size() == (n, 256)
        return x

    def _freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False


# Metric Learning
class NoneMetric(torch.nn.Module):
    def __init__(self):
        super(NoneMetric, self).__init__()

    def forward(self, x):
        return x


class DiagonalMetric(torch.nn.Module):
    def __init__(self, in_dim=4096):
        super(DiagonalMetric, self).__init__()
        weight = torch.zeros(1, in_dim, requires_grad=True)
        bias = torch.zeros(in_dim, requires_grad=True)
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        torch.nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        return self.weight * x + self.bias


class SymmetricMetric(torch.nn.Module):
    def __init__(self, in_dim=4096):
        super(SymmetricMetric, self).__init__()
        self.fc = torch.nn.Linear(in_dim, in_dim)
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        torch.nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        return self.fc(x)


class BilinearMetric(torch.nn.Module):
    def __init__(self, in_dim=4096, bi_in=256, bi_out=1):
        super(BilinearMetric, self).__init__()
        self.fc = torch.nn.Linear(in_dim, bi_in)
        self.bfc = torch.nn.Bilinear(bi_in, bi_in, bi_out)
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        torch.nn.init.kaiming_normal_(self.bfc.weight.data)
        torch.nn.init.constant_(self.fc.bias.data, val=0)
        torch.nn.init.constant_(self.bfc.bias.data, val=0)

    def forward(self, x):
        return self.fc(x)


# Loss Function
class TripletMarginLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.dist_p = None
        self.dist_n = None

    def forward(self, a, p, n):
        self.dist_p = torch.sqrt(((a - p) ** 2).sum(1))
        self.dist_n = torch.sqrt(((a - n) ** 2).sum(1))
        loss = torch.mean(((self.dist_p - self.dist_n) + self.margin).clamp(min=0))
        return loss

    def test(self, a, p, n):
        self.eval()
        self.dist_p = torch.sqrt(((a - p) ** 2).sum(1))
        self.dist_n = torch.sqrt(((torch.unsqueeze(a, 1) - n) ** 2).sum(2))
        self.train()
        return self.dist_p, self.dist_n

    def get_batch_accuracy(self):
        return torch.sum(self.dist_p < self.dist_n).item() / self.dist_p.size(0)


class BilinearTripletMarginLoss(torch.nn.Module):
    def __init__(self, bfc, margin=1.0):
        super(BilinearTripletMarginLoss, self).__init__()
        self.bfc = bfc
        self.margin = margin
        self.dist_p = None
        self.dist_n = None

    def forward(self, a, p, n):
        self.dist_p = self.bfc(a, p).max(1)[0]
        self.dist_n = self.bfc(a, n).max(1)[0]
        loss = torch.mean(((self.dist_p - self.dist_n) + self.margin).clamp(min=0))
        return loss

    def test(self, a, p, n):
        self.eval()
        self.dist_p = self.bfc(a, p).max(1)[0]
        exp_a = a.unsqueeze(1).expand(a.size(0), n.size(1), a.size(1)).contiguous()
        exp_n = n.contiguous()
        self.dist_n = self.bfc(exp_a, exp_n).max(2)[0]
        self.train()
        return self.dist_p, self.dist_n

    def get_batch_accuracy(self):
        return torch.sum(self.dist_p < self.dist_n).item() / self.dist_p.size(0)


# Main Process
class NetworkManager(object):
    def __init__(self, root, data_opts, net='AlexFC7', metric='None', dim=1, weight='official', freeze=False, flip=True,
                 val=True, batch=1, lr=1e-3, decay=0, margin=5.0, net_param=None, metric_param=None, choice=10):
        if net == 'AlexFC7':
            self._net = torch.nn.DataParallel(AlexFC7(freeze=freeze, pretrained=weight)).cuda()
            net_out_dim = 4096
        elif net == 'AlexConv5':
            self._net = torch.nn.DataParallel(AlexConv5(freeze=freeze, pretrained=weight)).cuda()
            net_out_dim = 256
        else:
            raise ValueError('Unavailable metric option.')

        if metric == 'None':
            self._metric = torch.nn.DataParallel(NoneMetric()).cuda()
            self._criterion = TripletMarginLoss(margin=margin).cuda()
        elif metric == 'Diagonal':
            self._metric = torch.nn.DataParallel(DiagonalMetric(in_dim=net_out_dim)).cuda()
            self._criterion = TripletMarginLoss(margin=margin).cuda()
        elif metric == 'Symmetric':
            self._metric = torch.nn.DataParallel(SymmetricMetric(in_dim=net_out_dim)).cuda()
            self._criterion = TripletMarginLoss(margin=margin).cuda()
        elif metric == 'Bilinear':
            self._metric = torch.nn.DataParallel(BilinearMetric(in_dim=net_out_dim, bi_out=dim)).cuda()
            self._criterion = BilinearTripletMarginLoss(bfc=self._metric.module.bfc, margin=margin).cuda()
        else:
            raise ValueError('Unavailable metric option.')
        # print(self._net)
        # print(self._metric)

        # define the total #choice
        self._n = choice

        # Load pre-trained parameters
        if net_param:
            self._load_net_param(net_param)
        if metric_param:
            self._load_metric_param(metric_param)

        self._flip = flip
        self._batch = batch
        self._val = val
        self._net_name = net
        self._metric_name = metric
        self._stats = []

        requires_grad = filter(lambda p: p.requires_grad, chain(self._net.parameters(), self._metric.parameters()))
        self._solver = torch.optim.Adam(requires_grad, lr=lr, weight_decay=decay)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._solver, mode='max', verbose=True)

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
                data = data.reshape(-1, 3, 227, 227)
                if self._flip:
                    idx = torch.randperm(data.size(0))[:data.size(0)//2]
                    data[idx] = data[idx].flip(3)
                self._solver.zero_grad()
                feat = self._metric(self._net(data))
                loss = self._criterion(feat[0::3, :], feat[1::3, :], feat[2::3, :])
                accu = self._criterion.get_batch_accuracy()
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

            if self._val:
                self._scheduler.step(val_accu)

        self._stats = np.array(self._stats)
        print('\nBest iteration stats: ' + str(best_iter) + '\n')
        return self._save()

    def test(self, val=False):
        if val:
            data_loader = self.val_data_loader
        else:
            data_loader = self.test_data_loader
            print('Testing.')

        self._net.eval()
        dist_mat = np.zeros((len(data_loader.dataset), self._n))
        batch = self._batch // 4

        for i, data in enumerate(data_loader):
            data = data.reshape(-1, 3, 227, 227)
            feat = self._metric(self._net(data))
            feat = feat.reshape(feat.size(0)//(self._n+1), self._n+1, -1)
            dist_p, dist_n = self._criterion.test(feat[:, 0, :], feat[:, 1, :], feat[:, 2:, :])
            dist_mat[i*batch:min((i+1)*batch, dist_mat.shape[0]), 0] = dist_p.cpu().detach().numpy()
            dist_mat[i*batch:min((i+1)*batch, dist_mat.shape[0]), 1:] = dist_n.cpu().detach().numpy()

        num_correct = np.sum(np.sum(dist_mat[:, 1:] > dist_mat[:, :1], axis=1) == self._n - 1)
        num_total = dist_mat.shape[0]
        self._net.train()

        if not val:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            np.save('test_result_' + timestamp, dist_mat)
            print('Test result saved: test_result_' + timestamp + '.npy')
            print('Test accuracy: {:f}'.format(num_correct/num_total))
            return 'test_result_' + timestamp + '.npy', num_correct/num_total
        else:
            return num_correct/num_total

    def _data_loader(self, root):
        train_data = self.data_opts['train']['set']
        test_data = self.data_opts['test']['set']
        val_data = self.data_opts['val']['set']
        train_cut = self.data_opts['train']['cut']
        test_cut = self.data_opts['test']['cut']
        val_cut = self.data_opts['val']['cut']
        ver = self.data_opts['ver']
        print('Train dataset: {:s}{:s}, test dataset: {:s}{:s}, val dataset: {:s}{:s}'
              .format(train_data, str(train_cut), test_data, str(test_cut), val_data, str(val_cut)))
        train_dataset = Sun360Dataset(root=root, train=True, dataset=train_data, cut=train_cut, version=ver)
        test_dataset = Sun360Dataset(root=root, train=False, dataset=test_data, cut=test_cut, version=ver)
        val_dataset = Sun360Dataset(root=root, train=False, dataset=val_data, cut=val_cut, version=ver)
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=self._batch, shuffle=True)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self._batch//4)
        val_data_loader = DataLoader(dataset=val_dataset, batch_size=self._batch//4)
        return train_data_loader, test_data_loader, val_data_loader

    def _save(self):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        net_path = self._net_name + '-param-' + timestamp
        metric_path = self._metric_name + '-param-' + timestamp
        path_stats = 'train_stats_' + timestamp
        torch.save(self._net.state_dict(), net_path)
        torch.save(self._metric.state_dict(), metric_path)
        np.save(path_stats, self._stats)
        print('Net model parameters saved: ' + net_path)
        print('Metric model parameters saved: ' + metric_path)
        print('Training stats saved: ' + path_stats + '.npy\n')
        return net_path, metric_path

    def _load_net_param(self, path):
        self._net.load_state_dict(torch.load(path))
        print('Net model parameters loaded: ' + path)

    def _load_metric_param(self, path):
        self._metric.load_state_dict(torch.load(path))
        print('Metric model parameters loaded: ' + path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', dest='net', type=str, default='AlexFC7', help='Choose the net(backbone network).')
    parser.add_argument('--metric', dest='metric', type=str, default='None', help='Choose the metric learning method.')
    parser.add_argument('--dim', dest='dim', type=int, default=1, help='Define bilinear out dimension.')
    parser.add_argument('--weight', dest='weight', type=str, default='official', help='Choose pretrained net model.')
    parser.add_argument('--n_param', dest='n_param', type=str, default=None, help='Initial net parameters.')
    parser.add_argument('--m_param', dest='m_param', type=str, default=None, help='Initial metric parameters.')
    parser.add_argument('--version', dest='version', type=int, default=0, help='Dataset version.')

    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Base learning rate for training.')
    parser.add_argument('--decay', dest='decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--margin', dest='margin', type=float, default=5.0, help='Margin for triplet loss.')

    parser.add_argument('--batch', dest='batch', type=int, default=256, help='Batch size.')
    parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='Epochs for training.')
    parser.add_argument('--verbose', dest='verbose', type=int, default=1, help='Printing frequency setting.')

    parser.add_argument('--freeze', dest='freeze', action='store_true', help='Choose freeze mode.')
    parser.add_argument('--no-freeze', dest='freeze', action='store_false', help='Choose non-freeze mode.')
    parser.set_defaults(freeze=True)

    parser.add_argument('--valid', dest='valid', action='store_true', help='Use validation.')
    parser.add_argument('--no-valid', dest='valid', action='store_false', help='Do not use validation.')
    parser.set_defaults(valid=True)

    args = parser.parse_args()

    if args.net not in ['AlexFC7', 'AlexConv5']:
        raise AttributeError('--net parameter must be \'AlexFC7\' or \'AlexConv5\'.')
    if args.metric not in ['None', 'Diagonal', 'Symmetric', 'Bilinear']:
        raise AttributeError('--metric parameter must be \'None\', \'Diagonal\', \'Symmetric\', \'Bilinear\'.')
    if args.weight not in ['official', 'places365']:
        raise AttributeError('--weight parameter must be \'official\' or \'places365\'')
    if args.version not in [0, 1, 2]:
        raise AttributeError('--version parameter must be in [0, 1, 2]')
    if args.dim <= 0:
        raise AttributeError('--lr parameter must > 0.')
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
    data_opts = {'train': {'set': 'train', 'cut': [0, 8000]},
                 'test': {'set': 'test', 'cut': [None, None]},
                 'val': {'set': 'train', 'cut': [8000, None]},
                 'ver': args.version}

    print('====Exp details====')
    print('Net: {:s}'.format(args.net))
    print('Metric: {:s}'.format(args.metric))
    if args.metric == 'Bilinear':
        print('Bilinear out dimension: {:d}'.format(args.dim))
    print('Pretrained net: ' + str(args.weight))
    print('Benchmark ver: {:d}'.format(args.version))
    print('Validation: ' + str(args.valid))
    print('Net parameters: ' + str(args.n_param))
    print('Metric parameters: ' + str(args.m_param))
    print('Freeze mode: ' + str(args.freeze))
    print('Margin: {:.1f}'.format(args.margin))
    print('#Epoch: {:d}, #Batch: {:d}'.format(args.epoch, args.batch))
    print('Learning rate: {:.0e}'.format(args.lr))
    print('Weight decay: {:.0e}'.format(args.decay))
    print('Learning rate scheduler used!\n')

    cnn = NetworkManager(root=root, data_opts=data_opts, net=args.net, metric=args.metric, dim=args.dim,
                         weight=args.weight, freeze=args.freeze, net_param=args.n_param, metric_param=args.m_param,
                         margin=args.margin, lr=args.lr, decay=args.decay, batch=args.batch, val=args.valid)
    cnn.train(epoch=args.epoch, verbose=args.verbose)
    cnn.test()


if __name__ == '__main__':
    main()
