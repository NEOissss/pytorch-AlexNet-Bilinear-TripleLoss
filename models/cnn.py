import argparse
from datetime import datetime
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from SUN360Dataset import Sun360Dataset


class TripletAlexFC7(torch.nn.Module):
    def __init__(self, freeze=None):
        torch.nn.Module.__init__(self)
        self.features = models.alexnet(pretrained=True).features
        fc_list = list(models.alexnet(pretrained=True).classifier.children())[:-2]
        self.fc = torch.nn.Sequential(*fc_list)

        # Freeze layers.
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


class TripletAlexConv5(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.features = models.alexnet(pretrained=True).features

    def forward(self, x):
        x = x.float()
        n = x.size()[0]
        assert x.size() == (n, 3, 227, 227)
        x = self.features(x)
        x = x.view(n, 256, -1)
        x = x.mean(2)
        assert x.size() == (n, 256)
        return x


class BilinearTripletAlex(torch.nn.Module):
    def __init__(self, freeze=None, bi_in=256, bi_out=1):
        torch.nn.Module.__init__(self)
        self.bi_in = bi_in
        self.bi_out = bi_out
        self.features = models.alexnet(pretrained=True).features
        fc_list = list(models.alexnet(pretrained=True).classifier.children())[:-1]
        fc_list.append(torch.nn.Linear(4096, self.bi_in))
        self.fc = torch.nn.Sequential(*fc_list)

        self.bfc = torch.nn.Bilinear(self.bi_in, self.bi_in, self.bi_out)

        # Freeze layers.
        if freeze:
            self._freeze()

        # Initialize the last fc layers.
        torch.nn.init.kaiming_normal_(self.fc[-1].weight.data)
        torch.nn.init.kaiming_normal_(self.bfc.weight.data)
        if self.fc[-1].bias is not None:
            torch.nn.init.constant_(self.fc[-1].bias.data, val=0)
        if self.bfc.bias is not None:
            torch.nn.init.constant_(self.bfc.bias.data, val=0)

    def forward(self, x):
        x = x.float()
        n = x.size()[0]
        assert x.size() == (n, 3, 227, 227)
        x = self.features(x)
        x = x.view(n, 256 * 6 * 6)
        x = self.fc(x)
        assert x.size() == (n, self.bi_in)
        return x

    def _freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False


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
        self.dist_p = self.bfc(a, p).squeeze()
        self.dist_n = self.bfc(a, n).squeeze()
        loss = torch.mean(((self.dist_p - self.dist_n) + self.margin).clamp(min=0))
        return loss

    def test(self, a, p, n):
        self.eval()
        self.dist_p = self.bfc(a, p).squeeze()
        exp_a = a.unsqueeze(1).expand(a.size(0), n.size(1), a.size(1)).contiguous()
        exp_n = n.contiguous()
        self.dist_n = self.bfc(exp_a, exp_n).squeeze()
        self.train()
        return self.dist_p, self.dist_n

    def get_batch_accuracy(self):
        return torch.sum(self.dist_p < self.dist_n).item() / self.dist_p.size(0)


class AlexManager(object):
    def __init__(self, root, data_opts, freeze='part', val=True, batch=1, lr=1e-3, decay=0,
                 margin=1.0, param_path=None, net='Triplet', flip=True, matterport=False):
        if net == 'Bilinear':
            self._net = torch.nn.DataParallel(BilinearTripletAlex(freeze=freeze)).cuda()
            self._criterion = BilinearTripletMarginLoss(bfc=self._net.module.bfc, margin=margin).cuda()
            self._bilinear = True
        elif net == 'Triplet':
            self._net = torch.nn.DataParallel(TripletAlexFC7(freeze=freeze)).cuda()
            self._criterion = TripletMarginLoss(margin=margin).cuda()
            self._bilinear = False
        elif net == 'TripletConv5':
            self._net = torch.nn.DataParallel(TripletAlexConv5()).cuda()
            self._criterion = TripletMarginLoss(margin=margin).cuda()
            self._bilinear = False
        else:
            raise ValueError('Unavailable net option.')
        # print(self._net)

        # define the total #choice
        if matterport:
            self._n = 4
        else:
            self._n = 10

        # Load pre-trained parameters
        if param_path:
            self._load(param_path)

        self._flip = flip
        self._batch = batch
        self._val = val
        self._net_name = net
        self._stats = []
        self._solver = torch.optim.Adam(filter(lambda p: p.requires_grad, self._net.parameters()),
                                        lr=lr, weight_decay=decay)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._solver, mode='max', patience=3)

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
                    data = self._data_flip(data)
                self._solver.zero_grad()
                feat = self._net(data)
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
                    self._scheduler.step(val_accu)
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
        dist_mat = np.zeros((len(data_loader.dataset), self._n))
        batch = self._batch // 4

        for i, data in enumerate(data_loader):
            data = data.reshape(-1, 3, 227, 227)
            feat = self._net(data)
            feat = feat.reshape(feat.size(0)//(self._n+1), self._n+1, -1)
            dist_p, dist_n = self._criterion.test(feat[:, 0, :], feat[:, 1, :], feat[:, 2:, :])
            dist_mat[i*batch:min((i+1)*batch, dist_mat.shape[0]), 0] = dist_p.cpu().detach().numpy()
            dist_mat[i*batch:min((i+1)*batch, dist_mat.shape[0]), 1:] = dist_n.cpu().detach().numpy()

        num_correct = np.sum(np.sum(dist_mat[:, 1:] > dist_mat[:, :1], axis=1) == self._n-1)
        num_total = dist_mat.shape[0]
        self._net.train()

        if not val:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            np.save('test_result_' + timestamp, dist_mat)
            print('Test accuracy saved: test_result_' + timestamp + '.npy')
            print('Test accuracy: {:f}'.format(num_correct/num_total))
            return 'test_result_' + timestamp + '.npy'
        else:
            return num_correct/num_total

    def _data_flip(self, data):
        idx = torch.randperm(data.size(0))[:data.size(0)//2]
        data[idx] = data[idx].flip(3)
        return data

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
    parser.add_argument('--net', dest='net', type=str, default='Triplet', help='Choose the network.')
    parser.add_argument('--param', dest='param', type=str, default=None, help='Initialize model parameters.')
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

    parser.add_argument('--valid', dest='valid', action='store_true', help=' Use validation.')
    parser.add_argument('--no-valid', dest='valid', action='store_false', help='Do not use validation.')
    parser.set_defaults(valid=True)

    args = parser.parse_args()

    if args.net not in ['Triplet', 'TripletConv5', 'Bilinear']:
        raise AttributeError('--net parameter must be \'Triplet\' or \'Bilinear\'.')
    if args.version not in [0, 1, 2]:
        raise AttributeError('--version parameter must be in [0, 1, 2]')
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
    print('Ver: {:d}'.format(args.version))
    print('Margin: {:.1f}'.format(args.margin))
    print('Validation: ' + str(args.valid))
    print('Pretrained parameters: ' + str(args.param))
    print('Freeze mode: ' + str(args.freeze))
    print('#Epoch: {:d}, #Batch: {:d}'.format(args.epoch, args.batch))
    print('Learning rate: {:.0e}'.format(args.lr))
    print('Weight decay: {:.0e}\n'.format(args.decay))

    cnn = AlexManager(root=root, data_opts=data_opts, net=args.net, freeze=args.freeze, val=args.valid,
                      margin=args.margin, lr=args.lr, decay=args.decay, batch=args.batch, param_path=args.param)
    cnn.train(epoch=args.epoch, verbose=args.verbose)
    cnn.test()


if __name__ == '__main__':
    main()
