import argparse
from datetime import datetime
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from SUN360Dataset import Sun360Dataset


class TripletAlex(torch.nn.Module):
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


class BilinearTripletAlex(torch.nn.Module):
    def __init__(self, freeze=None, bi_dim=256, fc_dim=4096):
        torch.nn.Module.__init__(self)
        self.bi_dim = bi_dim
        self.fc_dim = fc_dim
        self.features = models.alexnet(pretrained=True).features
        bfc_list = list(models.alexnet(pretrained=True).classifier.children())[:-1]
        bfc_list.append(torch.nn.Linear(4096, self.bi_dim))
        self.bfc = torch.nn.Sequential(*bfc_list)
        self.fc = torch.nn.Linear(self.bi_dim**2, self.fc_dim)

        # Freeze layers.
        if freeze:
            self._freeze()

        # Initialize the last bfc layers.
        torch.nn.init.kaiming_normal_(self.bfc[-1].weight.data)
        if self.bfc[-1].bias is not None:
            torch.nn.init.constant_(self.bfc[-1].bias.data, val=0)

        # Initialize the fc layers.
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        x = x.float()
        n = x.size()[0]
        assert x.size() == (n, 3, 227, 227)
        x = self.features(x)
        x = x.view(n, 256 * 6 * 6)
        x = self.bfc(x)
        assert x.size() == (n, self.bi_dim)
        x = x.view(n, -1, self.bi_dim)
        x = torch.matmul(torch.transpose(x, 1, 2), x)
        # Signed square root
        x = torch.sign(x).mul(torch.sqrt(x.abs()))
        # L2 normalization
        x = x.div(x.norm(2))
        assert x.size() == (n, self.bi_dim, self.bi_dim)
        x = x.view(n, self.bi_dim**2)
        x = self.fc(x)
        assert x.size() == (n, self.fc_dim)
        return x

    def _freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False
        for layer in self.bfc[:-1]:
            for param in layer.parameters():
                param.requires_grad = False


class AlexManager(object):
    def __init__(self, root, data_opts, freeze='part', val=True, batch=1, lr=1e-3, decay=0,
                 margin=1.0, param_path=None, net='Triplet'):
        if net == 'BilinearTriplet':
            self._net = torch.nn.DataParallel(BilinearTripletAlex(freeze=freeze)).cuda()
        elif net == 'Triplet':
            self._net = torch.nn.DataParallel(TripletAlex(freeze=freeze)).cuda()
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
        self._solver = torch.optim.Adam(filter(lambda p: p.requires_grad, self._net.parameters()),
                                        lr=lr, weight_decay=decay)
        # self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._solver, mode='max', factor=0.1,
        #                                                              patience=3, verbose=True, threshold=1e-4)

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
                data = data.reshape(-1, 3, 227, 227)
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
            data = data.reshape(-1, 3, 227, 227)
            feat = self._net(data)
            feat = feat.reshape(feat.size(0)//11, 11, -1)
            dist_p = torch.sqrt(((feat[:, 0, :] - feat[:, 1, :])**2).sum(1))
            dist_n = torch.sqrt(((feat[:, :1, :] - feat[:, 2:, :])**2).sum(2))
            dist_mat[i*self._batch:min((i+1)*self._batch, dist_mat.shape[0]), 0] = dist_p.cpu().detach().numpy()
            dist_mat[i*self._batch:min((i+1)*self._batch, dist_mat.shape[0]), 1:] = dist_n.cpu().detach().numpy()

        num_correct = np.sum(np.sum(dist_mat[:, 1:] > dist_mat[:, :1], axis=1) == 9)
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

    def _data_loader(self, root):
        train_data = self.data_opts['train']['set']
        test_data = self.data_opts['train']['set']
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', dest='net', type=str, default='Triplet', help='Choose the network.')
    parser.add_argument('--param', dest='param', type=str, default=None, help='Initialize model parameters.')

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

    if args.net not in ['Triplet', 'BilinearTriplet']:
        raise AttributeError('--net parameter must be \'Triplet\' or \'BilinearTriplet\'.')
    if args.lr <= 0:
        raise AttributeError('--lr parameter must > 0.')
    if args.decay <= 0:
        raise AttributeError('--decay parameter must > 0.')
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
    print('Freeze mode: ' + str(args.freeze))
    print('#Epoch: {:d}, #Batch: {:d}'.format(args.epoch, args.batch))
    print('Learning rate: {:f}\n'.format(args.lr))

    cnn = AlexManager(root=root, data_opts=data_opts, net=args.net, freeze=args.freeze, val=args.valid,
                      margin=args.margin, lr=args.lr, decay=args.decay, batch=args.batch, param_path=args.param)
    cnn.train(epoch=args.epoch, verbose=args.verbose)
    cnn.test()


if __name__ == '__main__':
    main()
