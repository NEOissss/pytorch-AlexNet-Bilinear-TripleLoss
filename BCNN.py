from datetime import datetime
import numpy as np
from scipy import misc
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from SUN360DataLoader import *

class TripletAlex(torch.nn.Module):
    def __init__(self, freeze=None):
        torch.nn.Module.__init__(self)
        self.features = models.alexnet(pretrained=True).features
        fc_list = list(models.alexnet().classifier.children())[:-2]
        self.fc = torch.nn.Sequential(*fc_list)

        # Freeze layers.
        if freeze:
            self._freeze(freeze)

    def forward(self, X):
        X = X.float()
        N = X.size()[0]
        assert X.size() == (N, 3, 227, 227)
        X = self.features(X)
        X = X.view(N, 256 * 6 * 6)
        X = self.fc(X)
        assert X.size() == (N, 4096)
        X = X.div(X.norm(2))
        return X

    def _freeze(self, option):
        if option == 'part':
            for param in self.features.parameters():
                param.requires_grad = False
        elif option == 'all':
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = False
        else:
            raise ValueError('Unavailable freeze option.')


class BilinearTripletAlex(torch.nn.Module):
    def __init__(self, freeze=None, bi_dim=256, fc_dim=4096):
        torch.nn.Module.__init__(self)
        self.bi_dim = bi_dim
        self.fc_dim = fc_dim
        self.features = models.alexnet(pretrained=True).features
        bfc_list = list(models.alexnet().classifier.children())[:-1]
        bfc_list.append(torch.nn.Linear(4096, self.bi_dim))
        self.bfc = torch.nn.Sequential(*bfc_list)
        self.fc = torch.nn.Linear(self.bi_dim**2, self.fc_dim)

        # Freeze layers.
        if freeze:
            self._freeze(freeze)

        # Initialize the last bfc layers.
        torch.nn.init.kaiming_normal_(self.bfc[-1].weight.data)
        if self.bfc[-1].bias is not None:
            torch.nn.init.constant_(self.bfc[-1].bias.data, val=0)

        # Initialize the fc layers.
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, X):
        X = X.float()
        N = X.size()[0]
        assert X.size() == (N, 3, 227, 227)
        X = self.features(X)
        X = X.view(N, 256 * 6 * 6)
        X = self.bfc(X)
        assert X.size() == (N, self.bi_dim)
        X = X.view(N, -1, self.bi_dim)
        X = torch.matmul(torch.transpose(X, 1, 2), X)
        # Signed sqrt
        X = torch.sign(X).mul(torch.sqrt(X.abs()))
        # L2 normalization
        X = X.div(X.norm(2))
        assert X.size() == (N, self.bi_dim, self.bi_dim)
        X = X.view(N, self.bi_dim**2)
        X = self.fc(X)
        assert X.size() == (N, self.fc_dim)
        return X

    def _freeze(self, option):
        if option == 'part':
            for param in self.features.parameters():
                param.requires_grad = False
            for layer in self.bfc[:-1]:
                for param in layer.parameters():
                    param.requires_grad = False
        elif option == 'all':
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.bfc.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = False
        else:
            raise ValueError('Unavailable freeze option.')


class AlexManager(object):
    def __init__(self, freeze='part', batch=1, epoch=1, lr=1e-3, margin=1.0, param_path=None, net='Triplet', data_cut=None):
        if net=='BilinearTriplet':
            self._net = torch.nn.DataParallel(BilinearTripletAlex(freeze=freeze)).cuda()
        elif net=='Triplet':
            self._net = torch.nn.DataParallel(TripletAlex(freeze=freeze)).cuda()
        else:
            raise ValueError('Unavailable net option.')
        self._data_cut = data_cut
        self._net_name = net
        self._stats = []
        self._timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        #print(self._net)
        # Load parameters
        if param_path:
            self._load(param_path)
        # Criterion.
        self._margin = margin
        self._criterion = torch.nn.TripletMarginLoss(margin = self._margin).cuda()
        # Batch size
        self._batch = batch
        # Epoch
        self._epoch = epoch
        # Image transform
        self.transform = transforms.Compose([transforms.Resize(227),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        # If not test
        if freeze != 'all':
            # Solver.
            self._solver = torch.optim.Adam(filter(lambda p: p.requires_grad, self._net.parameters()), lr=lr)
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._solver, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4)

    def train(self, verbose=None):
        """Train the network."""
        print('Training.')
        best_iter = [0, 0, 0, 0]
        for t in range(self._epoch):
            print("Epoch: {:d}\n".format(t+1))
            num_correct = 0
            num_total = 0
            iter_num = 0
            for a, p, n in self._data_loader():
                # Data.
                A, P, N = self._image_loader(a, p, n)
                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                feat_a = self._net(A)
                feat_p = self._net(P)
                feat_n = self._net(N)
                accu = torch.sum((feat_a-feat_p).abs().sum(1) < (feat_a-feat_n).abs().sum(1)) / feat_a.size(0)
                loss = self._criterion(feat_a, feat_p, feat_n)
                # Backward pass.
                loss.backward()
                self._solver.step()
                iter_num += 1
                self._stats.append([t+1, iter_num, accu, loss.item()])
                if accu > best_iter[2]:
                    best_iter = [t+1, iter_num, accu, loss.item()]
                if verbose and iter_num%verbose == 0:
                    print('Batch: {:d}'.format(iter_num))
                    print('A feature sum: {:.4f}'.format(feat_a.sum()))
                    print('P distance: {:.4f}, N distance: {:.4f}'.format(torch.sqrt(torch.sum((feat_a-feat_p)**2)), torch.sqrt(torch.sum((feat_a-feat_n)**2))))
                    print('Triplet loss: {:.4f}'.format(loss.item()))
                    if self._net_name=='Triplet':
                        print('fc-2 weight sum: {:.4f}'.format(self._net.module.fc[1].weight.abs().sum()))
                        print('fc-1 weight sum: {:.4f}\n'.format(self._net.module.fc[-1].weight.abs().sum()))
                    else:
                        print('fc-2 weight sum: {:.4f}'.format(self._net.module.bfc[-1].weight.abs().sum()))
                        print('fc-1 weight sum: {:.4f}\n'.format(self._net.module.fc.weight.abs().sum()))
        self._stats = np.array(self._stats)
        print('Best iteration stats: ' + str(best_iter) + '\n')
        return self._save()

    def test(self, data='test'):
        print('Testing.')
        self._net.eval()
        data_path = self._data_loader(train=False, data=data)
        dist_mat = np.zeros((len(data_path), 10))
        for i,j in enumerate(data_path):
            # Data.
            A, P, N = j
            # Forward pass.
            feat_a = self._net(self._single_image_loader(A[0]))
            feat_p = self._net(self._single_image_loader(P[0]))
            dist_mat[i,0] = torch.sqrt(torch.sum(torch.abs(feat_a - feat_p))).cpu().detach().numpy()
            for k, n in enumerate(N):
                feat_n = self._net(self._single_image_loader(n))
                dist_mat[i, k+1] = torch.sqrt(torch.sum(torch.abs(feat_a - feat_n))).cpu().detach().numpy()
        np.save('test_result_' + self._timestamp + '.npy', dist_mat)
        print('Test accuracy saved: test_result_' + self._timestamp + '.npy')
        num_correct = np.sum(np.sum(dist_mat[:,1:] > dist_mat[:,:1], axis=1) == 9)
        num_total = len(data_path)
        print('Test accuracy ', num_correct/num_total)


    def _data_loader(self, train=True, data='train'):
        if train:
            return sun360h_data_load(task='train', batch=self._batch, cut=self._data_cut)
        else:
            return sun360h_data_load(task='test', data=data, batch=self._batch, cut=self._data_cut)

    def _single_image_loader(self, x):
        y_t = torch.zeros(1, 3, 227, 227)
        y_t[0,:,:,:] = self.transform(Image.open(x))
        return y_t
    def _image_loader(self, a, p, n):
        k = len(a)
        a_t = torch.zeros(k, 3, 227, 227)
        p_t = torch.zeros(k, 3, 227, 227)
        n_t = torch.zeros(k, 3, 227, 227)
        for i in range(k):
            a_t[i,:,:,:] = self.transform(Image.open(a[i]))
            p_t[i,:,:,:] = self.transform(Image.open(p[i]))
            n_t[i,:,:,:] = self.transform(Image.open(n[i]))
        return a_t, p_t, n_t

    def _save(self):
        PATH = self._net_name + '-param-' + self._timestamp
        path_stats = 'train_stats_' + self._timestamp
        torch.save(self._net.state_dict(), PATH)
        np.save(path_stats, self._stats)
        print('Model parameters saved: ' + PATH)
        print('Training stats saved: ' + path_stats + '.npy\n')
        return PATH

    def _load(self, PATH):
        self._net.load_state_dict(torch.load(PATH))
        print('Model parameters loaded: ' + PATH)


def train(freeze='part', batch=10, epoch=20, lr=0.1, net='Triplet', verbose=2, path=None, data_cut=None):
    margin = 1.0 if net=='Triplet' else 5.0
    bcnn = AlexManager(freeze=freeze, batch=batch, epoch=epoch, lr=lr, margin=margin, param_path=path, net=net, data_cut=data_cut)
    return bcnn.train(verbose=verbose)

def test(net='Triplet', path=None, data='test', data_cut=None):
    bcnn = AlexManager(freeze='all', param_path=path, net=net, data_cut=data_cut)
    bcnn.test(data=data)

def main():
    ini_param = None
    freeze = 'part'
    batch_size = 20
    epoch_num = 10
    learning_rate = 0.001
    net_name = 'Triplet'
    verbose = 2
    test_data = 'test'
    data_size = None #[0, 100]

    path = train(freeze=freeze, batch=batch_size, epoch=epoch_num, lr=learning_rate, net=net_name, verbose=verbose, path=ini_param, data_cut=data_size)
    test(net=net_name, path=path, data=test_data, data_cut=data_size)
    #test(net=net_name, path=ini_param, data=test_data, data_cut=data_size)
    print('\n====Exp details====')
    print('Net: ' + net_name)
    if ini_param:
        print('Pretrained parameters: ' + ini_param)
    if freeze:
        print('Freeze mode: ' + freeze)
    print('Epoch: {:d}, Batch: {:d}'.format(epoch_num, batch_size))
    print('Test dataset: ' + test_data)
    if data_size:
        print('Data chunk start: {:d}, Data chunk length: {:d}'.format(data_size[0], data_size[1]))
    print('Learning rate: {:.4f}'.format(learning_rate))

if __name__ == '__main__':
    main()
