from datetime import datetime
import numpy as np
from scipy import misc
import torch
import torchvision.models as models
from SUN360DataLoader import *

class BilinearAlex(torch.nn.Module):
    def __init__(self, freeze=None):
        torch.nn.Module.__init__(self)
        self.features = models.alexnet(pretrained=True).features
        self.bi_dim = 128
        bfc_list = list(models.alexnet().classifier.children())[:-1]
        bfc_list.append(torch.nn.Linear(4096, self.bi_dim))
        bfc_list.append(torch.nn.ReLU(inplace=True))
        self.bfc = torch.nn.Sequential(*bfc_list)
        self.fc = torch.nn.Linear(self.bi_dim**2, 1024)

        # Freeze layers.
        if freeze:
            self._freeze(freeze)

        # Initialize the last bfc layers.
        torch.nn.init.kaiming_normal_(self.bfc[-2].weight.data)
        if self.bfc[-2].bias is not None:
            torch.nn.init.constant_(self.bfc[-2].bias.data, val=0)

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
        X = torch.sqrt(X)
        # L2 normalization
        X = X.div(X.norm(2))
        assert X.size() == (N, self.bi_dim, self.bi_dim)
        X = X.view(N, self.bi_dim**2)
        X = self.fc(X)
        assert X.size() == (N, 1024)
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


class BilinearAlexManager(object):
    def __init__(self, freeze='part', batch=1, epoch=1, param_path=None):
        self._net = torch.nn.DataParallel(BilinearAlex(freeze=freeze)).cuda()
        print(self._net)
        # Load parameters
        if param_path:
            self._load(param_path)
        # Criterion.
        self._margin = 1.0
        self._criterion = torch.nn.TripletMarginLoss(margin = self._margin).cuda()
        # Batch size
        self._batch = batch
        # Epoch
        self._epoch = epoch
        # If not test
        if freeze != 'all':
            # Solver.
            self._solver = torch.optim.Adam(filter(lambda p: p.requires_grad, self._net.parameters()), lr=1e-3)
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._solver, mode='max', factor=0.1, patience=3, verbose=True, threshold=1e-4)

    def train(self):
        """Train the network."""
        print('Training.')
        for t in range(self._epoch):
            print("Epoch: " + str(self._epoch))
            epoch_loss = []
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
                loss = self._criterion(feat_a, feat_p, feat_n)
                epoch_loss.append(loss.data[0])
                # Backward pass.
                loss.backward()
                self._solver.step()
                iter_num += 1
                if iter_num%1 == 0:
                    print('A feature sum: {:.4f}'.format(feat_a.sum()))
                    print('P distance: {:.4f}, N distance: {:.4f}'.format(torch.sqrt(torch.sum((feat_a-feat_p)**2)), torch.sqrt(torch.sum((feat_a-feat_n)**2))))
                    print('fc1 weight sum: {:.4f}'.format(self._net.module.bfc[-2].weight.sum()))
                    print('fc2 weight sum: {:.4f}'.format(self._net.module.fc.weight.sum()))
                    print('Triplet loss: {:.4f} \n'.format(epoch_loss[-1]))
        return self._save()

    def test(self, data='test'):
        print('Testing.')
        data_path = self._data_loader(train=False, data=data)
        dist_mat = np.zeros((len(data_path), 10))
        for i,j in enumerate(data_path):
            # Data.
            A, P, N = j
            # Forward pass.
            feat_a = self._net(self._single_image_loader(A[0]))
            feat_p = self._net(self._single_image_loader(P[0]))
            dist_mat[i,0] = torch.sqrt(torch.sum((feat_a - feat_p)**2)).cpu().detach().numpy()
            for k, n in enumerate(N):
                feat_n = self._net(self._single_image_loader(n))
                dist_mat[i, k+1] = torch.sqrt(torch.sum((feat_a - feat_n)**2)).cpu().detach().numpy()
        np.save('test_result.npy', dist_mat)

        num_correct = np.sum(np.sum(dist_mat[:,1:] > dist_mat[:,:1], axis=1) == 9)
        num_total = len(data_path)

        print('Test accuracy ', num_correct/num_total)


    def _data_loader(self, train=True, data='train'):
        if train:
            return sun360h_data_load(task='train', batch=self._batch)
        else:
            return sun360h_data_load(task='test', data=data, batch=self._batch)

    def _single_image_loader(self, x):
        y = np.ndarray([1, 3, 227, 227])
        y[0,:,:,:] = np.transpose(misc.imresize(misc.imread(x), size=(227,227,3)), (2,0,1))
        return torch.from_numpy(y)

    def _image_loader(self, a, p, n):
        k = len(a)
        a_numpy = np.ndarray([k, 3, 227, 227])
        p_numpy = np.ndarray([k, 3, 227, 227])
        n_numpy = np.ndarray([k, 3, 227, 227])
        for i in range(k):
            a_numpy[i,:,:,:] = np.transpose(misc.imresize(misc.imread(a[i]), size=(227,227,3)), (2,0,1))
            p_numpy[i,:,:,:] = np.transpose(misc.imresize(misc.imread(p[i]), size=(227,227,3)), (2,0,1))
            n_numpy[i,:,:,:] = np.transpose(misc.imresize(misc.imread(n[i]), size=(227,227,3)), (2,0,1))
        return torch.from_numpy(a_numpy), torch.from_numpy(p_numpy), torch.from_numpy(n_numpy)

    def _save(self):
        PATH = './bcnn-param-' + datetime.now().strftime('%Y%m%d%H%M%S')
        torch.save(self._net.state_dict(), PATH)
        print('Model parameters saved: ' + PATH)
        return PATH

    def _load(self, PATH):
        self._net.load_state_dict(torch.load(PATH))
        print('Model parameters loaded: ' + PATH)


def train():
    bcnn = BilinearAlexManager(freeze='part')
    return bcnn.train()

def test(path):
    bcnn = BilinearAlexManager(freeze='all', param_path=path)
    bcnn.test(data='train')


if __name__ == '__main__':
    path = train()
    test(path)
