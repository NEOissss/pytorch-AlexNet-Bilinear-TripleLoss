import os
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from SUN360Dataset import Sun360Dataset


class AlexFeature(torch.nn.Module):
    def __init__(self):
        super(AlexFeature, self).__init__()
        self.part1 = models.alexnet(pretrained=True).features
        self.part2 = torch.nn.Sequential(*list(models.alexnet(pretrained=True).classifier.children())[:-2])
        self.part1 = self.part1.double()
        self.part2 = self.part2.double()

    def forward(self, x):
        x = self.part1(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.part2(x)
        return x


def main():
    root = '/mnt/nfs/scratch1/gluo/SUN360/HalfHalf/'
    path = 'IMGs_fc7/'
    train_dataset = Sun360Dataset(root=root, train=False, dataset='train')
    test_dataset = Sun360Dataset(root=root, train=False, dataset='test')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    net = AlexFeature().float()
    net.eval()

    if not os.path.exists(root + path):
        os.mkdir(root + path)

    for i in ['train_v0', 'test_v0']:
        if not os.path.exists(root + path + i):
            os.mkdir(root + path + i)

    for i, data in enumerate(train_dataloader):
        feat = net(data.view(-1, 3, 227, 227))
        torch.save(feat, root + path + 'train_v0/' + '0'*(9-len(str(i))) + str(i) + '.pt')
    print('Extract {:d} training data features'.format(i+1))

    for i, data in enumerate(test_dataloader):
        feat = net(data.view(-1, 3, 227, 227))
        torch.save(feat, root + path + 'test_v0/' + '0'*(9-len(str(i))) + str(i) + '.pt')
    print('Extract {:d} testing data features'.format(i+1))


if __name__ == '__main__':
    main()
