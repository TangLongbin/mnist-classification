import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from tqdm import tqdm

# tunable parameters
DOWNLOAD_MNIST = False

if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)


class logistic_regression_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = nn.Sequential(
            nn.Linear(28*28, 10)
        )

    def forward(self, x):
        output = self.lr(x)
        return output


def logistic_regression(epochs=5, lr=0.001, batch_size=50):
    logistic_net = logistic_regression_net()
    optimizer = torch.optim.Adam(logistic_net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    test_images = torch.unsqueeze(test_data.data, dim=1).type(
        torch.FloatTensor)[:2000]/255.
    test_labels = test_data.targets[:2000]

    print('training start:')
    for epoch in range(epochs):
        for _, (b_x, b_y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            b_x = b_x.view(-1, 28*28)

            output = logistic_net(b_x)
            l = loss(output, b_y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        test_output = logistic_net(test_images.view(-1, 28*28))
        pred_labels = torch.max(test_output, 1)[1].data.numpy()
        accuracy = float((pred_labels == test_labels.data.numpy()).astype(
            int).sum()) / float(test_labels.size(0))
        print('Epoch: ', epoch, ', train loss: %.6f' %
              l.data.numpy(), ', test accuracy: %.4f' % accuracy)

    test_pred = logistic_net(test_images[:10].view(-1, 28*28))
    test_pred = torch.max(test_pred, 1)[1].data.numpy()
    print('prediction  ', test_pred)
    print('ground truth', test_labels[:10].numpy())


if __name__ == "__main__":
    logistic_regression()
