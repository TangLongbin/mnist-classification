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
else:
    DOWNLOAD_MNIST = False

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


def main(epochs=5, lr=0.0001, batch_size=50):
    logistic_net = logistic_regression_net()
    optimizer = torch.optim.Adam(logistic_net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False)

    print('training start:')
    for epoch in range(epochs):
        logistic_net.train()
        l_all = 0
        count = 0
        for _, (b_x, b_y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            b_x = b_x.view(-1, 28*28)

            output = logistic_net(b_x)
            l = loss(output, b_y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_all += l
            count += 1

        logistic_net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for b_x, b_y in test_loader:
                b_x = b_x.view(-1, 28*28)
                test_output = logistic_net(b_x)
                pred_y = torch.max(test_output, 1)[1]
                correct += (pred_y == b_y).sum().item()
                total += b_y.size(0)
            mean_loss = l_all.item()/count
            accuracy = correct / total
            print(f'Epoch: {epoch + 1}, Train Loss: {mean_loss:.6f}, Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()
