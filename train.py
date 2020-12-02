import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor, FloatTensor, LongTensor
x_fn = "x.txt"
y_fn = "y.txt"


# load y:
with open(y_fn, "r") as y_f:
    y = np.array([int(i) for i in y_f.readlines()])#.reshape((-1, 1))

# load x:
with open(x_fn, "r") as x_f:
    x = np.array([eval(i) for i in x_f.readlines()])

print(len(x))
print(len(y))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(14, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, max(y) + 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output


# Now let's train a model
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

model = Net().to(device)
optimizer = optim.Adadelta(model.parameters())

train_dataset = TensorDataset(Tensor(x[:-500]), LongTensor(y[:-500]))  # create your datset
test_dataset = TensorDataset(Tensor(x[-500:]), LongTensor(y[-500:]))  # create your datset

train_loader = DataLoader(train_dataset, batch_size=32)  # create your dataloader
test_loader = DataLoader(test_dataset, batch_size=1)  # create your dataloader

def train():
    model.train()
    for epoch in range(1000):
        log_interval = 100
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                test(model, device, test_loader)
                model.train()



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print("prediction: ", pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



train()
