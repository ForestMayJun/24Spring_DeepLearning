# cnn model for the pj
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.io import loadmat
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load data set
data = loadmat('/Users/lvangge/Documents/ /code/codes_/神经网络深度学习/pj1/pj1/digits.mat')
X = data['X']
y = data['y']
nLabels = max(y)[0]
Xvalid = data['Xvalid']
Xtest = data['Xtest']
yvalid = data['yvalid']
ytest = data['ytest']

# Standardize columns and add bias
def standardize_cols(M, mu=None, sigma2=None):
    M = M.astype(float)  # transform the matrix to float type
    nrows, ncols = M.shape

    if mu is None or sigma2 is None:
        mu = np.mean(M, axis=0)
        sigma2 = np.std(M, axis=0)
        # handle the situation that sigma == 0
        sigma2[sigma2 < np.finfo(float).eps] = 1

    S = M - mu
    if ncols > 0:
        S = S / sigma2

    return S, mu, sigma2

X, mu, sigma = standardize_cols(X)
# X = np.hstack((np.ones((X.shape[0], 1)), X)) # add bias column

Xvalid = standardize_cols(Xvalid, mu, sigma)[0]
# Xvalid = np.hstack((np.ones((Xvalid.shape[0], 1)), Xvalid))

Xtest = standardize_cols(Xtest, mu, sigma)[0]
# Xtest = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))

# Change the shape of data
X = X.reshape(-1, 1, 16, 16)
Xvalid = Xvalid.reshape(-1, 1, 16, 16)
Xtest = Xtest.reshape(-1, 1, 16, 16)

# Function to expand y to binary matrix
def linearInd2Binary(ind, nLabels):
    n = len(ind)
    y = np.zeros((n, nLabels))
    for i in range(n):
        y[i, int(ind[i])-1] = 1
    return y

# Trandform data to dataloader
y = linearInd2Binary(y,nLabels)
yvalid= linearInd2Binary(yvalid,nLabels)
ytest = linearInd2Binary(ytest,nLabels)

train_dataset = TensorDataset(torch.tensor(X,dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
valid_dataset = TensorDataset(torch.tensor(Xvalid,dtype=torch.float32), torch.tensor(yvalid, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(Xtest, dtype=torch.float32), torch.tensor(ytest,dtype=torch.float32))

train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Define train achitecture
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5) # -> 12*12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)# -> 8*8
        self.pool1 = nn.MaxPool2d(2) # -> 4*4
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, nLabels)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
model = Model()

# for x, y in train_dl:
#     print(y)
#     break

xx = torch.randn(1,1,16,16)
print(model(xx))

# Define loss function, learning rate and optimizer
loss_func = nn.CrossEntropyLoss()
learn_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

# Training loop
def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, train_err = 0, 0

    for X, y in dataloader:
        pred = model(X)
        loss = loss_func(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_err += (torch.argmax(F.softmax(pred, dim=1), dim=1) != torch.argmax(y, dim=1)).type(torch.float).sum().item()
        train_loss += loss.item()

    train_loss /= num_batches
    train_err /= size

    return train_loss, train_err

# Test-Valid loop
def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, test_err = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_func(pred, y)
            test_err += (torch.argmax(F.softmax(pred, dim=1), dim=1) != torch.argmax(y, dim=1)).type(torch.float).sum().item()
            test_loss += loss.item()

    test_loss /= num_batches
    test_err /= size
    
    return test_loss, test_err

# Main training and evaluating loop
if __name__ == '__main__':
    epochs = 10
    train_loss = []
    train_err = []
    valid_loss = []
    valid_err = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss, epoch_train_err = train(train_dl, model, loss_func, optimizer)

        model.eval()
        epoch_valid_loss, epoch_valid_err = test(valid_dl, model, loss_func)

        train_loss.append(epoch_train_loss)
        train_err.append(epoch_train_err)
        valid_loss.append(epoch_valid_loss)
        valid_err.append(epoch_valid_err)

        template = ('Epoch:{:2d}, train_err:{:.1f}%, train_loss:{:.3f}, valid_err:{:.1f}%, valid_loss:{:.3f}')
        print(template.format(epoch + 1, epoch_train_err*100, epoch_train_loss, epoch_valid_err*100, epoch_valid_loss))

    print("Traing done")

    test_loss, test_err = test(test_dl, model, loss_func)
    print(('Test_err:{:.1f}%, test_loss:{:.3f}').format(test_err*100, test_loss))

def show_plot():
    epoch_range = range(epoch)
    plt.figure(figsize=(12, 3))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, train_loss, label='Train Loss')
    plt.plot(epoch_range, valid_loss, label='Valid Loss')
    plt.legend()
    plt.title('Trian and Valid Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, train_err, label='Train Error')
    plt.plot(epoch_range, valid_err, label='Valid Error')
    plt.legend()
    plt.title('Trian and Valid Error')

    plt.show()

show_plot()


