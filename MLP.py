#!/usr/bin/python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
torch.set_printoptions(profile="full")
torch.manual_seed(0)


class MLP(nn.Module):

    def __init__(self, hidden_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(784, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.log_softmax(self.fc2(x))
            
            return x


if __name__ == '__main__':
    # Hyperparameters
    epoch_nbr = 100
    batch_size = 100
    hidden_size = 500
    learning_rate = 0.001

    # Data loading
    X0 = Variable(torch.from_numpy(np.load("data/trn_img.npy")).type(torch.FloatTensor))
    lbl0 = Variable(torch.from_numpy(np.load("data/trn_lbl.npy")).type(torch.FloatTensor).long())
    X1 = Variable(torch.from_numpy(np.load("data/dev_img.npy")).type(torch.FloatTensor))
    lbl1 = Variable(torch.from_numpy(np.load("data/dev_lbl.npy")).type(torch.FloatTensor).long())

    net = MLP(hidden_size)
    #Training
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for e in range(epoch_nbr):
        print("Epoch", e)
        j = 0
        for i in range(0, X0.data.size(0), batch_size):
            j = j + 1
            optimizer.zero_grad() # Reset all gradients to 0
            predictions_train = net(X0[i:i+batch_size])
            loss = nn.CrossEntropyLoss()(predictions_train, lbl0[i:i+batch_size])
            loss.backward()
            optimizer.step() # Perform the weights update
            if j % 10 == 0:
            	print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      				e, j * 100, len(X0),100. * j / 100, loss.data[0]))
    #evaluation
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for i in range(0, X1.data.size(0), batch_size):
        predictions_test = net(X1[i:i+batch_size])
        test_loss += nn.CrossEntropyLoss()(predictions_test, lbl1[i:i+batch_size]).data[0]
        _, predicted = torch.max(predictions_test.data, 1)
        total += lbl1[i:i+batch_size].size(0)
        correct += (predicted.cpu() == lbl1[i:i+batch_size].data).sum()

    test_loss /= total
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    