#!/usr/bin/python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
torch.set_printoptions(profile="full")
torch.manual_seed(0)


class LeNet(nn.Module):

    def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.drop = nn.Dropout2d()
            self.fc1 = nn.Linear(4*4*16, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
            x = x.view(-1, 1, 28, 28)
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(x.size(0), -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, training=self.training)
            x = F.log_softmax(self.fc3(x))
            
            return x


if __name__ == '__main__':
    # Hyperparameters
    epoch_nbr = 100
    batch_size = 100
    learning_rate = 0.009

    # Data loading
    X0 = Variable(torch.from_numpy(np.load("data/trn_img.npy")).type(torch.FloatTensor))
    lbl0 = Variable(torch.from_numpy(np.load("data/trn_lbl.npy")).type(torch.FloatTensor).long())
    X1 = Variable(torch.from_numpy(np.load("data/dev_img.npy")).type(torch.FloatTensor))
    lbl1 = Variable(torch.from_numpy(np.load("data/dev_lbl.npy")).type(torch.FloatTensor).long())

    net = LeNet()
    #Training
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for e in range(epoch_nbr):
        print("Epoch", e)
        j = 0
        for i in range(0, X0.data.size(0), batch_size):
            j = j + 1
            optimizer.zero_grad() # Reset all gradients to 0
            predictions_train = net(X0[i:i+batch_size])
            loss = F.nll_loss(predictions_train, lbl0[i:i+batch_size])
            loss.backward()
            optimizer.step() # Perform the weights update
            if j % 10 == 0:
            	print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      				e, j * batch_size, len(X0),100. * j / (len(X0)/batch_size), loss.data[0]))
    #evaluation
    net.eval()
    test_loss = 0
    correct = 0
    for i in range(0, X1.data.size(0), batch_size):
        predictions_test = net(X1[i:i+batch_size])
        test_loss += F.nll_loss(predictions_test, lbl1[i:i+batch_size], size_average=False).data[0]
        pred = predictions_test.data.max(1, keepdim=True)[1] 
        correct += pred.eq(lbl1[i:i+batch_size].data.view_as(pred)).cpu().sum()

    test_loss /= len(X1)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(X1),
        100. * correct / len(X1)))

