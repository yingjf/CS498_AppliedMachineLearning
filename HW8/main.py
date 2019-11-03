import torch
import torchvision
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = torchvision.datasets.MNIST('hw8_data', train=True, download=True, transform=transformations)
mnist_test = torchvision.datasets.MNIST('hw8_data', train=False, download=True, transform=transformations)

print(len(mnist_train))
print(len(mnist_train[0]))
mnist_train

print(len(mnist_test))
print(len(mnist_test[0]))
mnist_test

from torch.utils.data import DataLoader
batch_size_train = 32
train_loader = DataLoader(mnist_train, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size_train, shuffle=True)

from torch import nn


class OneLayerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OneLayerModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        return out

from torch import optim
input_dim = 784
output_dim = 10
learning_rate = 0.001
momentum = 0.5
model = OneLayerModel(input_dim, output_dim)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
writer = SummaryWriter('logs/expt1')


def train(model, train_loader, val_loader, loss_func, opt, num_epochs=10, writer=None):
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        for batch_size, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = loss_func(outputs, labels)

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Calculate accuracy
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()

        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch + 1, num_epochs, loss.item(),
                                                                                correct, len(train_loader.dataset),
                                                                                100. * correct / len(
                                                                                    train_loader.dataset)))
        writer.add_scalar('Training Loss', loss.item(), epoch + 1)
        writer.add_scalar('Training Accuracy', correct, epoch + 1)

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                output = model(images)
                test_loss += loss_func(output, labels).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum()
        test_loss /= len(val_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                  len(test_loader.dataset),
                                                                                  100. * correct / len(
                                                                                      val_loader.dataset)))
        writer.add_scalar('Validation Loss', test_loss, epoch + 1)
        writer.add_scalar('Validation Accuracy', correct, epoch + 1)

class TwoLayerModel(nn.Module):
    ## YOUR CODE HERE ##
    def __init__(self, input_dim, hidden_size, output_dim):
      super(TwoLayerModel, self).__init__()
      self.fc1 = nn.Linear(input_dim, hidden_size)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
      x = x.view(x.size(0), -1)
      out = self.fc1(x)
      out = self.relu(out)
      out = self.fc2(out)
      return out

## YOUR CODE HERE ##
hidden_size = 500
model2 = TwoLayerModel(input_dim, hidden_size, output_dim)
loss2 = nn.CrossEntropyLoss()
optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate, momentum=momentum)
writer2 = SummaryWriter('logs/expt2')