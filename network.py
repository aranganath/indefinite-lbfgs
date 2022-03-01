import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from ARCLSR1q import ARCLSR1
from indeflbfgstr import indefLBFGS
from torch.nn import functional as F
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=1000, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.next()

# plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')


input_size = 784
hidden_sizes = [128, 64]
output_size = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)


# optimizer = optim.Adam(model.parameters(), lr=0.003)
optimizer = indefLBFGS(model.parameters(), history_size = 10, eta=0.15, eta1=0.25)
# optimizer = ARCLSR1(model.parameters(), gamma1 = 1, gamma2 =10, eta1 = 0.15, eta2 = 0.25, history_size =10, mu=100)
time0 = time()
epochs = 100
train_loss = []
for e in range(epochs):
    running_loss = 0
    for i, (images, labels) in enumerate(trainloader):
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = nn.CrossEntropyLoss()(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            output = model(images)
            loss = nn.CrossEntropyLoss()(output, labels)
            if loss.requires_grad:
                loss.backward()
            return loss

        #And optimizes its weights here
        optimizer.step(closure=closure)
        
        running_loss += loss.item()
        train_loss.append(loss.item())
        print("Batch/Epoch {}/{} - Training loss: {}".format(i, e, loss.item()))
    
    print("Batch/Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        

import pickle
with open('results/lbfgs.pkl','wb') as handle:
    pickle.dump(train_loss, handle,protocol=pickle.HIGHEST_PROTOCOL)

print("\nTraining Time (in minutes) =",(time()-time0)/60)