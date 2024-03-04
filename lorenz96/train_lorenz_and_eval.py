import sys
sys.path.append('../notebooks')

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torchdiffeq import odeint_adjoint as odeint

import ftmle
import lorenz96

warnings.filterwarnings('ignore')

class Lorenz96Integrator(nn.Module):
    def __init__(self, odefunc, integration_time, method='rk4', step_size=0.01):
        super(Lorenz96Integrator, self).__init__()
        self.odefunc = odefunc
        self.integration_time = integration_time
        self.method = method
        self.step_size = step_size

    def forward(self, x):
        if self.integration_time[-1] > 0:
            x = odeint(self.odefunc, x, self.integration_time, method=self.method, options=dict(step_size=self.step_size))
            x = x[-1]
        return x

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(seed)
# random.seed(seed)
# rng = np.random.RandomState(seed=seed)

#General hyper parameters
num_epochs = 20         # number of epochs
num_batch = 500        # number of batch
learning_rate = 0.001   # learning rate
image_size = 28*28      # size of image
label_size = 10
samplingInterval = 0.05
ftmle_batch= 2048

#Special hyper parameters
systemCount = 500

#iterationSize = 25

# Do you use a GPU?
if torch.cuda.is_available():
   device = 'cuda'
   torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
   device = 'cpu'

#device = 'cpu'

#----------------------------------------------------------
# Make data set

# transformation
transform = transforms.Compose([
    transforms.ToTensor()
    ])

# Obtain MNIST data
# https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
# for training
train_dataset = datasets.MNIST(
    '../data',               # directory of data
    train = True,           # obtain training data
    download = True,        # if you don't have data, download
    transform = transform   # trainsform to tensor
)
# evaluation
test_dataset = datasets.MNIST(
    '../data', 
    train = False,
    transform = transform
)

# data loader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = num_batch,
    shuffle = True,
    #num_workers = 2,
    generator=torch.Generator(device='cuda')
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,     
    batch_size = num_batch,
    shuffle = False,
    #num_workers = 2,
    generator=torch.Generator(device='cuda')
)

"""
Change the following:
"""
f = 4.5
t = 5

commandLineConfigs = {"f": f, "iteration": t}

model = lorenz96.ODEBlock(image_size, label_size, systemCount, commandLineConfigs).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

bestResult = 0
bestLoss = 1e6

initialStates = torch.empty([0,systemCount])

for epoch in tqdm(range(num_epochs)):
    initialStates = torch.empty([0,systemCount])
    # train
    model.train()  
    loss_sum = 0    
    for inputs, labels in train_dataloader:

        # send data to gpu, if gpu can be used
        inputs = inputs.to(device)
        labels = labels.to(device)

        # initialize optimizer
        optimizer.zero_grad()

        # process neural net
        inputs = inputs.view(-1, image_size) # reshape the data
        outputs = model(inputs)

        # calculate loss
        loss = criterion(outputs, labels)
        loss_sum += loss

        # calculate gradient
        loss.backward()

        # optimize the weight
        optimizer.step()

    #torch.save(model.state_dict(), 'model_weights_after.pth')
    #np.savetxt("weight_after.txt", model.fc1.weight.cpu().detach().numpy())

    #PCA make PC vectors
    # pca_initialStates = PCA(n_components=n_components)
    # pca_initialStates.fit(initialStates)

    # pca_finalStates = PCA(n_components=n_components)
    # pca_finalStates.fit(finalStates)        

    #----------------------------------------------------------
    # evaluate
    model.eval()  # change model mode to the evaluation

    loss_sum = 0
    correct = 0
    count = 0

    first = True
    y_pred = np.empty((0), int)
    y_test = np.empty((0), int)

    with torch.no_grad():

        for inputs, labels in test_dataloader:

            # send data to gpu, if gpu can be used
            inputs = inputs.to(device)
            labels = labels.to(device)

            # process neural net
            inputs = inputs.view(-1, image_size) 
            # if first:
            #     outputs = model.forward_plot(inputs)
            #     stateFigureFileName = "figure/colormap/" + configName + "_epoch=" + str(epoch)
            #     model.plotStates(stateFigureFileName)
            #     first = False
            # else:
            outputs = model(inputs)

            if epoch + 1 == num_epochs:
                weightFileName = f'./save_models/lorenz_f{f}_t{t}.pth'
                torch.save(model.state_dict(), weightFileName)

                initialStates = torch.concatenate([initialStates, model.initialStates])

                ode_layer = Lorenz96Integrator(lorenz96.Lorenz96_ODEfunc(f), torch.tensor([0.0, t]))

                ftmle_iteration_layers = []
                for i in tqdm(range(0, len(initialStates), ftmle_batch)):
                    end = min(i+ftmle_batch, len(initialStates))
                    input_data = torch.from_numpy(initialStates[i:end])
                    exp, ftmle = ftmle.compute_jacob_ftmle(ode_layer, input_data, t, device)
                    ftmle_iteration_layers.append(ftmle)
                ftmle_iteration_layers = np.concatenate(ftmle_iteration_layers, axis=0)
                print(f, max(ftmle_iteration_layers), min(ftmle_iteration_layers))
                np.save(f'./result/ftmle/ftmle_f{f}_t{train_dataset}.npy', ftmle_iteration_layers)

            # calculate loss
            loss_sum += criterion(outputs, labels)

            # obtain answer labels
            pred = outputs.argmax(1)
            # count correct data
            correct += pred.eq(labels.view_as(pred)).sum().item()
            count += 1               
            #hoge = pred.cpu().detach().numpy()
            y_pred = np.append(y_pred, pred.cpu().detach().numpy())
            y_test = np.append(y_test, labels.view_as(pred).cpu().detach().numpy())
            pred = outputs.argmax(1)
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")
        bestResult = max(correct/len(test_dataset), bestResult)
        bestLoss = min(loss_sum.item() / len(test_dataloader), bestLoss)

