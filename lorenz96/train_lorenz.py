import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torchdiffeq import odeint_adjoint as odeint
import lorenz96

warnings.filterwarnings('ignore')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(seed)
# random.seed(seed)
# rng = np.random.RandomState(seed=seed)

# General hyper parameters
num_epochs = 20 # number of epochs
num_batch = 500 # number of batch
learning_rate = 0.001 # learning rate
image_size = 28*28 # size of image
label_size = 10
samplingInterval = 0.05
ftmle_batch= 2048

# Special hyper parameters
systemCount = 500
iterationSize = 5.0 # change here for your interested iteration time

# Do you use a GPU?
if torch.cuda.is_available():
   device = 'cuda'
   torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
   device = 'cpu'

#----------------------------------------------------------
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
fs = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
iterations = [iterationSize]*len(fs)  

for f, iteration in zip(fs, iterations):
    print(f'~~~~ f = {f}, t = {iteration} ~~~~')
    commandLineConfigs = {"f": f, "iteration": iteration}

    model = lorenz96.ODEBlock(image_size, label_size, systemCount, commandLineConfigs).to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    bestResult = 0
    bestLoss = 1e6

    initialStates = np.empty([0,systemCount])
    finalStates = np.empty([0,systemCount])
    class_labels = np.empty([0])

    for epoch in tqdm(range(num_epochs)):
        initialStates = np.empty([0,systemCount])
        finalStates = np.empty([0,systemCount])
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

            #save initial and final states for PCA
            # initialStates = np.concatenate([initialStates, model.initialStates.cpu().detach().numpy()])
            # finalStates = np.concatenate([finalStates, model.finalStates.cpu().detach().numpy()])

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

        y_pred = np.empty((0), int)
        y_test = np.empty((0), int)

        with torch.no_grad():

            for inputs, labels in test_dataloader:

                # send data to gpu, if gpu can be used
                inputs = inputs.to(device)
                labels = labels.to(device)

                # process neural net
                inputs = inputs.view(-1, image_size) 
                outputs = model(inputs)

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

    weightFileName = f'./save_models/lorenz_f{f}_t{iteration}.pth'
    torch.save(model.state_dict(), weightFileName)