import os
import time
import logging
import random
from pickle import FALSE

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import common
import plot

#----------------------------------------------------------
#General hyper parameters
num_epochs = 20         # number of epochs
num_batch = 500         # number of batch
learning_rate = 0.001   # learning rate
image_size = 28*28      # size of image
num_neuron = 500

#Special hyper parameters
iterationSize = 5
logisticParameter = 3.8

# Do you use a GPU?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#----------------------------------------------------------
# Make data set

def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

torch_fix_seed()

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
    shuffle = True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,     
    batch_size = num_batch,
    shuffle = True)

#----------------------------------------------------------
# Definition of neural net
class Net(nn.Module):
    def __init__(self, input_size, output_size, iterationCount, logisticParameter):
        super(Net, self).__init__()

        # Instances of class
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(14*14*16, num_neuron)
        self.fc2 = nn.Linear(num_neuron, output_size)
        
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        #hyper parameters
        self.iterationCount = iterationCount
        self.logisticParameter = logisticParameter

    def forward(self, x):
    # forward propagation
        x = self.pool(self.act(self.conv1(x)))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    # backward propagation is not defined explicitly, but we should compair both cases

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("result", exist_ok=True)
os.makedirs("result/weight", exist_ok=True)
os.makedirs("result/state", exist_ok=True)
os.makedirs("figure", exist_ok=True)
os.makedirs("figure/colormap", exist_ok=True)
os.makedirs("figure/table", exist_ok=True)

#logging
logging.basicConfig(filename='status.log', level=logging.INFO, format=common.LOG_FORMAT)
iteration = [0]
result = 0
losses = 10

iteration = range(iterationSize)

#iterate through the iteration timestep
for t in iteration:
    initialTime = time.time()
    trainingFileName = common.RESULT_DIR_NAME + '/training'
    logging.info("Start cluculation: " + str(time.time()) + " Iteration: " + str(iteration[t])+ ", device: " + device)
    print("Iteration count: " + str(iteration[t]))
    #----------------------------------------------------------
    # make neural net
    model = Net(image_size, 10, t, logisticParameter).to(device)

    # save the weight
    torch.save(model.state_dict(), 'weight/model_weights_iteration='+ str(iteration[t]) +'_before.pth')
    np.savetxt("weight/weight1_iteration="+ str(iteration[t]) +"_before.txt", model.fc1.weight.cpu().detach().numpy())
    
    torch.save(model.state_dict(), 'model_weights_before.pth')


    #print(model.fc1.weight)
    #print(model.fc1.bias)

    #----------------------------------------------------------
    # make loss function
    criterion = nn.CrossEntropyLoss() 

    #----------------------------------------------------------
    # make optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    #----------------------------------------------------------
    # train
    model.train()  

    for epoch in range(num_epochs): 
        # train
        model.train()  

        loss_sum = 0    
        for inputs, labels in train_dataloader:

            # send data to gpu, if gpu can be used
            #inputs /= image_size
            inputs = inputs.to(device)
            labels = labels.to(device)

            # initialize optimizer
            optimizer.zero_grad()

            # process neural net
            #inputs = inputs.view(-1, image_size) # reshape the data
            outputs = model(inputs)
            #outputs = model.forward_plot(inputs, epoch)

            # calculate loss
            loss = criterion(outputs, labels)
            loss_sum += loss

            # calculate gradient
            loss.backward()

            # optimize the weight
            optimizer.step()
        #torch.save(model.state_dict(), 'model_weights_after.pth')
        #np.savetxt("weight_after.txt", model.fc1.weight.cpu().detach().numpy())
    
        # output
        #print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")
        #with open(trainingFileName, 'a') as f_handle:
        #    f_handle.write(str(epoch+1) + " " + str(loss_sum.item()) + "\n")



        #----------------------------------------------------------
        # evaluate
        model.eval()  # change model mode to the evaluation

        loss_sum = 0
        correct = 0
        count = 0

        with torch.no_grad():
            first = True
            y_pred = np.empty((0), int)
            y_test = np.empty((0), int)

            for inputs, labels in test_dataloader:

                # send data to gpu, if gpu can be used
                #inputs /= image_size
                inputs = inputs.to(device)
                labels = labels.to(device)

                # process neural net
                #inputs = inputs.view(-1, image_size) 
                if first:
                    outputs = model(inputs)
                    first = False
                else:
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
        with open(trainingFileName, 'a') as f_handle:
            f_handle.write(str(epoch+1) + " " + str(correct/len(test_dataset)) + " " + str(loss_sum.item() / len(test_dataloader)) + "\n")

        result = max(correct/len(test_dataset), result)
        losses = min(loss_sum.item() / len(test_dataloader), losses)

        #fileName = 'figure/table/epoch='+ str(epoch)
        #plot.plot_mnistTable(y_test, y_pred, fileName)

        
    weightFileName = 'result/weight/model_weights_after'
    weightFileName += '.pth'
    torch.save(model.state_dict(), weightFileName)
    #np.savetxt("result/weight/weight_fc1_after.txt", model.fc1.weight.cpu().detach().numpy())
    #np.savetxt("result/weight/weight_fc2_after.txt", model.fc2.weight.cpu().detach().numpy())

    
    fileName = common.RESULT_DIR_NAME + "/accuracy.txt"
    with open(fileName, 'a') as f_handle:
        f_handle.write(str(iteration) + " " + str(result) + " " + str(losses) + "\n")
    
    print("Test maximum accuracy: {:.5f}, minimum loss: {:.5f}".format(result, losses))
    logging.info("total time: " + str(time.time() - initialTime))

os.makedirs("figure", exist_ok=True)
figureFileName = "figure/accuracy_chaos_outputTrainOnly"
plot.plot_accuracy(iteration, result, figureFileName)

