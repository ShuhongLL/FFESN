import os
import sys
import math
import copy
import time
import datetime
import logging
from pickle import FALSE

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import common
import file
import plot
import lorenz96
from torchdiffeq import odeint_adjoint as odeint

#General hyper parameters
image_size = 28*28      # size of image
label_size = 10

#Special hyper parameters
systemCount = 500
#iterationSize = 25

if torch.cuda.is_available():
   device = 'cuda'
   torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
   device = 'cpu'

def neuralOde(configs):
    os.makedirs(common.RESULT_DIR_NAME, exist_ok=True)
    os.makedirs(common.RESULT_DIR_NAME + '/log', exist_ok=True)
    os.makedirs(common.RESULT_DIR_NAME + '/weight', exist_ok=True)
    os.makedirs(common.RESULT_DIR_NAME + '/state', exist_ok=True)
    os.makedirs(common.FIGURE_DIR_NAME, exist_ok=True)
    os.makedirs(common.FIGURE_DIR_NAME + '/colormap', exist_ok=True)
    os.makedirs(common.FIGURE_DIR_NAME + '/table', exist_ok=True)
    os.makedirs(common.FIGURE_DIR_NAME + '/pca', exist_ok=True)

    configName = ''
    for key, value in configs.items():
        configName += '_' + str(key) + '=' + str(value)

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
        batch_size = configs['batch_size'],
        shuffle = True,
        #num_workers = 2,
        generator=torch.Generator(device='cuda')
        )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,     
        batch_size = configs['batch_size'],
        shuffle = False,
        #num_workers = 2,
        generator=torch.Generator(device='cuda')
        )

    # PCA analysis for original data
    # train_images = train_dataloader.dataset.data.detach().numpy().reshape(-1, 28*28)
    # train_labels = train_dataloader.dataset.targets.detach().numpy().reshape(-1, )
    # test_images = test_dataloader.dataset.data.detach().numpy().reshape(-1, 28*28)
    # test_labels = test_dataloader.dataset.targets.detach().numpy().reshape(-1, )
    
    # n_components = 2
    # pca = PCA(n_components=n_components)
    # pca.fit(train_images)
    # test_feature = pca.transform(test_images)

    # plot.plot_scatter(test_feature, test_labels, 'figure/pca/mnist' + configName)

# logging
    logFileName = common.RESULT_DIR_NAME + '/log/status' + configName + '.log'

    logging.basicConfig(filename=logFileName, level=logging.INFO, format=common.LOG_FORMAT)
    initialTime = time.time()
    logText = 'Start cluculation: ' + str(datetime.datetime.now())
    for key, value in configs.items():
        logText += ', ' + str(key) + ': ' + str(value)
    logText += ', device: ' + device
    logging.info(logText)

    trainingFileName = common.RESULT_DIR_NAME + '/training' + configName + '.txt'
    bestResult = 0
    bestLoss = 1e6

    #----------------------------------------------------------
    # make neural net
    model = lorenz96.ODEBlock(image_size, label_size, systemCount, configs).to(device)
    print(f"~~~~Start training a new model with F = {configs['f']} iteration = {configs['iteration']} ~~~~")
    
    # save the weight
    # weightFileName = 'result/weight/model_weights_before' + configName + '.pth'
    # torch.save(model.state_dict(), weightFileName)
    #np.savetxt("result/weight/weight_fc1_before.txt", model.fc1.weight.cpu().detach().numpy())
    #np.savetxt("result/weight/weight_fc2_before.txt", model.fc2.weight.cpu().detach().numpy())

    #torch.save(model.state_dict(), 'model_weights_before.pth')

    #----------------------------------------------------------
    # make loss function
    criterion = nn.CrossEntropyLoss() 

    #----------------------------------------------------------
    # make optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate']) 

    #----------------------------------------------------------

#training
    for epoch in range(configs['num_epochs']): 
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
            initialStates = np.concatenate([initialStates, model.initialStates.cpu().detach().numpy()])
            finalStates = np.concatenate([finalStates, model.finalStates.cpu().detach().numpy()])

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

        initialStates = np.empty([0,systemCount])
        finalStates = np.empty([0,systemCount])
        pcaLabels = np.empty([0])

        #----------------------------------------------------------
        # evaluate
        model.eval()  # change model mode to the evaluation

        loss_sum = 0
        correct = 0
        count = 0

        # first = True
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
                #     stateFigureFileName = 'figure/colormap/' + configName + '_epoch=' + str(epoch)
                #     model.plotStates(stateFigureFileName)
                #     first = False
                # else:
                outputs = model(inputs)

                #save initial and final states and labels for PCA
                # initialStates = np.concatenate([initialStates, model.initialStates.cpu().detach().numpy()])
                # finalStates = np.concatenate([finalStates, model.finalStates.cpu().detach().numpy()])
                # pcaLabels = np.concatenate([pcaLabels, labels.cpu().detach().numpy()]) 

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

        #PCA for test data
        # initialStates = pca_initialStates.transform(initialStates)
        # finalStates = pca_finalStates.transform(finalStates)
        # mnistPcaFileName = 'figure/pca/initialStates' + configName + '_epoch='+ str(epoch)
        # plot.plot_scatter(initialStates, pcaLabels, mnistPcaFileName)
        # mnistPcaFileName = 'figure/pca/finalStates' + configName + '_epoch='+ str(epoch)
        # plot.plot_scatter(finalStates, pcaLabels, mnistPcaFileName)

        print("==>>> epoch: {}, test loss: {:.6f}, acc: {:.5f}".format(epoch+1, loss_sum.item() / len(test_dataloader), correct * 1.0 / len(test_dataset)))
        # print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")
        with open(trainingFileName, 'a') as f_handle:
            f_handle.write(str(epoch+1) + ' ' + str(correct/len(test_dataset)) + ' ' + str(loss_sum.item() / len(test_dataloader)) + '\n')

        bestResult = max(correct/len(test_dataset), bestResult)
        bestLoss = min(loss_sum.item() / len(test_dataloader), bestLoss)

        fileName = 'figure/table/table' + configName + '_epoch='+ str(epoch)
        # plot.plot_mnistTable(y_test, y_pred, fileName)

        
    weightFileName = 'result/weight/model_weights_after'
    for key, value in configs.items():
       weightFileName += '_' + str(key) + '=' + str(value)
    weightFileName += '.pth'
    torch.save(model.state_dict(), weightFileName)
    np.savetxt("result/weight/weight_fc1_after.txt", model.fc1.weight.cpu().detach().numpy())
    np.savetxt("result/weight/weight_fc2_after.txt", model.fc2.weight.cpu().detach().numpy())

    fileName = common.RESULT_DIR_NAME + '/accuracy.txt'
    with open(fileName, 'a') as f_handle:
        for key, value in configs.items(): f_handle.write(str(value) + ' ')
        f_handle.write(str(bestResult) + ' ' + str(bestLoss) + '\n')

    logging.info('total time: ' + str(time.time() - initialTime))