import os
import functools
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.multiprocessing as multiprocessing
from multiprocessing import Pool

import lorenz96_neuralODE_PCA

PROCESSE_COUNT = 1
JOB_COUNT = 8

scaleFactor = 1.5399265

def iterateRun(args):
    iter_step = args.iteration_step
    total_iterations = args.total_iterations
    iterations = np.around(np.arange(iter_step, total_iterations+iter_step, iter_step), 2)

    for iteration in iterations:
        configs = {}
        configs['num_epochs'] = args.epochs
        configs['learning_rate'] = args.lr
        configs['batch_size'] = args.batch_size
        configs['f'] = args.f
        configs['iteration'] = iteration
        # print(configs)
        n = int(args.f/0.25)-1

        with torch.cuda.device(n % torch.cuda.device_count()):
            lorenz96_neuralODE_PCA.neuralOde(configs)
    
        
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # if multiprocessing.get_start_method() == 'fork':
    #     multiprocessing.set_start_method('spawn', force=True)    
    
    parser = argparse.ArgumentParser(description='Lorenz96 setting parser')
    parser.add_argument('--epochs', type=int, default=20, help='# of epochs to train')
    parser.add_argument('--batch_size', type=int, default=500, help='# of batches on datasets')
    parser.add_argument('--f', type=float, help='externel F')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n', type=int, default=5, help='number of repeat')
    parser.add_argument('--total_iterations', type=float, default=5.0, help='total # of iterations')
    parser.add_argument('--iteration_step', type=float, default=0.2, help='step length of iterations')
    args = parser.parse_args()

    for i in range(args.n):
        print(f"~~~~run idex={i}~~~~")
        iterateRun(args)
