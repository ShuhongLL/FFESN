import os
import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torchdiffeq import odeint_adjoint as odeint

import lorenz96
import ftmle

os.makedirs('./result', exist_ok=True)
os.makedirs('./result/ftmle', exist_ok=True)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(seed)

num_epochs = 20 # number of epochs
num_batch = 500 # number of batch
learning_rate = 0.001 # learning rate
image_size = 28*28 # size of image
label_size = 10
samplingInterval = 0.05
stepSize = 0.01

systemCount = 500
iterationSize = 5 # change this to your interested iteration time!

device = 'cuda:0'
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    '../data',
    train = True,
    download = True,
    transform = transform
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

def eval_state(model):
    initialStates = np.empty([0,systemCount])
    finalStates = np.empty([0,systemCount])
    class_labels = np.empty([0])
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.view(-1, image_size) 
            outputs = model(inputs)

            initialStates = np.concatenate([initialStates, model.initialStates.cpu().detach().numpy()])
            finalStates = np.concatenate([finalStates, model.finalStates.cpu().detach().numpy()])
            class_labels = np.concatenate([class_labels, labels.cpu().detach().numpy()]) 

    return initialStates, finalStates, class_labels

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
    
fs = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
iterations = [iterationSize] * len(fs)
# fs = np.repeat(fs, 2)

batch_size = 2048

for f, iteration in zip(fs, iterations):
    if os.path.exists(f'./result/ftmle/ftmle_f{f}_t{iteration}.npy'):
        continue
    print(f'~~~~ f = {f}, iteration = {iteration} ~~~~')
    model = lorenz96.ODEBlock(image_size, label_size, systemCount, {"f": f, "iteration": iteration})
    model.load_state_dict(torch.load(f'./save_models/lorenz_f{f}_t{iteration}.pth'))
    model = model.to(device)
    initialStates, finalStates, labels = eval_state(model)

    ode_layer = Lorenz96Integrator(lorenz96.Lorenz96_ODEfunc(f), torch.tensor([0.0, iteration]))

    ftmle_iteration_layers = []
    for i in tqdm(range(0, len(initialStates), batch_size)):
        end = min(i+batch_size, len(initialStates))
        input_data = torch.from_numpy(initialStates[i:end])
        # exp, ftmle = ftmle.compute_jacob_ftmle(ode_layer, input_data, iteration, device)
        exp, ftmle = ftmle.compute_jacob_ftmle_low_mem(ode_layer, input_data, iteration, device, chunk_size=1024, random_svd=True, k=10)

        ftmle_iteration_layers.append(ftmle)
    ftmle_iteration_layers = np.concatenate(ftmle_iteration_layers, axis=0)
    print(f, max(ftmle_iteration_layers), min(ftmle_iteration_layers))
    np.save(f'./result/ftmle/ftmle_f{f}_t{iteration}.npy', ftmle_iteration_layers)

    del initialStates, finalStates, labels

