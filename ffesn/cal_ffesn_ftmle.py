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

import ftmle
from model.esn import DESN

os.makedirs('./result', exist_ok=True)
os.makedirs('./result/ftmle', exist_ok=True)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(seed)
rng = np.random.RandomState(seed=seed)

num_batch = 500 # number of batch
image_size = 28*28 # size of image
label_size = 10

systemCount = 500
iterationSize = 30  # change this to your interested iteration value!

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

            initial_states = np.concatenate([initialStates, model.initial_states.cpu().detach().numpy()])
            final_states = np.concatenate([finalStates, model.final_states.cpu().detach().numpy()])
            class_labels = np.concatenate([class_labels, labels.cpu().detach().numpy()]) 

    return initial_states, final_states, class_labels
    
rhos = [round(0.1 * i, 2) for i in range(1, 21)]
iterations = [iterationSize] * len(rhos)
# fs = np.repeat(fs, 2)

batch_size = 1024

for rho, iteration in zip(rhos, iterations):
    if os.path.exists(f'./result/ftmle/ftmle_f{rho}_t{iteration}.npy'):
        continue
    print(f'~~~~ f = {rho}, iteration = {iteration} ~~~~')
    model = DESN(dim_reservoir=systemCount, dim_u=image_size, dim_y=label_size,
                 iteration=iteration, rho=rho, rng=rng, density=0.5)
    model.load_state_dict(torch.load(f'./save_models/ffesn_rho{rho}_t{iteration}.pth'))
    model = model.to(device)
    initial_states, final_states, labels = eval_state(model)

    esn_layer = model.ESNs

    ftmle_iteration_layers = []
    for i in tqdm(range(0, len(initial_states), batch_size)):
        end = min(i+batch_size, len(initial_states))
        input_data = torch.from_numpy(initial_states[i:end])
        exp, ftmle = ftmle.compute_jacob_ftmle(esn_layer, input_data, iteration, device)
        # exp, ftmle = ftmle.compute_jacob_ftmle_low_mem(ode_layer, input_data, iteration, device, chunk_size=1024, random_svd=True, k=10)

        ftmle_iteration_layers.append(ftmle)
    ftmle_iteration_layers = np.concatenate(ftmle_iteration_layers, axis=0)
    print(rho, max(ftmle_iteration_layers), min(ftmle_iteration_layers))
    np.save(f'./result/ftmle/ftmle_rho{rho}_t{iteration}.npy', ftmle_iteration_layers)

    del initial_states, final_states, labels
