import os
import warnings
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import csv
import random
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.esn import DESN
from model.util import EarlyStopper

warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.RandomState(seed=seed)
    return rng

def train_and_test(train_loader, test_loader, epochs, dim_reservoir, rho, iteration, lr, rng,
                   device_ids, save=False):
    print(f"~~~~Start training a new model with rho = {rho} iteration = {iteration} ~~~~")
    model = DESN(dim_reservoir=dim_reservoir, dim_u=dim_u, dim_y=dim_y,
                 iteration=iteration, rho=rho, rng=rng, density=0.5)
    device = torch.device(f"cuda:{device_ids[0]}")
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    

    # early_stopper = EarlyStopper(patience=5, min_delta=0)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    avg_losses = []
    avg_accuracy = []
    for epoch in range(epochs):
        # trainning
        avg_train_loss = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x).to(device), Variable(target).to(device)
            x = torch.flatten(x, start_dim=2)
            out = model(x)
            out = torch.squeeze(out)
            loss = criterion(out, target)
            avg_train_loss = avg_train_loss * 0.9 + loss.data * 0.1
            loss.backward()
            optimizer.step()

        # testing
        with torch.no_grad():
            correct_cnt, total_loss = 0, 0
            total_cnt = 0
            for batch_idx, (x, target) in enumerate(test_loader):
                x, target = Variable(x, volatile=True).to(device), Variable(target, volatile=True).to(device)
                x = torch.flatten(x, start_dim=2)
                out = model(x)
                out = torch.squeeze(out)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                total_cnt += x.data.size()[0]
                correct_cnt += (pred_label == target.data).sum()

                total_loss += loss.item()
                
            avg_loss = total_loss / len(test_loader)
            avg_losses.append(avg_loss)
            accuracy = correct_cnt * 1.0 / total_cnt
            avg_accuracy.append(accuracy)
            print("==>>> epoch: {}, test loss: {:.6f}, acc: {:.5f}".format(epoch+1, avg_loss, accuracy))
            # if early_stopper.early_stop(ave_loss):             
            #     break
    avg_losses = avg_losses + [0] * (epochs - len(avg_losses))
    avg_losses = np.array(avg_losses)

    if save:
        # save model
        weightFileName = f'./save_models/ffesn_rho{rho}_t{iteration}.pth'
        torch.save(model.state_dict(), weightFileName)
        # save results
        avg_losses = np.array(avg_losses)
        avg_accuracy = np.array(avg_accuracy)

        np.save(f'./result/ffesn_loss_rho{rho}_iter{iteration}.npy', avg_losses)
        np.save(f'./result/ffesn_accur_rho{rho}_iter{iteration}.npy', avg_accuracy)

    return model, avg_losses

"""
Compute colormaps.
"""

if __name__ == '__main__':
    # parse cmd arguments.
    parser = argparse.ArgumentParser(description='DESN setting parser')
    parser.add_argument('--epochs', type=int, default=80, help='# of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='# of batches on datasets')
    parser.add_argument('--rho', type=float, help='spectral radius')
    parser.add_argument('--iteration', type=int, help='iteration number')
    parser.add_argument('--n', type=int, default=1, help='number of repeat')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset')
    parser.add_argument('--save', type=lambda x: (str(x).lower() == 'true'), default=True, help='Save the output (true or false)')
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='device ids')

    args = parser.parse_args()
    g_epochs = args.epochs
    g_batch_size = args.batch_size
    g_rho = args.rho
    g_n = args.n
    g_device_id = args.device
    g_lr = args.lr

    print("<=========Model=Setting=============>")
    print(f'Number of EPOCH = {g_epochs}')
    print(f'BATCH size = {g_batch_size}')
    print(f'Rho = {g_rho}')
    print(f'Device ids = {g_device_id}')

    """
    Hyper-parameters.
    """
    dim_u = 28 * 28
    dim_y = 10
    dim_x = 500

    root = '../data'
    device = torch.device("cuda")
    if not os.path.exists(root):
        os.mkdir(root)

    rng = set_seed(args.seed)
    os.makedirs('./save_models', exist_ok=True)
    os.makedirs('./result', exist_ok=True)
    
    # rhos = np.linspace(0.1, 2.0, 20).astype(float)
    # rhos = np.round(rhos, 2)
    Ts = np.linspace(1, 30, 30).astype(int)

    """
    Load datasets.
    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    if args.dataset == 'mnist':
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
    else:
        train_set = dset.FashionMNIST(root=root, train=True, transform=trans, download=True)
        test_set = dset.FashionMNIST(root=root, train=False, transform=trans, download=True)

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=g_batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=g_batch_size,
                    shuffle=False)

    print("Total trainning batch number: {}".format(len(train_loader)))
    print("Total testing batch number: {}".format(len(test_loader)))
    print()

    # for i in range(g_n):
    # print(f"~~~~run idex={i}~~~~")
    _, loss = train_and_test(train_loader, test_loader, g_epochs, dim_reservoir=dim_x, rho=g_rho,
                             iteration=iteration, lr=g_lr, rng=rng, device_ids=g_device_id, save=args.save)
