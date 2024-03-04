import torch
import numpy as np
import torch.nn as nn
import networkx as nx
from numpy.random import RandomState


class ESNLayer(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.w0 = nn.Parameter(weights, requires_grad=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x @ self.w0.t()
        x = self.tanh(x)
        return x


class LinearOut(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.output_dim = output_dim
        super(LinearOut, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, False)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class DESN(torch.nn.Module):
    def __init__(self, dim_reservoir, dim_u, dim_y, iteration,
                 rho, rng, density=1.0, seed=42):
        super(DESN, self).__init__()
        self.rng = rng
        self.seed = seed
        weight = self.init_weight(dim_reservoir, rho, self.rng, density)
        self.input = torch.nn.Linear(dim_u, dim_reservoir, False)
        self.initial_weight = weight
        self.ESNs = []
        for _ in range(iteration):
            self.ESNs.append(ESNLayer(weight))
        self.ESNs = nn.ModuleList(self.ESNs)
        self.output = LinearOut(dim_reservoir, dim_y)

    def init_weight(self, dim_x, rho, rng, density=1.0):
        """
        init fixed intrinsic weight in ESN
        """
        dtype = np.float32
        num_edges = int(dim_x * (dim_x - 1) * density / 2)
        G = nx.gnm_random_graph(dim_x, num_edges, self.seed)
        w_net = np.array(nx.to_numpy_array(G))
        w_net = w_net.astype(dtype)
        w_net *= rng.uniform(-1.0, 1.0, (dim_x, dim_x))

        # Compute the spectral radius = the maximum in absolute value of eigenvalue
        eigv_list = np.linalg.eig(w_net)[0]
        spectral_radius = np.max(np.abs(eigv_list))

        # Normalize the spectral radius and specify to the predefined value (rho)
        w_net *= rho / spectral_radius
        return torch.from_numpy(w_net)

    def forward(self, x):
        x = self.input(x)
        self.initial_states = x
        for i in range(len(self.ESNs)):
            x = self.ESNs[i](x)
        self.final_states = x
        x = self.output(x)
        return x

    def get_init_weight(self):
        return self.initial_weight
