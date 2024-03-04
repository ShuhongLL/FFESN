import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

import plot

stepSize = 0.01

class Lorenz96:
    def __init__(self, f):
        self.f = f

    def differential(self, x, time):
        return (torch.roll(x, -1, 1)-torch.roll(x, 2, 1))*torch.roll(x, +1, 1)-x + self.f

class Lorenz96_ODEfunc(nn.Module):
    def __init__(self, f):
        super(Lorenz96_ODEfunc, self).__init__()
        self.f = f

    def forward(self, t, x):
        return (torch.roll(x, -1, 1)-torch.roll(x, 2, 1))*torch.roll(x, +1, 1)-x + self.f
    

#----------------------------------------------------------
# Definition of neural net
class ODEBlock(nn.Module):
    def __init__(self, input_size, output_size, systemCount, config):
        super(ODEBlock, self).__init__()
        self.odefunc = Lorenz96_ODEfunc(config["f"])
        self.integration_time = torch.tensor([0.0, config["iteration"]])
        self.integration_time2 = torch.tensor(np.arange(0.0, config["iteration"] + stepSize, stepSize))

        self.fc1 = nn.Linear(input_size, systemCount)
        #self.fc1.weight.requires_grad = False
        self.fc2 = nn.Linear(systemCount, output_size)
        self.fileCount = 0

        self.plotStateCount = 100
        self.plotDataCount = 10

    def forward(self, x):
        #self.integration_time = self.integration_time.type_as(x)
        x = self.fc1(x)
        self.initialStates = x        
        if self.integration_time[-1] > 0:
            x = odeint(self.odefunc, x, self.integration_time, method='rk4', options=dict(step_size=stepSize))
            x = x[-1]
        self.finalStates = x
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def forward_plot(self, x):
        #self.integration_time = self.integration_time.type_as(x)
        x = self.fc1(x)
        self.initialStates = x
        #x = x.reshape(x.shape[0], systemCount)
        if self.integration_time2[-1] > 0:
            x = odeint(self.odefunc, x, self.integration_time2, method='rk4', options=dict(step_size=stepSize))
            self.states = x[:,:self.plotDataCount,:self.plotStateCount].cpu().detach().numpy()
            x = x[-1]
        #x = x.reshape(x.shape[0], systemCount)
        self.finalStates = x
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    #状態をファイル出力
    def plotStates(self, fileName):
        if self.integration_time2[-1] > 0:
            region = [0, self.integration_time[-1].cpu().detach().numpy(), 0, self.plotStateCount]            
            for i in range(self.plotDataCount):
                plot.plot_timeseriesColorMap(self.states[:,i,:], fileName + "_batch=" + str(i), region)
    