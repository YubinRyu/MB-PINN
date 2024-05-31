import math
import scipy.io
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.interpolate import griddata

# Parameters
H = 60
E = 4.2e9
nu = 0.2
Cl = 6e-5

class DataSampler:
    # Initialize the class
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]

    def sample(self, batch_size):
        idx = np.random.choice(self.N, batch_size, replace=False)
        X_batch = self.X[idx, :]
        Y_batch = self.Y[idx, :]

        return X_batch, Y_batch

class DNN1(torch.nn.Module):
    def __init__(self, layers):
        super(DNN1, self).__init__()

        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh

        layer_list = list()

        # Neural Network
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))

        layerdict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerdict)

    def forward(self, x, lb, ub):
        z = 2.0 * (x - lb) / (ub - lb) - 1.0

        out = self.layers(z)

        return abs(out)

class DNN2(torch.nn.Module):
    def __init__(self, layers):
        super(DNN2, self).__init__()

        self.depth = len(layers) - 1
        self.activation = torch.nn.Sigmoid

        layer_list = list()

        # Neural Network
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))

        layerdict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerdict)

    def forward(self, x, lb, ub):
        z = 2.0 * (x - lb[1:4]) / (ub[1:4] - lb[1:4]) - 1.0

        out = self.layers(z)

        return abs(out)

def net_1(x, t, mu, q0, network, X_lb, X_ub):
    output = network(torch.cat([x, t, mu, q0], dim=1).float(), X_lb.float(), X_ub.float())

    return output

def net_2(t, mu, q0, network, X_lb, X_ub):
    output = network(torch.cat([t, mu, q0], dim=1).float(), X_lb.float(), X_ub.float())

    return output

def net_pde(x, t, mu, q0, network1, X_lb, X_ub):
    w = net_1(x, t, mu, q0, network1, X_lb, X_ub)

    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_t = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), retain_graph=True, create_graph=True)[0]

    pde = (np.pi * H) / 4 * w_t - (np.pi * E) / (128 * mu * (1 - nu ** 2)) * ((3 * w ** 2 * (w_x ** 2) + (w ** 3) * w_xx)) + H * 2 * Cl / (t + 50) ** 0.5

    return pde

def net_ic(x, t, mu, q0, network1, X_lb, X_ub):
    w = net_1(x, t, mu, q0, network1, X_lb, X_ub)
    ic = w

    return ic

def net_bc(x, t, mu, q0, network1, X_lb, X_ub):
    w = net_1(x, t, mu, q0, network1, X_lb, X_ub)

    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), retain_graph=True, create_graph=True)[0]

    bc = - (np.pi * E) / (128 * mu * (1 - nu ** 2)) * w_x * (w ** 3) - q0

    return bc

def net_mb(x, t, mu, q0, network1, network2, X_lb, X_ub):
    l = net_2(t, mu, q0, network2, X_lb, X_ub)

    data = torch.cat([x, t, l], dim=1)
    data_bool = data[:, 0:1] >= data[:, 2:3]

    condition = torch.all(data_bool, dim=1)

    mb = net_1(l, t, mu, q0, network1, X_lb, X_ub)[condition]

    return mb

def train(optimizer1, optimizer2, model1, model2, data_loader, device, X_lb, X_ub, X_PDE, X_IC, X_BC, X_MB, X_net2, Y_net2):
    model1.train()
    model2.train()

    running_train_loss = 0.0
    running_1_loss = 0.0
    running_2_loss = 0.0
    running_3_loss = 0.0
    running_4_loss = 0.0
    running_5_loss = 0.0

    for train_batch, target_batch in data_loader:
        loss_for_net1, loss1, loss2, loss3, loss4, loss5 = train_loss_net1(train_batch, target_batch, device, model1, model2, X_lb, X_ub, X_PDE, X_IC, X_BC, X_MB)

        optimizer1.zero_grad()
        loss_for_net1.backward(retain_graph=True)
        optimizer1.step()

        loss_for_net2 = train_loss_net2(device, model1, model2, X_lb, X_ub, X_MB, X_net2, Y_net2)

        optimizer2.zero_grad()
        loss_for_net2.backward(retain_graph=True)
        optimizer2.step()

        running_train_loss += loss_for_net1
        running_1_loss += loss1
        running_2_loss += loss2
        running_3_loss += loss3
        running_4_loss += loss4
        running_5_loss += loss5

    length = len(data_loader)
    train_loss_value = running_train_loss / length
    loss1 = running_1_loss / length
    loss2 = running_2_loss / length
    loss3 = running_3_loss / length
    loss4 = running_4_loss / length
    loss5 = running_5_loss / length

    return train_loss_value, loss1, loss2, loss3, loss4, loss5

def train_loss_net1(train_batch, target_batch, device, network1, network2, X_lb, X_ub, X_PDE, X_IC, X_BC, X_MB):
    x = torch.tensor(train_batch[:, 0:1], requires_grad=True).float().to(device)
    t = torch.tensor(train_batch[:, 1:2], requires_grad=True).float().to(device)
    mu = torch.tensor(train_batch[:, 2:3], requires_grad=True).float().to(device)
    q0 = torch.tensor(train_batch[:, 3:4], requires_grad=True).float().to(device)

    w = torch.tensor(target_batch[:, 0:1], requires_grad=True).float().to(device)

    x_pde = torch.tensor(X_PDE[:, 0:1], requires_grad=True).float().to(device)
    t_pde = torch.tensor(X_PDE[:, 1:2], requires_grad=True).float().to(device)
    mu_pde = torch.tensor(X_PDE[:, 2:3], requires_grad=True).float().to(device)
    q0_pde = torch.tensor(X_PDE[:, 3:4], requires_grad=True).float().to(device)

    x_ic = torch.tensor(X_IC[:, 0:1], requires_grad=True).float().to(device)
    t_ic = torch.tensor(X_IC[:, 1:2], requires_grad=True).float().to(device)
    mu_ic = torch.tensor(X_IC[:, 2:3], requires_grad=True).float().to(device)
    q0_ic = torch.tensor(X_IC[:, 3:4], requires_grad=True).float().to(device)

    x_bc = torch.tensor(X_BC[:, 0:1], requires_grad=True).float().to(device)
    t_bc = torch.tensor(X_BC[:, 1:2], requires_grad=True).float().to(device)
    mu_bc = torch.tensor(X_BC[:, 2:3], requires_grad=True).float().to(device)
    q0_bc = torch.tensor(X_BC[:, 3:4], requires_grad=True).float().to(device)

    x_mb = torch.tensor(X_MB[:, 0:1], requires_grad=True).float().to(device)
    t_mb = torch.tensor(X_MB[:, 1:2], requires_grad=True).float().to(device)
    mu_mb = torch.tensor(X_MB[:, 2:3], requires_grad=True).float().to(device)
    q0_mb = torch.tensor(X_MB[:, 3:4], requires_grad=True).float().to(device)

    output1 = net_1(x, t, mu, q0, network1, X_lb, X_ub)
    pde = net_pde(x_pde, t_pde, mu_pde, q0_pde, network1, X_lb, X_ub)
    ic = net_ic(x_ic, t_ic, mu_ic, q0_ic, network1, X_lb, X_ub)
    bc = net_bc(x_bc, t_bc, mu_bc, q0_bc, network1, X_lb, X_ub)
    mb = net_mb(x_mb, t_mb, mu_mb, q0_mb, network1, network2, X_lb, X_ub)

    # Empirical Loss
    loss1 = torch.mean(torch.abs(w - output1))

    # PDE Residual
    loss2 = torch.mean(torch.abs(pde))

    if math.isnan(loss2) == True:
        loss2 = torch.tensor(0).float().to(device)
    else:
        loss2 = loss2

    # IC Residual
    loss3 = torch.mean(torch.abs(ic))

    if math.isnan(loss3) == True:
        loss3 = torch.tensor(0).float().to(device)
    else:
        loss3 = loss3

    # BC Residual
    loss4 = torch.mean(torch.abs(bc))

    if math.isnan(loss4) == True:
        loss4 = torch.tensor(0).float().to(device)
    else:
        loss4 = loss4

    # Moving Boundary Residual
    loss5 = torch.mean(torch.abs(mb))

    if math.isnan(loss5) == True:
        loss5 = torch.tensor(0).float().to(device)
    else:
        loss5 = loss5

    loss = loss1

    return loss, loss1, loss2, loss3, loss4, loss5

def train_loss_net2(device, network1, network2, X_lb, X_ub, X_MB, X_net2, Y_net2):
    x_mb = torch.tensor(X_MB[:, 0:1], requires_grad=True).float().to(device)
    t_mb = torch.tensor(X_MB[:, 1:2], requires_grad=True).float().to(device)
    mu_mb = torch.tensor(X_MB[:, 2:3], requires_grad=True).float().to(device)
    q0_mb = torch.tensor(X_MB[:, 3:4], requires_grad=True).float().to(device)

    t_mb_0 = torch.tensor(X_net2[:, 0:1], requires_grad=True).float().to(device)
    mu_mb_0 = torch.tensor(X_net2[:, 1:2], requires_grad=True).float().to(device)
    q0_mb_0 = torch.tensor(X_net2[:, 2:3], requires_grad=True).float().to(device)
    l_mb_0 = torch.tensor(Y_net2, requires_grad=True).float().to(device)

    t_mb_0 = torch.tensor(t_mb_0, dtype=torch.float64)
    mu_mb_0 = torch.tensor(mu_mb_0, dtype=torch.float64)
    q0_mb_0 = torch.tensor(q0_mb_0, dtype=torch.float64)
    l_mb_0 = torch.tensor(l_mb_0, dtype=torch.float64)

    t_mb_0 = torch.tensor(t_mb_0, requires_grad=True).float().to(device)
    mu_mb_0 = torch.tensor(mu_mb_0, requires_grad=True).float().to(device)
    q0_mb_0 = torch.tensor(q0_mb_0, requires_grad=True).float().to(device)
    l_mb_0 = torch.tensor(l_mb_0, requires_grad=True).float().to(device)

    mb = net_mb(x_mb, t_mb, mu_mb, q0_mb, network1, network2, X_lb, X_ub)
    mb_ = net_2(t_mb_0, mu_mb_0, q0_mb_0, network2, X_lb, X_ub)

    # Moving Boundary Residual
    loss5 = torch.mean(torch.abs(mb))

    # Empirical Loss
    loss6 = torch.mean(torch.abs(l_mb_0.unsqueeze(-1) - mb_))

    if math.isnan(loss5) == True:
        loss5 = torch.tensor(0).float().to(device)
    else:
        loss5 = loss5

    loss = 0

    return loss

def validation_loss(X_val, Y_val, device, network1, network2, X_lb, X_ub, X_PDE, X_IC, X_BC, X_MB):
    x = torch.tensor(X_val[:, 0:1], requires_grad=True).float().to(device)
    t = torch.tensor(X_val[:, 1:2], requires_grad=True).float().to(device)
    mu = torch.tensor(X_val[:, 2:3], requires_grad=True).float().to(device)
    q0 = torch.tensor(X_val[:, 3:4], requires_grad=True).float().to(device)

    w = torch.tensor(Y_val[:, 0:1], requires_grad=True).float().to(device)

    x_pde = torch.tensor(X_PDE[:, 0:1], requires_grad=True).float().to(device)
    t_pde = torch.tensor(X_PDE[:, 1:2], requires_grad=True).float().to(device)
    mu_pde = torch.tensor(X_PDE[:, 2:3], requires_grad=True).float().to(device)
    q0_pde = torch.tensor(X_PDE[:, 3:4], requires_grad=True).float().to(device)

    x_ic = torch.tensor(X_IC[:, 0:1], requires_grad=True).float().to(device)
    t_ic = torch.tensor(X_IC[:, 1:2], requires_grad=True).float().to(device)
    mu_ic = torch.tensor(X_IC[:, 2:3], requires_grad=True).float().to(device)
    q0_ic = torch.tensor(X_IC[:, 3:4], requires_grad=True).float().to(device)

    x_bc = torch.tensor(X_BC[:, 0:1], requires_grad=True).float().to(device)
    t_bc = torch.tensor(X_BC[:, 1:2], requires_grad=True).float().to(device)
    mu_bc = torch.tensor(X_BC[:, 2:3], requires_grad=True).float().to(device)
    q0_bc = torch.tensor(X_BC[:, 3:4], requires_grad=True).float().to(device)

    x_mb = torch.tensor(X_MB[:, 0:1], requires_grad=True).float().to(device)
    t_mb = torch.tensor(X_MB[:, 1:2], requires_grad=True).float().to(device)
    mu_mb = torch.tensor(X_MB[:, 2:3], requires_grad=True).float().to(device)
    q0_mb = torch.tensor(X_MB[:, 3:4], requires_grad=True).float().to(device)

    output1 = net_1(x, t, mu, q0, network1, X_lb, X_ub)
    pde = net_pde(x_pde, t_pde, mu_pde, q0_pde, network1, X_lb, X_ub)
    ic = net_ic(x_ic, t_ic, mu_ic, q0_ic, network1, X_lb, X_ub)
    bc = net_bc(x_bc, t_bc, mu_bc, q0_bc, network1, X_lb, X_ub)
    mb = net_mb(x_mb, t_mb, mu_mb, q0_mb, network1, network2, X_lb, X_ub)

    # Empirical Loss
    loss1 = torch.mean(torch.abs(w - output1))

    # PDE Residual
    loss2 = torch.mean(torch.abs(pde))

    if math.isnan(loss2) == True:
        loss2 = torch.tensor(0).float().to(device)
    else:
        loss2 = loss2

    # IC Residual
    loss3 = torch.mean(torch.abs(ic))

    if math.isnan(loss3) == True:
        loss3 = torch.tensor(0).float().to(device)
    else:
        loss3 = loss3

    # BC Residual
    loss4 = torch.mean(torch.abs(bc))

    if math.isnan(loss4) == True:
        loss4 = torch.tensor(0).float().to(device)
    else:
        loss4 = loss4

    # Moving Boundary Residual
    loss5 = torch.mean(torch.abs(mb))

    if math.isnan(loss5) == True:
        loss5 = torch.tensor(0).float().to(device)
    else:
        loss5 = loss5

    val_loss = loss1

    return val_loss
