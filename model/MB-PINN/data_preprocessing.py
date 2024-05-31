import torch
import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

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

def data_load(device):
    n_data = 10000
    data = pd.read_csv('Net1_Data.csv')
    X_star = data[['x', 't', 'mu', 'q0']]
    Y_star = data['w']
    X_star = X_star.to_numpy()
    Y_star = Y_star.to_numpy()
    Y_star = np.reshape(Y_star, newshape=(Y_star.shape[0], 1))

    x = list(set(data['x'].to_numpy().flatten()))
    t = list(set(data['t'].to_numpy().flatten()))

    x = np.array(sorted(x))
    t = np.unique(np.round(np.array(sorted(t)), decimals=1))

    lb = X_star.min(0)
    ub = X_star.max(0)

    X, Y = DataSampler(X_star, Y_star).sample(n_data)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('Trainset Size : {}, Validation Size : {}'.format(X_train.shape[0], X_val.shape[0]))

    X_star = torch.tensor(X_star, dtype=torch.float64)
    Y_star = torch.tensor(Y_star, dtype=torch.float64)

    X_train = torch.tensor(X_train, dtype=torch.float64)
    Y_train = torch.tensor(Y_train, dtype=torch.float64)

    X_val = torch.tensor(X_val, dtype=torch.float64)
    Y_val = torch.tensor(Y_val, dtype=torch.float64)

    X_lb = torch.tensor(lb).to(device)
    X_ub = torch.tensor(ub).to(device)

    train_set = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Data Preprocessing for Network 2
    data = pd.read_csv('Net2_Data.csv')
    X_net2 = data[['t', 'mu', 'q0']]
    Y_net2 = data['l']
    X_net2 = X_net2.to_numpy()
    Y_net2 = Y_net2.to_numpy()

    # Data Preprocessing for PDE
    nn = 5
    xx = np.linspace(lb[0], ub[0], nn)
    tt = np.linspace(lb[1], ub[1], nn)
    mumu = np.linspace(lb[2], ub[2], nn)
    q0q0 = np.linspace(lb[3], ub[3], nn)

    XX, TT, MUMU, Q0Q0 = np.meshgrid(xx, tt, mumu, q0q0)
    X_PDE = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None], MUMU.flatten()[:, None], Q0Q0.flatten()[:, None]))  # (nn * nn, 2)
    X_PDE = torch.tensor(X_PDE).to(device)

    # Data Preprocessing for IC (t=0)
    nn = 5
    xx = np.linspace(lb[0], ub[0], nn)
    tt = np.linspace(lb[1], lb[1], nn)
    mumu = np.linspace(lb[2], ub[2], nn)
    q0q0 = np.linspace(lb[3], ub[3], nn)

    XX, TT, MUMU, Q0Q0 = np.meshgrid(xx, tt, mumu, q0q0)
    X_IC = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None], MUMU.flatten()[:, None], Q0Q0.flatten()[:, None]))  # (nn * nn, 2)
    X_IC = torch.tensor(X_IC).to(device)

    # Data Preprocessing for BC (x=0)
    nn = 5
    xx = np.linspace(lb[0], lb[0], nn)
    tt = np.linspace(lb[1], ub[1], nn)
    mumu = np.linspace(lb[2], ub[2], nn)
    q0q0 = np.linspace(lb[3], ub[3], nn)

    XX, TT, MUMU, Q0Q0 = np.meshgrid(xx, tt, mumu, q0q0)
    X_BC = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None], MUMU.flatten()[:, None], Q0Q0.flatten()[:, None]))  # (nn * nn, 2)
    X_BC = torch.tensor(X_BC).to(device)

    # Data Preprocessing for Moving Boundary
    nn = 5
    xx = np.linspace(lb[0], ub[0], nn)
    tt = np.linspace(lb[1], ub[1], nn)
    mumu = np.linspace(lb[2], ub[2], nn)
    q0q0 = np.linspace(lb[3], ub[3], nn)

    XX, TT, MUMU, Q0Q0 = np.meshgrid(xx, tt, mumu, q0q0)
    X_MB = np.hstack((XX.flatten()[:, None], TT.flatten()[:, None], MUMU.flatten()[:, None], Q0Q0.flatten()[:, None]))  # (nn * nn, 2)
    X_MB = torch.tensor(X_MB).to(device)

    return X_star, Y_star, train_loader, X_lb, X_ub, X_val, Y_val, X_PDE, X_IC, X_BC, X_MB, x, t, X_net2, Y_net2
