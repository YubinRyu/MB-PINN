import torch
from utils import net_1, net_2

def predict_1(train, device, model, X_lb, X_ub):
    x = torch.tensor(train[:, 0:1], requires_grad=True).float().to(device)
    t = torch.tensor(train[:, 1:2], requires_grad=True).float().to(device)
    mu = torch.tensor(train[:, 2:3], requires_grad=True).float().to(device)
    q0 = torch.tensor(train[:, 3:4], requires_grad=True).float().to(device)

    model.eval()
    output = net_1(x, t, mu, q0, model, X_lb, X_ub)
    output = output.detach().cpu().numpy()

    return output

def predict_2(train, device, model, X_lb, X_ub):
    t = torch.tensor(train[:, 1:2], requires_grad=True).float().to(device)
    mu = torch.tensor(train[:, 2:3], requires_grad=True).float().to(device)
    q0 = torch.tensor(train[:, 3:4], requires_grad=True).float().to(device)

    model.eval()
    output = net_2(t, mu, q0, model, X_lb, X_ub)
    output = output.detach().cpu().numpy()

    return output
