import os
import sys
import time
import warnings

from utils import *
from data_preprocessing import data_load
from evaluate import predict_1, predict_2

warnings.filterwarnings('ignore')

# Directory Setting
dir = os.path.dirname(os.path.abspath(__file__))

# Device Setting
device = torch.cuda.current_device()
torch.cuda.set_device(device)

X_star, Y_star, train_loader, X_lb, X_ub, X_val, Y_val, X_PDE, X_IC, X_BC, X_MB, x, t, X_net2, Y_net2 = data_load(device)

# Model Setting
model1 = DNN1([4, 100, 100, 100, 1])
model2 = DNN2([3, 100, 100, 1])

model1.cuda(device)
model2.cuda(device)

# Train Setting
best_epoch = 1
start_epoch = 1
last_epoch = 10000

ti = time.time()
trigger_times = 0

patience = 5000
best_val_loss = 100000.0

optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.0001)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0001)

scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min', factor=0.8, patience=200, verbose=False, min_lr=1e-6)
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', factor=0.8, patience=200, verbose=False, min_lr=1e-6)

# List for Loss Monitoring
loss_list, val_loss_list = [], []
w_loss_list, pde_loss_list, ic_loss_list, bc_loss_list, mb_loss_list = [], [], [], [], []

# Start Training
sys.stdout.flush()

for epoch in range(start_epoch, last_epoch+1):

    loss, loss1, loss2, loss3, loss4, loss5 = train(optimizer1, optimizer2, model1, model2, train_loader, device, X_lb, X_ub, X_PDE, X_IC, X_BC, X_MB, X_net2, Y_net2)

    if epoch % 1 == 0:
        val_loss = validation_loss(X_val, Y_val, device, model1, model2, X_lb, X_ub, X_PDE, X_IC, X_BC, X_MB)
        scheduler1.step(val_loss)
        scheduler2.step(val_loss)

        print('Iteration: %d, Loss: %.3e, Val Loss (1): %.3e, Learning Rate 1: %.3e, Learning Rate 2: %.3e, Best Val Epoch : %d, Time: %.4f'
              % (epoch, loss.item(), val_loss.item(), optimizer1.param_groups[0]['lr'], optimizer2.param_groups[0]['lr'], best_epoch, time.time() - ti))
        print('w: %.3e, pde: %.3e, ic: %.3e, bc: %.3e, mb: %.3e'
              % (loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()))

        loss_list.append(loss.item())
        val_loss_list.append(val_loss.item())

        w_loss_list.append(loss1.item())
        pde_loss_list.append(loss2.item())
        ic_loss_list.append(loss3.item())
        bc_loss_list.append(loss4.item())
        mb_loss_list.append(loss5.item())

        # Saving Best Model - Network 1 (Main)
        if torch.cuda.current_device() == 0:
            if val_loss.item() < best_val_loss:

                best_val_loss = val_loss.item()
                best_epoch = epoch

                print('Best Validation Loss: ', best_val_loss)
                print('Saving Best Model...')

                CHECKPOINT_PATH = dir + '/best_model1.pth'
                torch.save(model1.state_dict(), CHECKPOINT_PATH)

                CHECKPOINT_PATH = dir + '/best_model2.pth'
                torch.save(model2.state_dict(), CHECKPOINT_PATH)

                print('Trigger Times : 0')
                trigger_times = 0

                # Save checkpoint
                torch.save({'epoch': epoch,
                            'loss': loss, 'val_loss': val_loss,
                            'best_epoch': best_epoch, 'trigger_times': trigger_times,
                            'loss_history': loss_list,
                            'val_loss_1_history': val_loss_list,
                            'w_loss_history': w_loss_list, 'pde_loss_history': pde_loss_list,
                            'ic_loss_history': ic_loss_list, 'bc_loss_history': bc_loss_list,
                            'mb_loss_history': mb_loss_list},
                            dir + '/checkpoint.pt')

            else:
                trigger_times += 1
                print('Trigger Times :', trigger_times)

                if trigger_times >= patience:
                    print('Early Stopping!')
                    break
