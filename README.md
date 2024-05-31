# MB-PINN
We propose a physics-informed neural network with moving boundary constraint (MB-PINN) to model hydraulic fracture propagation under various operating conditions. We utilized two independent neural networks to approximate the latent solution and the moving boundary. This approach enables the model to capture the intricate dynamics of the moving boundary problem. As a result, the proposed model shows superior prediction accuracy not only within the interpolation range, but also in extrapolation scenarios. This exceptional performance highlights its potential as a surrogate model for optimizatin hydraulic fracturing operations in oil and gas reservoirs. 

## Installation
Run the following command to set up.

    git clone https://github.com/YubinRyu/MB-PINN.git
    
## Data availability
The additional data that support the findings in this study is available upon reasonable request to the corresponding authors. 

## Model
Run the following command to train PINN-Ex. 

    python model/PINN/main_ddp.py
