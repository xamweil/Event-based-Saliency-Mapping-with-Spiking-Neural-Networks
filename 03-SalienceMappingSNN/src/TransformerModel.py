import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.checkpoint import checkpoint
class SpikeWrapper(nn.Module):
    def __init__(self, spiking_layer):
        super(SpikeWrapper, self).__init__()
        self.spiking_layer = spiking_layer
        self.mem = None


    def forward(self, x):
        spikes, self.mem = self.spiking_layer(x)
        return spikes

    def reset(self):
        # Call the reset method of the spiking layer to clear its internal state
        #if hasattr(self.spiking_layer, 'mem_reset'):
        if self.mem is not None:
            self.spiking_layer.reset_mem()



class Net(nn.Module):
    def __init__(self, device=None):
        super(Net, self).__init__()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")



        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=10),
            SpikeWrapper(snn.Leaky(beta=0.9, threshold=0.5,
                           learn_beta=True, learn_threshold=True,
                           spike_grad=surrogate.fast_sigmoid(slope=25))),
            nn.MaxPool2d(4, stride=2),
            SpikeWrapper(snn.Leaky(beta=0.9, threshold=0.5,
                           learn_beta=True, learn_threshold=True,
                           spike_grad=surrogate.fast_sigmoid(slope=25))),
            nn.Conv2d(8, 10, kernel_size=8),
            SpikeWrapper(snn.Leaky(beta=0.9, threshold=0.5,
                           learn_beta=True, learn_threshold=True,
                           spike_grad=surrogate.fast_sigmoid(slope=25))),
            nn.MaxPool2d(4, stride=3),
            nn.Conv2d(10, 12, kernel_size=6),
            SpikeWrapper(snn.Leaky(beta=0.9, threshold=0.5,
                                   learn_beta=True, learn_threshold=True,
                                   spike_grad=surrogate.fast_sigmoid(slope=25))),
            nn.MaxPool2d(2, stride=2)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(12 * 126 * 55, 32),
            SpikeWrapper(snn.Leaky(beta=0.9, threshold=0.5,
                           learn_beta=True, learn_threshold=True,
                           spike_grad=surrogate.fast_sigmoid(slope=25))),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.to(self.device)
        self.reset_states()
    # Spatial transformer network forward function

    def stn(self, x):
        theta_list = []
        frame_range = [0, 15, 30, 46, 61, 77, 97, 112, 123] # frames for gradient checkpoint
        for n in range(8): # splits up the time integration in 3 chunks to allow for gradient checkpoints to save VRAM
            internal_states = {}
            for name, module in self.named_modules():
                if isinstance(module, SpikeWrapper):
                    internal_states[name] = module.spiking_layer.mem.clone()
            x_block = x[:, frame_range[n]:frame_range[n+1]].clone().requires_grad_()

            dtheta_list = checkpoint(self.gradient_checkpoint_time_integration, x_block, internal_states, use_reentrant=False)

            theta_list.extend(dtheta_list)


        # Stack the thetas over a new time-dimension and compute the mean.
        theta_rate = torch.mean(torch.stack(theta_list, dim=0), dim=0)
        grid = F.affine_grid(theta_rate, [len(x), 1, 720, 1280]) # Not convertable to spikes (I think)
        x = F.grid_sample(x[:, int(len(x[0])/2), 1].unsqueeze(1), grid)

        return x
    def gradient_checkpoint_time_integration(self, dx, internal_states):
        dtheta_list = []
        named_modules = dict(self.named_modules())
        for name, module in self.named_modules():
            if isinstance(module, SpikeWrapper) and name in internal_states:
                module.spiking_layer.mem = internal_states[name].clone().requires_grad_()



        for frame in range(len(dx[0])):

            xs = self.localization(dx[:, frame])

            xs = xs.view(-1, 12 * 126 * 55)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)

            dtheta_list.append(theta)

        return dtheta_list
    def forward(self, x):
        # reset model states
        # transform the input
        x = self.stn(x)

        return x

    def reset_states(self):
          # Iterate over all submodules and call reset() if available
          for module in self.modules():
              if isinstance(module, SpikeWrapper):
                  module.reset()

    def loss_fct(self, prediction, target):
        """
        prediction and target are expected to have shape [B, 1, H, W].
        Returns:
            total_loss (Tensor, scalar),
            fp_rate (Tensor, scalar),
            fn_rate (Tensor, scalar),
            tp_rate (Tensor, scalar),
            tn_rate (Tensor, scalar),
            total_spikes (Tensor, scalar)
        """



        # Prepare a convolution kernel (shape [1,1,3,3]) on the correct device
        kernel = (torch.tensor([[0.25, 0.5, 0.25],
                                [0.5, 1.0, 0.5],
                                [0.25, 0.5, 0.25]]) / 4.0
                  ).unsqueeze(0).unsqueeze(0).to(target.device)

        # Convolve 'target' to allow for some "forgiving" neighborhood
        #    target:   [B, 1, H, W]
        #    kernel:   [1, 1, 3, 3]
        #    output:   [B, 1, H, W]
        target_conv = F.conv2d(target, kernel, padding=1, stride=1)


        N_tp = target.sum().item()  # total number of '1' in target
        N_fp = (1 - target).sum().item()  # total number of '0' in target
        N_fn = N_tp
        N_tn = N_fp

        fp = (prediction * (1 - target)).sum()  # false positives
        fn = ((1 - prediction) * target).sum()  # false negatives
        tp = (prediction * target_conv).sum()  # true positives (forgiving version)
        tn = ((1 - prediction) * (1 - target)).sum()  # true negatives

        # Sparsity and spike-rate penalties
        #    prediction.mean() => average spike value across entire batch
        spike_rate = prediction.mean()
        sparsity_penalty = spike_rate

        # If spike_rate is very low or very high, penalize accordingly
        anti_silence_penalty = 0.001 / (spike_rate + 1e-4)
        anti_scream_penalty = 0.001 / (1 - spike_rate + 1e-4)

        # Weight coefficients
        weight_true_pos = -1.5
        weight_false_pos = 1.5
        weight_false_neg = 1
        weight_true_neg = -1
        weight_sparsity = 0.3

        # Final loss

        total_loss = (
                weight_true_pos * tp / (N_tp + 1e-8) +  # reward correct spikes
                weight_false_pos * fp / (N_fp + 1e-8) +  # penalize incorrect spikes
                weight_false_neg * fn / (N_fn + 1e-8) +  # penalize missing spikes
                weight_true_neg * tn / (N_tn + 1e-8) +  # reward correct negatives
                weight_sparsity * sparsity_penalty +  # encourage sparse activity
                anti_scream_penalty + anti_silence_penalty
        )


        # Rrates are normalized by total positives/negatives
        return (
            total_loss,
            (fp / (N_fp + 1e-8)).item(),
            (fn / (N_fn + 1e-8)).item(),
            (tp / (N_tp + 1e-8)).item(),
            (tn / (N_tn + 1e-8)).item(),
            prediction.sum().item()  # Num of spikes in prediction
        )