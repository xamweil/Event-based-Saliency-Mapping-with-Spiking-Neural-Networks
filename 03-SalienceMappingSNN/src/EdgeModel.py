import snntorch as snn
from snntorch import surrogate
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F

class EdgeModel(nn.Module):
    def __init__(self, device = None):
        super(EdgeModel, self).__init__()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pix_frame_padding = 150
        self.device = device
        self.conv1 = torch.nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.lif1 = snn.Leaky(beta=0.9, threshold=0.5, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.lif2 = snn.Leaky(beta=0.5, threshold=0.3, spike_grad=surrogate.fast_sigmoid(slope=25))

        # Temporal Feature Extraction
        self.sconv_lstm1 = snn.SConv2dLSTM(in_channels=32, out_channels=1, kernel_size=3,
                                           spike_grad=surrogate.fast_sigmoid(slope=25), threshold=0.1,
                                           learn_threshold=True)

        # Recurrent Spiking Output Layer
        self.rleaky = snn.RLeaky(beta=0.9, threshold=0.7, learn_beta=True, learn_threshold=True, learn_recurrent=True,
                                 conv2d_channels=1, kernel_size=1)

        # Apply Xavier initialization
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)

        self.to(self.device)
        self.reset_states()
    def forward(self, x, reset_mem=False):
        if reset_mem:
            self.reset_states()

        # First layer: conv1 -> LIF
        x = self.conv1(x)
        x, self.mem1 = self.lif1(x, self.mem1)

        # Second layer: conv2 -> LIF
        x = self.conv2(x)
        x, self.mem2 = self.lif2(x, self.mem2)


        # Spiking ConvLSTM
        x, self.syn1, self.mem3 = self.sconv_lstm1(x, self.syn1, self.mem3)

        # Recurrent spiking output
        self.rspk, self.mem4 = self.rleaky(x, self.rspk, self.mem4)

        if self.pix_frame_padding > 0:
            out = self.rspk[:, :, :, self.pix_frame_padding:-self.pix_frame_padding]
        else:
            out = self.rspk
        return out

    def reset_states(self):
        """
                Re-initialize internal states (membrane potentials, etc.)
                in the same device and dtype as the model’s parameters.
                """
        self.mem1 = self.lif1.reset_mem().to(self.device)
        self.mem2 = self.lif2.reset_mem().to(self.device)
        self.syn1, self.mem3 = (state.to(self.device) for state in self.sconv_lstm1.reset_mem())
        self.rspk, self.mem4 = (state.to(self.device) for state in self.rleaky.reset_mem())
        #self.rleaky.reset_mem().to(self.device)

    def detach_states(self):
        if self.mem1 is not None:
            self.mem1 = self.mem1.detach()
        if self.mem2 is not None:
            self.mem2 = self.mem2.detach()
        if self.syn1 is not None:
            self.syn1 = self.syn1.detach()
        if self.mem3 is not None:
            self.mem3 = self.mem3.detach()
        if self.mem4 is not None:
            self.mem4 = self.mem4.detach()
        if self.rspk is not None:
            self.rspk = self.rspk.detach()

    def get_internal_states(self):
        return self.mem1.clone().detach(), self.mem2.clone().detach(), self.mem3.clone().detach(), self.mem4.clone().detach(), self.syn1.clone().detach(), self.rspk.clone().detach()

    def set_internal_states(self, states):
        self.mem1 = states[0]
        self.mem2 = states[1]
        self.mem3 = states[2]
        self.mem4 = states[3]
        self.syn1 = states[4]
        self.rspk = states[5]

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

        # 6) Weight coefficients
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

        # 8) Return useful stats
        #    Example rates are normalized by total positives/negatives
        return (
            total_loss,
            (fp / (N_fp + 1e-8)).item(),
            (fn / (N_fn + 1e-8)).item(),
            (tp / (N_tp + 1e-8)).item(),
            (tn / (N_tn + 1e-8)).item(),
            prediction.sum().item()  # Num of spikes in prediction
        )