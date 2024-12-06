import numpy as np
import torch
import torch.nn.init as init
import snntorch as snn
from snntorch import surrogate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import time
import os
import pickle
from PlotTrainingsProgress import PlotTrainingsProgress

class EventToSalienceMap:
    def __init__(self, CAMERA_RES_X = 1280 , CAMERA_RES_Y = 720, LAYER1_N_NEURONS_X = 11130, device=None):
        """
            Initializes the EventToSalienceMap object.
            Parameters:
                CAMERA_RES_X (int): Resolution of the camera in the x-direction (default 1280).
                CAMERA_RES_Y (int): Resolution of the camera in the y-direction (default 720).
                LAYER1_N_NEURONS_X (int): Number of neurons in the x-direction for the first layer (default 11130).
            Attributes:
                self.device (torch.device): Device to use, either 'cuda' (GPU) or 'cpu'.
                self.input_data (np.ndarray): Input data corresponding to the first layer.
                self.lif_train_output_layer (snn.Leaky): Layer to generate trainings date of Leaky Integrate and Fire (LIF) neurons.
                self.lif_input_layer (snn.Leaky): First layer of Leaky Integrate and Fire (LIF) neurons.
                self.mem1 (None): Membrane potential for the first layer of neurons.
                self.spk_track (list): List to store spike activity at each timestep.
            """
        self.camera_res_x = CAMERA_RES_X
        self.camera_res_y = CAMERA_RES_Y
        self.layer1_n_neurons_x = LAYER1_N_NEURONS_X
        self.layer1_n_neurons_y = CAMERA_RES_Y

        #initialize torch with cuda
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #init input and trainings layer
        self.input_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y), device=self.device)
        self.train_output_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y), device=self.device)
        self.lif_input_layer = snn.Leaky(beta = 0.1, threshold=0.5, reset_mechanism="subtract").to(self.device) # Is the overall input layer and the off-centre edge map
        self.lif_train_output_layer = snn.Leaky(beta=0.9, threshold=1, reset_mechanism="subtract").to(self.device)
        # Two membrane potentials to be able to call update function in parallel for training.
        self.mem_on = None         # stores
        self.mem_off = None      # Stores membrane potentials of input layer

        # input resets for different training modes:
        self.input_data_reset = True
        self.train_output_data_reset = False


    def get_angle_for_4s_sweep(self, t_min):
        """
         Returns the angle based on the time for a constant velocity 360-degree sweep over 4 seconds.
         Parameters:
             t_min (float): The time (in seconds) used to calculate the angle.
         Returns:
             float: The angle (in degrees) at time t_min.
         """
        return (360/4)*t_min

    #Reads and returns the event.bin (binary format) file of the form [x, y, timestamp, polarity]
    def get_event_file(self, filepath):
        """
          Reads and returns the event file in binary format containing events in the form [x, y, timestamp, polarity].
          Parameters:
              filepath (str): Path to the binary event file.
          Returns:
              np.ndarray: A numpy array of shape (N, 4), where N is the number of events. Each row contains:
                          [x, y, timestamp, polarity].
          """
        with open(filepath, 'rb') as f:
            # Read the entire binary file into a numpy array (assuming float32 for x, y, timestamp, polarity)
            events = np.fromfile(f, dtype=np.float32).reshape(-1, 4)[8:] #for some reason, the 8 first entrys are always corrupted and go to inf or 0
        invalid_x_coords = events[:, 0] > self.camera_res_x
        events[:, 0] = np.clip(events[:, 0], 0, self.camera_res_x - 1)
        events[:, 1] = np.clip(events[:, 1], 0, self.camera_res_y - 1)
        return events

    def update_input(self, data, x_start):
        """
         Updates the input tensor that represents the input to the first layer of neurons. The update only affects
         the field of view (FOV) corresponding to the current angle and automatically feeds the updated input
         to the first layer of LIF neurons.

         Parameters:
             data (torch.Tensor): The current input matrix (excitatory/inhibitory data).
             x_start (int): The starting x-coordinate where the input should be updated in the first layer.

         Side effects:
             - Updates `self.input_data` with the new input.
             - Automatically feeds `self.input_data` to the first layer.
         """
        x_end = x_start + self.camera_res_x

        # Reverse the mapping direction
        reverse_x_start = len(self.input_data) - x_start
        reverse_x_end = reverse_x_start - self.camera_res_x

        if reverse_x_end < 0:
            split_idx = abs(reverse_x_end)
            self.input_data[reverse_x_end:] = data[:split_idx]
            self.input_data[:reverse_x_start] = data[split_idx:]
        else:
            self.input_data[reverse_x_end:reverse_x_start] = data

        # After updating input_data, automatically feed it into the first layer
        spk = self.feed_input_to_layer()
        if self.input_data_reset:
            self.input_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y), device=self.device)
        return spk


    def feed_input_to_layer(self):
        """
        Feeds the updated input data to the first layer of LIF neurons and stores the spiking activity.
        Side effects:
            - Updates the membrane potential for the specified layer (`mem_off` or `mem_on`).
        Returns:
            spk (torch.Tensor): The spiking activity of the specified layer of neurons.
        """
        # check what membrane potential is passed

        # Initialize the membrane potential if it's None
        if self.mem_off is None:
            self.mem_off = torch.zeros_like(self.input_data, device=self.device)

        spk, self.mem_off = self.lif_input_layer(self.input_data, self.mem_off)

        return spk

    def update_train_output(self, data, x_start):
        """
         Updates the input tensor that represents the input to the first layer of neurons. The update only affects
         the field of view (FOV) corresponding to the current angle and automatically feeds the updated input
         to the first layer of LIF neurons.

         Parameters:
             data (torch.Tensor): The current input matrix (excitatory/inhibitory data).
             x_start (int): The starting x-coordinate where the input should be updated in the first layer.

         Side effects:
             - Updates `self.input_data` with the new input.
             - Automatically feeds `self.input_data` to the first layer.
         """
        x_end = x_start + self.camera_res_x

        # Reverse the mapping direction
        reverse_x_start = len(self.input_data) - x_start
        reverse_x_end = reverse_x_start - self.camera_res_x

        if reverse_x_end < 0:
            split_idx = abs(reverse_x_end)
            self.train_output_data[reverse_x_end:] = data[:split_idx]
            self.train_output_data[:reverse_x_start] = data[split_idx:]
        else:
            self.train_output_data[reverse_x_end:reverse_x_start] = data

        # After updating input_data, automatically feed it into the first layer
        spk = self.feed_train_output_to_layer()

        if self.train_output_data_reset:
            self.train_output_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y), device=self.device)
        return spk

    def feed_train_output_to_layer(self):
        """
        Feeds the updated input data to the first layer of LIF neurons and stores the spiking activity.
        Side effects:
            - Updates the membrane potential for the specified layer (`mem_off` or `mem_on`).
        Returns:
            spk (torch.Tensor): The spiking activity of the specified layer of neurons.
        """
        # check what membrane potential is passed

        # Initialize the membrane potential if it's None
        if self.mem_on is None:
            self.mem_on = torch.zeros_like(self.input_data, device=self.device)

        spk, self.mem_on = self.lif_train_output_layer(self.train_output_data, self.mem_on)


        return spk

    def update_events_step_by_step(self, events, dt=4 / 480, get_angle=None):
        """
        Updates the input layer step by step with event data over time.

        Parameters:
            events (torch.Tensor): Preloaded tensor of events (x, y, timestamp, polarity).
            dt (float): The time interval (in seconds) for each time window. Default is set to `4/480` seconds.
            get_angle (callable, optional): A function that returns the angle for the corresponding time window.
                                            If None, defaults to `get_angle_for_4s_sweep`.

        Returns:
            generator: Yields spikes at each timestep.
        """
        if get_angle is None:
            get_angle = self.get_angle_for_4s_sweep  # Default angle method

        t_min, t_max = 0, dt

        data = torch.zeros((self.camera_res_x, self.camera_res_y), device=self.device)

        # Get the last timestamp
        t_end = events[:, 2].max().item()

        while t_min < t_end-dt:
            # Filter events based on timestamp for the current time window
            mask = (events[:, 2] >= t_min) & (events[:, 2] < t_max)
            filtered_events = events[mask]  # Filter events in the current time window

            # Write filtered events into the data tensor
            x_coords = filtered_events[:, 0].long()  # X coordinates
            y_coords = filtered_events[:, 1].long()  # Y coordinates
            polarities = filtered_events[:, 3]  # Polarity values (-1 or 1)

            # Apply filtered events into the input data tensor
            data[x_coords, y_coords] = polarities

            # Get angle and calculate x_start
            angle = get_angle(t_min)
            x_start = int((self.layer1_n_neurons_x / 360) * angle)

            # Update input and yield the spike output
            spikes = self.update_input(data, x_start)

            # Reset data tensor for the next time window
            data.fill_(0)

            # Increment the time window
            t_min = t_max
            t_max += dt

            # Yield the spikes generated in this timestep
            yield spikes, x_start

    def update_train_events_step_by_step(self, events, dt=4 / 480, get_angle=None):
        """
                Updates the trainings output layer step by step with event data over time.

                Parameters:
                    events (torch.Tensor): Preloaded tensor of events (x, y, timestamp, polarity).
                    dt (float): The time interval (in seconds) for each time window. Default is set to `4/480` seconds.
                    get_angle (callable, optional): A function that returns the angle for the corresponding time window.
                                                    If None, defaults to `get_angle_for_4s_sweep`.

                Returns:
                    generator: Yields spikes at each timestep.
                """
        if get_angle is None:
            get_angle = self.get_angle_for_4s_sweep  # Default angle method

        t_min, t_max = 0, dt

        data = torch.zeros((self.camera_res_x, self.camera_res_y), device=self.device)

        # Get the last timestamp
        t_end = events[:, 2].max().item()

        while t_min < t_end - dt:
            # Filter events based on timestamp for the current time window
            mask = (events[:, 2] >= t_min) & (events[:, 2] < t_max)
            filtered_events = events[mask]  # Filter events in the current time window

            # Write filtered events into the data tensor
            x_coords = filtered_events[:, 0].long()  # X coordinates
            y_coords = filtered_events[:, 1].long()  # Y coordinates
            polarities = filtered_events[:, 3]  # Polarity values (-1 or 1)

            # Apply filtered events into the input data tensor
            data[x_coords, y_coords] = polarities

            # Get angle and calculate x_start
            angle = get_angle(t_min)
            x_start = int((self.layer1_n_neurons_x / 360) * angle)

            # Update input and yield the spike output
            spikes = self.update_train_output(data, x_start)

            # Reset data tensor for the next time window
            data.fill_(0)

            # Increment the time window
            t_min = t_max
            t_max += dt

            # Yield the spikes generated in this timestep
            yield spikes, x_start

    def build_on_centre_snn(self):
        class OnCentreSNN(torch.nn.Module):
            def __init__(self, device = self.device):
                super(OnCentreSNN, self).__init__()
                self.device = device
                self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
                self.lif1 = snn.Leaky(beta=0.9, threshold=0.5, spike_grad=surrogate.fast_sigmoid(slope=25))
                self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
                self.lif2 = snn.Leaky(beta=0.5, threshold=0.3, spike_grad=surrogate.fast_sigmoid(slope=25))

                # Temporal Feature Extraction
                self.sconv_lstm1 = snn.SConv2dLSTM(in_channels=32, out_channels=1, kernel_size=3,
                                                  spike_grad=surrogate.fast_sigmoid(slope=25), threshold=0.1, learn_threshold=True)



                # Recurrent Spiking Output Layer
                self.rleaky = snn.RLeaky(beta=0.9, threshold=0.7, learn_beta=True, learn_threshold=True, learn_recurrent=True,
                                         conv2d_channels=1, kernel_size=3)

                # Membrane potentials for the neurons
                self.mem1 = self.lif1.reset_mem().to(self.device)
                self.mem2 = self.lif2.reset_mem().to(self.device)
                self.syn1, self.mem3 = (state.to(self.device) for state in self.sconv_lstm1.reset_mem())
                self.rspk, self.mem4 = (state.to(self.device) for state in self.rleaky.reset_mem())



                # Apply Xavier initialization
                init.xavier_uniform_(self.conv1.weight)
                init.xavier_uniform_(self.conv2.weight)
                #init.xavier_uniform_(self.sconv_lstm1.conv.weight)
                #init.xavier_uniform_(self.rleaky.weight)


            def forward(self, x, reset_mem=False):
                if reset_mem:
                    self.reset_states()

                # First layer: conv1 -> LIF
                x = self.conv1(x)
                x, self.mem1 = self.lif1(x, self.mem1)
                #print("lif1 ", x.sum(), x.shape)
                # Second layer: conv2 -> LIF
                x = self.conv2(x)
                x, self.mem2 = self.lif2(x, self.mem2)
                #print("lif2 ", x.sum(), x.shape)
                #print("mem2", self.mem2.sum())

                # Spiking ConvLSTM
                x, self.syn1, self.mem3 = self.sconv_lstm1(x, self.syn1, self.mem3)
                #print("lstm ", x.sum(), x.shape)


                # Recurrent spiking output
                self.rspk, self.mem4 = self.rleaky(x, self.rspk, self.mem4)

                return self.rspk

            def reset_states(self):
                self.mem1 = self.lif1.reset_mem().to(self.device)
                self.mem2 = self.lif2.reset_mem().to(self.device)
                self.syn1, self.mem3 = (state.to(self.device) for state in self.sconv_lstm1.reset_mem())
                self.rspk, self.mem4 = (state.to(self.device) for state in self.rleaky.reset_mem())
                #self.rleaky.reset_mem()

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

                # Instantiate the model and move it to the appropriate device
        model = OnCentreSNN().to(self.device)
        return model




    def build_convolutional_snn_for_on_centre(self):
        """
        Builds a convolutional SNN to generate the on-centre edge map from the off-centre spike input.

        Returns:
            model (torch.nn.Module): The convolutional SNN.
        """

        class ConvSNNOnCentre(torch.nn.Module):
            def __init__(self):
                super(ConvSNNOnCentre, self).__init__()
                # Define the convolutional layers with LIF neurons
                self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
                self.lif1 = snn.Leaky(beta=0.9, threshold=0.5, spike_grad=surrogate.fast_sigmoid(slope=25))
                self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
                self.lif2 = snn.Leaky(beta=0.9, threshold=0.5, spike_grad=surrogate.fast_sigmoid(slope=25))
                self.conv3 = torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
                self.lif3 = snn.Leaky(beta=0.9, threshold=0.3, spike_grad=surrogate.fast_sigmoid(slope=25), learn_beta=True, learn_threshold=True)
                # Membrane potentials for the LIF neurons
                self.mem1 = None
                self.mem2 = None
                self.mem3 = None
                #"""
                # Apply Xavier initialization
                init.xavier_uniform_(self.conv1.weight)
                init.xavier_uniform_(self.conv2.weight)
                init.xavier_uniform_(self.conv3.weight)
                """
                init.uniform_(self.conv1.weight, a=0.1, b=0.5)
                init.uniform_(self.conv2.weight, a=0.1, b=0.5)
                init.uniform_(self.conv3.weight, a=0.1, b=0.3)
                """
            def forward(self, x, reset_mem=False):
                if reset_mem:
                    self.mem1 = None
                    self.mem2 = None
                    self.mem3 = None
                # Initialize membrane potentials if they are None
                if self.mem1 is None:
                    self.mem1 = torch.zeros_like(x)
                if self.mem2 is None:
                    self.mem2 = torch.zeros_like(x)
                if self.mem3 is None:
                    self.mem3 = torch.zeros_like(x)

                # First layer: conv1 -> LIF
                x = self.conv1(x)
                x, self.mem1 = self.lif1(x, self.mem1)

                # Second layer: conv2 -> LIF
                x = self.conv2(x)
                x, self.mem2 = self.lif2(x, self.mem2)

                # Third layer: conv3 -> LIF
                x = self.conv3(x)
                x, self.mem3 = self.lif3(x, self.mem3)

                return x

            def reset_state(self):
                self.mem1 = None
                self.mem2 = None
                self.mem3 = None

            def detach_states(self):
                """Detaches all membrane potentials from the computation graph."""
                if self.mem1 is not None:
                    self.mem1 = self.mem1.detach()
                if self.mem2 is not None:
                    self.mem2 = self.mem2.detach()
                if self.mem3 is not None:
                    self.mem3 = self.mem3.detach()

        # Instantiate the model and move it to the appropriate device
        model = ConvSNNOnCentre().to(self.device)
        return model


    def loss_function_mask_training(self, predictions, target, x_start):
        surrogate_grad = surrogate.fast_sigmoid(slope=25)
        sur_pred = surrogate_grad(predictions)

        # false positive / false negative penalty and true positive reward up to start of current frame
        x_end = x_start - self.camera_res_x

        if x_end < 0:
            N_tp = target.sum()
            N_fp = (1 - target).sum()
            N_fn = N_tp
            fp = (sur_pred * (1 - target)).sum()
            fn = ((1 - sur_pred) * target).sum()
            tp = (sur_pred * target).sum()
        else:
            N_tp = target[x_end:].sum()
            N_fp = (1 - target[x_end:]).sum()
            N_fn = N_tp

            fp = (sur_pred * (1 - target)).sum()
            fn = ((1 - sur_pred[x_end:]) * target[x_end:]).sum()

            tp = (sur_pred[x_end:] * target[x_end:]).sum()

        sparsity_penalty = torch.mean(sur_pred)

        weight_true_pos = -1.5
        weight_false_pos = 1
        weight_false_neg = 0.5
        weight_sparsity = 0.5 # old 0.3
        # Total loss:
        # - Reward for true positives (maximize correct spikes),
        # - Penalty for false positives (minimize incorrect spikes),
        # - Sparsity penalty to avoid excessive spiking.
        total_loss = (
                weight_true_pos * tp / N_tp +  # reward for correct spikes
                weight_false_pos * fp / N_fp +  # penalize incorrect spikes
                weight_false_neg * fn / N_fn +  # penalize missing spikes
                weight_sparsity * sparsity_penalty  # encourage sparse activity
        )

        return total_loss


    def loss_function_temporal_aligned(self, predictions, target, x_start):

        surrogate_grad = surrogate.fast_sigmoid(slope=25)
        sur_pred = surrogate_grad(predictions)

        # false positive / false negatige penalty and true positive reward up to start of current frame
        x_end = x_start - self.camera_res_x

        if x_end < 0:
            N_tp = target.sum().item()
            N_fp = (1-target).sum().item()
            N_fn = N_tp
            fp = (sur_pred * (1 - target)).sum()
            fn = ((1 - sur_pred) * target).sum()
            tp = (sur_pred * target).sum()
        else:
            N_tp = target[x_end:].sum().item()
            N_fp = (1-target[x_end:]).sum().item()
            N_fn = N_tp

            fp = (sur_pred * (1 - target)).sum()
            fn = ((1 - sur_pred[x_end:]) * target[x_end:]).sum()

            tp = (sur_pred[x_end:] * target[x_end:]).sum()
    

        sparsity_penalty = torch.mean(sur_pred)
    

        weight_true_pos = -1.5
        weight_false_pos = 1.5
        weight_false_neg = 0.5
        weight_sparsity = 0.3
        # Total loss:
        # - Reward for true positives (maximize correct spikes),
        # - Penalty for false positives (minimize incorrect spikes),
        # - Sparsity penalty to avoid excessive spiking.
        total_loss = (
                weight_true_pos * tp / N_tp +  # reward for correct spikes
                weight_false_pos * fp / N_fp +  # penalize incorrect spikes
                weight_false_neg * fn / N_fn +  # penalize missing spikes
                weight_sparsity * sparsity_penalty  # encourage sparse activity
        )

        return total_loss


    def _get_tp_to_frame(self, prediction, target, x_start):
        """
        calculates the ratio of entries that are 1 between two tensors up to the end of the current frame.


        Parameters:
            prediction (torch.Tensor): Tensor that holds the predictions
            target (torch.Tensor): Tensor that holds the targets
            x_start (int): Holds the start position of the frame

        Return:
             Ratio of correct predictions in one frame.
        """
        x_end = x_start - self.camera_res_x
        if x_end > 0:
            tp = prediction[x_end:]*target[x_end:]
            N = target[x_end:].sum().item()
        else:
            tp = prediction * target
            N = target.sum().item()

        if N==0:
            return 1
        return tp.sum().item()/N
    def get_fp_out_of_frame(self, prediction, target, x_start):
        """
        Counts the False positives in the prediction up to the end of the current frame.

        Parameters:
            prediction (torch.Tensor): Tensor that holds the predictions
            target (torch.Tensor): Tensor that holds the targets
            x_start (int): Holds the start position of the frame

        Return:
            ratio of false positives outside the current frame.
        """
        x_end = x_start - self.camera_res_x

        if x_end < 0:
            N = (1-target).sum().item()
            fp = (prediction * (1 - target)).sum().item()
        else:
            N = (1 - target[x_end:]).sum().item()
            fp = (prediction[x_end:] * (1 - target[x_end:])).sum().item()
        if N==0:
            return fp
        return fp/N

    def get_fn_out_of_frame(self, prediction, target, x_start):
        """
        Counts the False positives in the prediction up to the end of the current frame.

        Parameters:
            prediction (torch.Tensor): Tensor that holds the predictions
            target (torch.Tensor): Tensor that holds the targets
            x_start (int): Holds the start position of the frame

        Return:
            ratio of false positives outside the current frame.
        """
        x_end = x_start - self.camera_res_x

        if x_end < 0:
            N = target.sum().item()
            fn = ((1 - prediction) * target).sum().item()
        else:
            N = target[x_end:].sum().item()
            fn = ((1 - prediction[x_end:]) * target[x_end:]).sum().item()
        if N==0:
            return fn
        return fn/N
    def _get_accuracy_mask_training(self, predictions, target_mask, x_start):
        """
        Calculates the accuracy from the part of the mask that already had an input, up to the current frame.
        Assumes a sweep that starts at x=0.

        Parameters:
            predictions (torch.Tensor): Tensor with prediction spikes.

            target_mask (torch.Tensor): Tensor with target spikes containing the spikes of all frames.

            x_start (int): Start position of current frame.

        Return:
             acc (float): accuracy of predictions up to the current frame.
        """

        x_end = x_start+self.camera_res_x

        if x_end > len(target_mask):
            N = len(target_mask)*self.camera_res_y
            acc = (predictions == target_mask).sum().item()/N
        else:
            N = x_end *self.camera_res_y
            acc = (predictions[:x_end] == target_mask[x_end]).sum().item()/N

        return acc

    def _get_false_positive_mask_training(self, predictions, target_mask, x_start):
        """
        Calculates the false positives from the part of the mask that already had an input, up to the current frame.
        Assumes a sweep that starts at x=0.

        Parameters:
            predictions (torch.Tensor): Tensor with prediction spikes.

            target_mask (torch.Tensor): Tensor with target spikes containing the spikes of all frames.

            x_start (int): Start position of current frame.

        Return:
            fp (int): Number of false positives up to the current frame.
        """
        x_end = x_start + self.camera_res_x

        if x_end > len(target_mask):
            N = len(target_mask) * self.camera_res_y
            fp = (predictions * (1-target_mask)).sum().item() / N
        else:
            N = x_end * self.camera_res_y
            fp = (predictions[:x_end] * (1-target_mask[x_end])).sum().item() / N

        return fp

    def _get_false_negative_mask_training(self, predictions, target_mask, x_start):
        """
        Calculates the false negatives from the part of the mask that already had an input, up to the current frame.
        Assumes a sweep that starts at x=0.

        Note: This number might be quite high as the membrane potentials decay over time, producing a false
            negative that is not a problem.

        Parameters:
            predictions (torch.Tensor): Tensor with prediction spikes.

            target_mask (torch.Tensor): Tensor with target spikes containing the spikes of all frames.

            x_start (int): Start position of current frame.

        Return:
            fn (int): Number of false negatives up to the current frame.
        """
        x_end = x_start + self.camera_res_x

        if x_end > len(target_mask):
            N = len(target_mask) * self.camera_res_y
            fn = ((1-predictions) * target_mask).sum().item() / N
        else:
            N = x_end * self.camera_res_y
            fn = ((1-predictions[:x_end]) * target_mask[x_end]).sum().item() / N

        return fn


    def _accumulate_on_centre_spikes(self, on_centre_events):
        """
        Accumulate spikes for on-centre neurons over the entire event sequence to remove temporal alignment.
        Creates a binary spike mask indicating whether each neuron spikes at least once.

        Parameters:
            on_centre_events (torch.Tensor): Event data for on-centre edge map.

        Returns:
            torch.Tensor: Binary spike mask (1 if neuron spikes at least once, 0 otherwise).
        """
        spike_mask = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y), device=self.device)

        # Iterate over the event sequence and accumulate spikes
        for on_spikes, _ in self.update_events_step_by_step(on_centre_events):
            spike_mask = torch.max(spike_mask, (on_spikes > 0).float())  # Update spike mask: 1 if spiked at least once

        return spike_mask

    def save_training_history(self, train_history, val_history, test_history, save_dir="training_logs"):
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        def save_history(file_name, history):
            with open(file_name, "w") as f:
                f.write(repr(history))  # Save the exact structure in a Python-readable format

        save_history(os.path.join(save_dir, "train_history.txt"), train_history)
        save_history(os.path.join(save_dir, "val_history.txt"), val_history)
        save_history(os.path.join(save_dir, "test_history.txt"), test_history)

        print(f"Training history saved in '{save_dir}'")

    def load_training_history(self, file_name):
        with open(file_name, "r") as f:
            return eval(f.read())  # Reconstruct the exact structure using eval()


    def save_scene_lists(self, train_scenes, val_scenes, test_scenes, name):
        """
        Saves the train, validation, and test scene lists to text files.
        Files are overwritten if they already exist.
        """
        # Save train scenes
        with open(name+'train_list.txt', 'w') as train_file:
            for scene in train_scenes:
                train_file.write(f"{scene}\n")

        # Save validation scenes
        with open(name+'validation_list.txt', 'w') as val_file:
            for scene in val_scenes:
                val_file.write(f"{scene}\n")

        # Save test scenes
        with open(name+'test_list.txt', 'w') as test_file:
            for scene in test_scenes:
                test_file.write(f"{scene}\n")

    def _split_dataset(self, data_dir, ignore_file, split_ratio):
        """
        Helper function to split dataset into train, validation, and test sets.
        """
        # Load ignored scenes
        with open(ignore_file, 'r') as f:
            ignored_scenes = f.read().splitlines()

        # Collect valid scene folders
        scenes = [d for d in os.listdir(data_dir) if d not in ignored_scenes and os.path.isdir(os.path.join(data_dir, d))]

        # Shuffle and split dataset
        np.random.shuffle(scenes)
        n_train = int(split_ratio[0] * len(scenes))
        n_val = int(split_ratio[1] * len(scenes))

        train_scenes = scenes[:n_train]
        val_scenes = scenes[n_train:n_train + n_val]
        test_scenes = scenes[n_train + n_val:]

        return train_scenes, val_scenes, test_scenes

    def _load_scene_data(self, scene, data_dir):
        """
        Helper function to load the event data from the scene folders.
        """
        off_centre_path = os.path.join(data_dir, scene, 'off_centre\event', 'event.bin')
        on_centre_path = os.path.join(data_dir, scene, 'on_centre\event', 'event.bin')

        off_centre_events = torch.tensor(self.get_event_file(off_centre_path), device=self.device)
        on_centre_events = torch.tensor(self.get_event_file(on_centre_path), device=self.device)

        return off_centre_events, on_centre_events

    def load_scene_spikes(self, scene, data_dir):
        off_centre_path = os.path.join(data_dir, scene, 'off_centre\in_spk', 'input_spikes.pkl')
        on_centre_path = os.path.join(data_dir, scene, 'on_centre\out_spk', 'output_spikes.pkl')

        off_centre_spikes = self._get_spk_file(off_centre_path)
        on_centre_spikes = self._get_spk_file(on_centre_path)

        return off_centre_spikes, on_centre_spikes

    def _get_spk_file(self, file_path):
        if not os.path.exists(file_path):
            print("file does not extist")
            return []
        with open(file_path, "rb") as f:
            file = pickle.load(f)
        return file



    def _plot_training_history(self, train_loss_history, val_loss_history, train_acc_history, val_acc_history):
        """
        Helper function to plot and save training history.
        """
        epochs = range(1, len(train_loss_history) + 1)

        # Plot Loss
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss_history, 'bo-', label='Training Loss')
        plt.plot(epochs, val_loss_history, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_acc_history, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_acc_history, 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Save plot
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history saved as 'training_history.png'")
    def visualize_layer_spikes_2d(self, spk, duration,layer_n_neuron_x=None, layer_n_neuron_y=None, output_file="layer_spike_activity_2d.mp4"):
        """
        Visualizes the spiking activity of neurons in the first layer over time in 2D.
        Each neuron is represented as a pixel in a video, with the same resolution as the neuron grid (11130x720).

        Parameters:
            duration (float): The total duration (in seconds) over which the spiking activity should be visualized.
            output_file (str): The name of the output video file (default: 'spike_activity_2d.mp4').
        """
        #define default layer size (size of saliency maps)
        if layer_n_neuron_x == None:
            layer_n_neuron_x = self.layer1_n_neurons_x
        if layer_n_neuron_y == None:
            layer_n_neuron_y = self.layer1_n_neurons_y

        # Calculate number of time steps based on duration and the length of the spiking data
        num_steps = len(spk)
        dt = duration / num_steps  # Time per step

        # Prepare dimensions for neurons grid
        num_neurons_x = layer_n_neuron_x
        num_neurons_y = layer_n_neuron_y

        #assert np.shape(spikes) == (num_steps, layer_n_neuron_x, layer_n_neuron_y), 'Shape of spikes does not match layer shape'

        # Set up the figure and axis for displaying frames
        fig, ax = plt.subplots()

        # Create an empty image with the same dimensions as the neuron grid
        img = ax.imshow(np.zeros((num_neurons_y, num_neurons_x)), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')  # Hide the axes for a cleaner output

        # Progress bar for rendering frames
        progress_bar = tqdm(total=num_steps, desc="Rendering frames")

        def update_plot(frame):
            # Get the spikes for the current timestep (stored in self.spk_track)
            spikes = spk[frame].cpu().numpy().T  # Shape is (layer_n_neuron_x, layer_n_neuron_y)

            # Create an RGB image: red for spikes, gray for no spikes
            image_data = np.zeros((num_neurons_y, num_neurons_x, 3), dtype=np.uint8)

            # Neurons that spiked are red, others are gray
            image_data[spikes > 0] = [255, 0, 0]  # Red for spikes
            image_data[spikes == 0] = [128, 128, 128]  # Gray for no spikes

            # Update the image with the new data
            img.set_data(image_data)

            # Update the progress bar
            progress_bar.update(1)

            return [img]

        # Create the animation
        ani = animation.FuncAnimation(fig, update_plot, frames=num_steps, interval=dt * 1000, blit=True)
        print("Rendering complete, starting video save...")

        # Track time for the video save phase
        save_start_time = time.time()

        # Progress bar during the saving phase
        with tqdm(total=num_steps, desc="Saving video") as progress_bar:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=2500)

            # Custom callback to update the progress bar during saving
            def update_progress(current_frame, total_frames):
                progress_bar.update(1)

            ani.save(output_file, writer=writer, progress_callback=update_progress)

        # Measure how long saving took
        save_time = time.time() - save_start_time
        print(f"Video saving complete, took {save_time:.2f} seconds.")

        # Close the progress bar when done
        progress_bar.close()
        print(f"2D animation saved as {output_file}")

if __name__ == "__main__":
    import os

    event = EventToSalienceMap()
    data_dir = r"C:\Users\maxeb\Documents\Radboud\Master\Master Thesis\Dataset tests\MasterThesisDataSet"
    ignore_file = os.path.join(data_dir, "ignore.txt")
    #event.train_on_centre_SNN(data_dir, ignore_file)
    event.train_on_centre_SNN_tempo_aligned(data_dir, ignore_file)

