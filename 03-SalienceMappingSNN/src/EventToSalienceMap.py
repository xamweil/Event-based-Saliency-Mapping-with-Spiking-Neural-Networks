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
                #print("mem 1", self.mem1.mean().item())
                #print("layer 1",x.mean().item())
                # Second layer: conv2 -> LIF
                x = self.conv2(x)
                x, self.mem2 = self.lif2(x, self.mem2)
                #print("mem 2", self.mem2.mean().item())
                #print("layer 2", x.mean().item())
                # Third layer: conv3 -> LIF
                x = self.conv3(x)
                x, self.mem3 = self.lif3(x, self.mem3)
                #print("mem 3", self.mem3.mean().item())
                #print("layer 3", x.mean().item())
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
    def train_on_centre_SNN_mask(self, data_dir, ignore_file, mode, split_ratio=(0.7, 0.15, 0.15), epochs=10,
                                          early_stopping=False, patience=5, learning_rate=1e-3,
                                          device_ids=None, plot_progress = True,
                                          model_dir = r"C:\Users\maxeb\Documents\Radboud\Master\Master Thesis\SNNsalienceMaps\models\VX_"):
        """
        Trains a convolutional spiking neural network (SNN) for generating on-centre edge maps using a mask-based
        training approach. Supports training, validation, and testing across multiple scenes while managing membrane
        potentials and tracking training statistics.

        Parameters:

            data_dir (str):
                Path to the dataset directory (where the scene folders are stored).
            ignore_file (str):
                Path to the 'ignore.txt' file containing scene folders to ignore during training.
            mode (str):
                Training mode, either "InputMask" or "MembraneMask". Determines the configuration of the SNN layers:
                - "InputMask": Uses membrane potential resetting in the input layer.
                - "MembraneMask": Preserves membrane potentials across frames in the input layer.
            split_ratio (tuple):
                Ratio for splitting the dataset into training, validation, and test sets. Default is (0.7, 0.15, 0.15).
            epochs (int):
                Maximum number of epochs to train. Default is 10.
            early_stopping (bool):
                If True, stops training early based on validation loss. Default is False.
            patience (int):
                Patience for early stopping (only applies if early_stopping=True). Default is 5.
            learning_rate (float):
                Learning rate for the Adam optimizer. Default is 1e-3.
            device_ids (list or None):
                List of device IDs for multi-GPU training using `torch.nn.DataParallel`. If None, training occurs on the
                initialized device. Default is None.
            plot_progress (bool):
                If True, saves training progress plots (e.g., loss, true positives) for each epoch and scene. Default is True.
            model_dir (str):
                Path to the directory where trained models, plots, and logs will be saved. Default is a predefined directory.

        Returns:
            None
        """
        if plot_progress:
            ptp = PlotTrainingsProgress()

        if mode == "InputMask":
            self.lif_input_layer = snn.Leaky(beta = 0.1, threshold=0.5, reset_mechanism="subtract").to(self.device)
            self.lif_train_output_layer = snn.Leaky(beta=0.1, threshold=0.5, reset_mechanism="subtract").to(self.device)
            self.input_data_reset = False
            self.train_output_data_reset = False
            reset_mem = True
        elif mode == "MembraneMaks":
            self.lif_input_layer = snn.Leaky(beta=1, threshold=0.5, reset_mechanism="none").to(self.device)
            self.lif_train_output_layer = snn.Leaky(beta=0.1, threshold=0.5, reset_mechanism="subtract").to(self.device)
            self.input_data_reset = True
            self.train_output_data_reset = False
            reset_mem = False
        else:
            print("No valid mode. Valid training modes are 'InputMask' and 'MembraneMask' " )
            return []



        train_scenes, val_scenes, test_scenes = self._split_dataset(data_dir, ignore_file, split_ratio)
        self.save_scene_lists(train_scenes, val_scenes, test_scenes, os.path.join(model_dir, "{}_".format(mode)))

        # Initialize the SNN model and optimizer
        model = self.build_convolutional_snn_for_on_centre()
        if device_ids:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Loss and performance tracking
        best_val_loss = float('inf')
        patience_counter = 0
        train_history = []
        val_history = []

        # Training Loop
        for epoch in range(epochs):
            pbar = tqdm(train_scenes, total=len(train_scenes),
                        desc="Training: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                            tp="-", fp="-", fn="-", l="-", ep=epoch, sc="-", fr="-"))
            model.train()  # Set model to training mode
            train_stats = {"loss": [], "tp": [], "fp": [], "fn": [], "n_spk": []}
            validation_stats = {"loss": [], "tp": [], "fp": [], "fn": [], "n_spk": []}


            # Process training data one scene at a time

            for scene in pbar:
                if plot_progress:
                    ptp.update_scene("Training", scene)
                pbar.set_description(
                    "Training: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                        tp="-", fp="-", fn="-", l="-", ep=epoch, sc=scene, fr="-"))

                off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)
                train_ls, train_tp, train_fp, train_fn, train_n_spk = [], [], [], [], []

                # Reset membrane potentials and inputs
                self.mem_on = None
                self.mem_off = None
                model.reset_state()
                self.train_output_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y), device=self.device)
                self.input_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y), device=self.device)
                for frame, ((off_spikes, x_start), (on_spikes, _)) in enumerate(zip(self.update_events_step_by_step(off_centre_spikes),
                                                                              self.update_train_events_step_by_step(on_centre_spikes))):
                    x_start = int(self.layer1_n_neurons_x - x_start)

                    optimizer.zero_grad()


                    # Forward pass without membrane potential reset
                    predictions = model(off_spikes.unsqueeze(0).unsqueeze(0), reset_mem=reset_mem)  # Add batch and channel dimensions


                    loss = self.loss_function_mask_training(predictions.squeeze(), on_spikes, x_start)

                    # Update progress Bar
                    tp = self._get_tp_to_frame(predictions.squeeze(), on_spikes, x_start)
                    fp = self.get_fp_out_of_frame(predictions.squeeze(), on_spikes, x_start)
                    fn = self.get_fn_out_of_frame(predictions.squeeze(), on_spikes, x_start)

                    pbar.set_description(
                        "Training: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                            tp=round(tp, 4),
                            fp=round(fp, 4),
                            fn=round(fn, 4),
                            l=round(loss.item(), 4), ep=epoch, sc=scene,
                            fr=frame
                        ))

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    # detach membrane potentials from computational graph
                    model.detach_states()
                    # append statistics
                    train_ls.append(loss.item())
                    train_tp.append(tp)
                    train_fp.append(fp)
                    train_fn.append(fn)
                    train_n_spk.append([predictions.squeeze().sum().item(), on_spikes.sum().item()])
                    if plot_progress:
                        ptp.update_plot(off_spikes, predictions.squeeze(), on_spikes, train_tp, train_fp, train_fn, os.path.join(model_dir, "TrainingPlots"), scene +"ep_{}".format(epoch))
                    # empty cache
                    torch.cuda.empty_cache()
                # save training statistics per scene
                train_stats["loss"].append(train_ls)
                train_stats["tp"].append(train_tp)
                train_stats["fp"].append(train_fp)
                train_stats["fn"].append(train_fn)
                train_stats["n_spk"].append(train_n_spk)
                # clear memory
                train_ls.clear()
                train_tp.clear()
                train_fp.clear()
                train_fn.clear()
                train_n_spk.clear()

            #Validation
            model.eval()
            with torch.no_grad():
                pbar = tqdm(val_scenes, total=len(val_scenes),
                            desc="Validation: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                                tp="-", fp="-", fn="-", l="-", ep=epoch, sc="-", fr="-"))
                for scene in pbar:
                    pbar.set_description(
                        "Validation: tp={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                            tp="-", fp="-", fn="-", l="-", ep=epoch, sc=scene, fr="-"))
                    if plot_progress:
                        ptp.update_scene("Validation", scene)


                    off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)
                    val_ls, val_tp, val_fp, val_fn, val_n_spk = [], [], [], [], []

                    # Reset input layer membrane potential
                    self.mem_on = None
                    self.mem_off = None
                    self.train_output_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y),
                                                         device=self.device)
                    self.input_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y),
                                                  device=self.device)
                    for frame, ((off_spikes, x_start), (on_spikes, _)) in enumerate(
                            zip(self.update_events_step_by_step(off_centre_spikes),
                                self.update_train_events_step_by_step(on_centre_spikes))):

                        x_start = int(self.layer1_n_neurons_x - x_start)

                        # Forward pass without membrane potential reset
                        predictions = model(off_spikes.unsqueeze(0).unsqueeze(0),
                                            reset_mem=reset_mem)  # Add batch and channel dimensions


                        loss = self.loss_function_mask_training(predictions.squeeze(), on_spikes, x_start)

                        # Update progress Bar
                        tp = self._get_tp_to_frame(predictions.squeeze(), on_spikes, x_start)
                        fp = self.get_fp_out_of_frame(predictions.squeeze(), on_spikes, x_start)
                        fn = self.get_fn_out_of_frame(predictions.squeeze(), on_spikes, x_start)

                        pbar.set_description(
                            "Validation: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                                tp=round(tp, 4),
                                fp=round(fp, 4),
                                fn=round(fn, 4),
                                l=round(loss.item(), 4), ep=epoch, sc=scene,
                                fr=frame
                            ))

                        # append statistics
                        val_ls.append(loss.item())
                        val_tp.append(tp)
                        val_fp.append(fp)
                        val_fn.append(fn)
                        val_n_spk.append([predictions.squeeze().sum().item(), on_spikes.sum().item()])
                        if plot_progress:
                            ptp.update_plot(off_spikes, predictions.squeeze(), on_spikes, val_tp, val_fp, val_fn,
                                            os.path.join(model_dir, "ValidationPlots"), scene + "ep_{}".format(epoch))

                        # empty cache
                        torch.cuda.empty_cache()

                    # save validation statistics  per scene
                    validation_stats["loss"].append(val_ls)
                    validation_stats["tp"].append(val_tp)
                    validation_stats["fp"].append(val_fp)
                    validation_stats["fn"].append(val_fn)
                    validation_stats["n_spk"].append(val_n_spk)
                    # clear memory
                    val_ls.clear()
                    val_tp.clear()
                    val_fp.clear()
                    val_fn.clear()
                    val_n_spk.clear()

            train_history.append(train_stats)
            val_history.append(validation_stats)

        torch.save(model.state_dict(), os.path.join(model_dir, "{}_model.pth".format(mode)))

        torch.cuda.empty_cache()



        # Testing
        model.load_state_dict(torch.load(os.path.join(model_dir, "{}_model.pth").format(mode)))
        test_history = {"loss": [], "tp": [], "fp": [], "fn": [], "n_spk": []}
        model.eval()
        with torch.no_grad():
            pbar = tqdm(test_scenes, total=len(val_scenes),
                        desc="Validation: tp={tp}, FP={fp}, FN={fn}, loss={l}, scene={sc}, frame={fr}/480, scene_nr".format(
                            tp="-", fp="-", fn="-", l="-", sc="-", fr="-"))
            for scene in pbar:
                pbar.set_description(
                    "Test: tp={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                        tp="-", fp="-", fn="-", l="-", ep=epoch, sc=scene, fr="-"))

                if plot_progress:
                    ptp.update_scene("Test", scene)
                off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)
                test_ls, test_tp, test_fp, test_fn, test_n_spk = [], [], [], [], []

                # Reset input layer membrane potential
                self.mem_on = None
                self.mem_off = None
                self.train_output_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y),
                                                     device=self.device)
                for frame, ((off_spikes, x_start), (on_spikes, _)) in enumerate(
                        zip(self.update_events_step_by_step(off_centre_spikes),
                            self.update_train_events_step_by_step(on_centre_spikes))):

                    x_start = int(self.layer1_n_neurons_x - x_start)

                    # Forward pass without membrane potential reset
                    predictions = model(off_spikes.unsqueeze(0).unsqueeze(0),
                                        reset_mem=reset_mem)  # Add batch and channel dimensions


                    loss = self.loss_function_mask_training(predictions.squeeze(), on_spikes, x_start)

                    # Update progress Bar
                    tp = self._get_tp_to_frame(predictions.squeeze(), on_spikes, x_start)
                    fp = self.get_fp_out_of_frame(predictions.squeeze(), on_spikes, x_start)
                    fn = self.get_fn_out_of_frame(predictions.squeeze(), on_spikes, x_start)

                    pbar.set_description(
                        "Test: tp={tp}, FP={fp}, FN={fn}, loss={l}, scene={sc}, frame={fr}/480, scene_nr".format(
                            tp=round(tp, 4),
                            fp=round(fp, 4),
                            fn=round(fn, 4),
                            l=round(loss.item(), 4), sc=scene,
                            fr=frame
                        ))

                    # append statistics
                    test_ls.append(loss.item())
                    test_tp.append(tp)
                    test_fp.append(fp)
                    test_fn.append(fn)
                    test_n_spk.append([predictions.squeeze().sum().item(), on_spikes.sum().item()])


                    if plot_progress:
                        ptp.update_plot(off_spikes, predictions.squeeze(), on_spikes, test_tp, test_fp, test_fn,
                                        os.path.join(model_dir,"TestPlots"), scene + "ep_{}".format(epoch))
                    # empty cache
                    torch.cuda.empty_cache()

                # save validation statistics  per scene
                test_history["loss"].append(test_ls)
                test_history["tp"].append(test_tp)
                test_history["fp"].append(test_fp)
                test_history["fn"].append(test_fn)
                test_history["n_spk"].append(test_n_spk)


        self.save_training_history(train_history, val_history, test_history, save_dir=os.path.join(model_dir, "training_logs"))  # shapes (epoch, dict), (epochs, dict), (dict)


    def train_on_centre_SNN_temporal_aligned(self, data_dir, ignore_file, split_ratio=(0.7, 0.15, 0.15), epochs=10,
                                          early_stopping=False, patience=5, learning_rate=1e-3,
                                          device_ids=None, plot_progress = True):
        """
        Parameters:
        data_dir (str): Path to the dataset directory (where the scene folders are stored).
        ignore_file (str): Path to the 'ignore.txt' file containing scene folders to ignore.
        split_ratio (tuple): Ratio for splitting the dataset into train, validation, and test sets. Default is (0.7, 0.15, 0.15).
        epochs (int): Maximum number of epochs to train. Default is 20.
        early_stopping (bool): If True, enables early stopping based on validation loss. Default is False.
        patience (int): Patience for early stopping (only applies if early_stopping=True). Default is 5.
        learning_rate (float): Learning rate for the Adam optimizer. Default is 1e-3.
        plot_training (bool): If True, saves a plot of the training and validation loss/accuracy. Default is True.
        device_ids (tulple): Contains devices to train on, if None trains on initialized device. Default is None
        """
        if plot_progress:
            ptp = PlotTrainingsProgress()

        # Training mode for temporal aligned:
        self.lif_input_layer = snn.Leaky(beta = 0.1, threshold=0.5, reset_mechanism="subtract").to(self.device)
        self.lif_train_output_layer = snn.Leaky(beta=0.1, threshold=0.5, reset_mechanism="subtract").to(self.device)
        self.input_data_reset = True
        self.train_output_data_reset = False


        train_scenes, val_scenes, test_scenes = self._split_dataset(data_dir, ignore_file, split_ratio)
        self.save_scene_lists(train_scenes, val_scenes, test_scenes, r"models\V1_TA\temp_aligned_")

        # Initialize the SNN model and optimizer
        model = self.build_convolutional_snn_for_on_centre()
        if device_ids:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Loss and performance tracking
        best_val_loss = float('inf')
        patience_counter = 0
        train_history = []
        val_history = []

        # Training Loop
        for epoch in range(epochs):
            pbar = tqdm(train_scenes, total=len(train_scenes),
                        desc="Training: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                            tp="-", fp="-", fn="-", l="-", ep=epoch, sc="-", fr="-"))
            model.train()  # Set model to training mode
            train_stats = {"loss": [], "tp": [], "fp": [], "fn": [], "n_spk": []}
            validation_stats = {"loss": [], "tp": [], "fp": [], "fn": [], "n_spk": []}


            # Process training data one scene at a time

            for scene in pbar:
                if plot_progress:
                    ptp.update_scene("Training", scene)
                pbar.set_description(
                    "Training: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                        tp="-", fp="-", fn="-", l="-", ep=epoch, sc=scene, fr="-"))

                off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)
                train_ls, train_tp, train_fp, train_fn, train_n_spk = [], [], [], [], []

                # Reset membrane potentials
                self.mem_on = None
                self.mem_off = None
                model.reset_state()
                self.train_output_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y), device=self.device)
                for frame, ((off_spikes, x_start), (on_spikes, _)) in enumerate(zip(self.update_events_step_by_step(off_centre_spikes),
                                                                              self.update_train_events_step_by_step(on_centre_spikes))):
                    x_start = int(self.layer1_n_neurons_x - x_start)

                    optimizer.zero_grad()


                    # Forward pass without membrane potential reset
                    predictions = model(off_spikes.unsqueeze(0).unsqueeze(0), reset_mem=False)  # Add batch and channel dimensions


                    loss = self.loss_function_temporal_aligned(predictions.squeeze(), on_spikes, x_start)
                    #pre = predictions.squeeze().cpu().numpy()
                    #on_spk = on_spikes.cpu().numpy()
                    # Update progress Bar
                    tp = self._get_tp_to_frame(predictions.squeeze(), on_spikes, x_start)
                    fp = self.get_fp_out_of_frame(predictions.squeeze(), on_spikes, x_start)
                    fn = self.get_fn_out_of_frame(predictions.squeeze(), on_spikes, x_start)

                    pbar.set_description(
                        "Training: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                            tp=round(tp, 4),
                            fp=round(fp, 4),
                            fn=round(fn, 4),
                            l=round(loss.item(), 4), ep=epoch, sc=scene,
                            fr=frame
                        ))

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    # detach membrane potentials from computational graph
                    model.detach_states()
                    # append statistics
                    train_ls.append(loss.item())
                    train_tp.append(tp)
                    train_fp.append(fp)
                    train_fn.append(fn)
                    train_n_spk.append([predictions.squeeze().sum().item(), on_spikes.sum().item()])
                    if plot_progress:
                        ptp.update_plot(off_spikes, predictions.squeeze(), on_spikes, train_tp, train_fp, train_fn, "TrainingPlots", scene +"ep_{}".format(epoch))
                    # empty cache
                    torch.cuda.empty_cache()
                # save training statistics per scene
                train_stats["loss"].append(train_ls)
                train_stats["tp"].append(train_tp)
                train_stats["fp"].append(train_fp)
                train_stats["fn"].append(train_fn)
                train_stats["n_spk"].append(train_n_spk)
                # clear memory
                train_ls.clear()
                train_tp.clear()
                train_fp.clear()
                train_fn.clear()
                train_n_spk.clear()

            #Validation
            model.eval()
            with torch.no_grad():
                pbar = tqdm(val_scenes, total=len(val_scenes),
                            desc="Validation: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                                tp="-", fp="-", fn="-", l="-", ep=epoch, sc="-", fr="-"))
                for scene in pbar:
                    pbar.set_description(
                        "Validation: tp={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                            tp="-", fp="-", fn="-", l="-", ep=epoch, sc=scene, fr="-"))
                    if plot_progress:
                        ptp.update_scene("Validation", scene)


                    off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)
                    val_ls, val_tp, val_fp, val_fn, val_n_spk = [], [], [], [], []

                    # Reset input layer membrane potential
                    self.mem_on = None
                    self.mem_off = None
                    self.train_output_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y),
                                                         device=self.device)
                    for frame, ((off_spikes, x_start), (on_spikes, _)) in enumerate(
                            zip(self.update_events_step_by_step(off_centre_spikes),
                                self.update_train_events_step_by_step(on_centre_spikes))):

                        x_start = int(self.layer1_n_neurons_x - x_start)

                        # Forward pass without membrane potential reset
                        predictions = model(off_spikes.unsqueeze(0).unsqueeze(0),
                                            reset_mem=reset_mem)  # Add batch and channel dimensions
                        reset_mem = False

                        loss = self.loss_function_temporal_aligned(predictions.squeeze(), on_spikes, x_start)

                        # Update progress Bar
                        tp = self._get_tp_to_frame(predictions.squeeze(), on_spikes, x_start)
                        fp = self.get_fp_out_of_frame(predictions.squeeze(), on_spikes, x_start)
                        fn = self.get_fn_out_of_frame(predictions.squeeze(), on_spikes, x_start)

                        pbar.set_description(
                            "Validation: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                                tp=round(tp, 4),
                                fp=round(fp, 4),
                                fn=round(fn, 4),
                                l=round(loss.item(), 4), ep=epoch, sc=scene,
                                fr=frame
                            ))

                        # append statistics
                        val_ls.append(loss.item())
                        val_tp.append(tp)
                        val_fp.append(fp)
                        val_fn.append(fn)
                        val_n_spk.append([predictions.squeeze().sum().item(), on_spikes.sum().item()])
                        if plot_progress:
                            ptp.update_plot(off_spikes, predictions.squeeze(), on_spikes, val_tp, val_fp, val_fn,
                                            "ValidationPlots", scene + "ep_{}".format(epoch))

                        # empty cache
                        torch.cuda.empty_cache()

                    # save validation statistics  per scene
                    validation_stats["loss"].append(val_ls)
                    validation_stats["tp"].append(val_tp)
                    validation_stats["fp"].append(val_fp)
                    validation_stats["fn"].append(val_fn)
                    validation_stats["n_spk"].append(val_n_spk)
                    # clear memory
                    val_ls.clear()
                    val_tp.clear()
                    val_fp.clear()
                    val_fn.clear()
                    val_n_spk.clear()

            train_history.append(train_stats)
            val_history.append(validation_stats)

        torch.save(model.state_dict(), "Temporal_aligned_model.pth")

        torch.cuda.empty_cache()



        # Testing
        model.load_state_dict(torch.load("Temp_aligned_model.pth"))
        test_history = {"loss": [], "tp": [], "fp": [], "fn": [], "n_spk": []}
        model.eval()
        with torch.no_grad():
            pbar = tqdm(test_scenes, total=len(val_scenes),
                        desc="Validation: tp={tp}, FP={fp}, FN={fn}, loss={l}, scene={sc}, frame={fr}/480, scene_nr".format(
                            tp="-", fp="-", fn="-", l="-", sc="-", fr="-"))
            for scene in pbar:
                pbar.set_description(
                    "Test: tp={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                        tp="-", fp="-", fn="-", l="-", ep=epoch, sc=scene, fr="-"))

                if plot_progress:
                    ptp.update_scene("Test", scene)
                off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)
                test_ls, test_tp, test_fp, test_fn, test_n_spk = [], [], [], [], []

                # Reset input layer membrane potential
                self.mem_on = None
                self.mem_off = None
                self.train_output_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y),
                                                     device=self.device)
                for frame, ((off_spikes, x_start), (on_spikes, _)) in enumerate(
                        zip(self.update_events_step_by_step(off_centre_spikes),
                            self.update_train_events_step_by_step(on_centre_spikes))):

                    x_start = int(self.layer1_n_neurons_x - x_start)

                    # Forward pass without membrane potential reset
                    predictions = model(off_spikes.unsqueeze(0).unsqueeze(0),
                                        reset_mem=reset_mem)  # Add batch and channel dimensions
                    reset_mem = False

                    loss = self.loss_function_temporal_aligned(predictions.squeeze(), on_spikes, x_start)

                    # Update progress Bar
                    tp = self._get_tp_to_frame(predictions.squeeze(), on_spikes, x_start)
                    fp = self.get_fp_out_of_frame(predictions.squeeze(), on_spikes, x_start)
                    fn = self.get_fn_out_of_frame(predictions.squeeze(), on_spikes, x_start)

                    pbar.set_description(
                        "Test: tp={tp}, FP={fp}, FN={fn}, loss={l}, scene={sc}, frame={fr}/480, scene_nr".format(
                            tp=round(tp, 4),
                            fp=round(fp, 4),
                            fn=round(fn, 4),
                            l=round(loss.item(), 4), sc=scene,
                            fr=frame
                        ))

                    # append statistics
                    test_ls.append(loss.item())
                    test_tp.append(tp)
                    test_fp.append(fp)
                    test_fn.append(fn)
                    test_n_spk.append([predictions.squeeze().sum().item(), on_spikes.sum().item()])


                    if plot_progress:
                        ptp.update_plot(off_spikes, predictions.squeeze(), on_spikes, test_tp, test_fp, test_fn,
                                        "TestPlots", scene + "ep_{}".format(epoch))
                    # empty cache
                    torch.cuda.empty_cache()

                # save validation statistics  per scene
                test_history["loss"].append(test_ls)
                test_history["tp"].append(test_tp)
                test_history["fp"].append(test_fp)
                test_history["fn"].append(test_fn)
                test_history["n_spk"].append(test_n_spk)


        self.save_training_history(train_history, val_history, test_history)  # shapes (epoch, dict), (epochs, dict), (dict)

    def loss_function_mask_training(self, predictions, target, x_start):
        surrogate_grad = surrogate.fast_sigmoid(slope=25)
        sur_pred = surrogate_grad(predictions)

        # false positive / false negatige penalty and true positive reward up to start of current frame
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

        # Save each list to a separate file
        with open(os.path.join(save_dir, "train_history.txt"), "w") as f:
            for item in train_history:
                f.write(f"{item}\n")

        with open(os.path.join(save_dir, "val_history.txt"), "w") as f:
            for item in val_history:
                f.write(f"{item}\n")

        with open(os.path.join(save_dir, "test_history.txt"), "w") as f:
            for item in test_history:
                f.write(f"{item}\n")

        print(f"Training history saved in '{save_dir}'")


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

