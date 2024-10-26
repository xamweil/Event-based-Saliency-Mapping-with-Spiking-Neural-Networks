import numpy as np
import torch
import torch.nn.init as init
import snntorch as snn
from snntorch import surrogate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
import os


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
                self.l1_lif (snn.Leaky): First layer of Leaky Integrate and Fire (LIF) neurons.
                self.mem1 (None): Membrane potential for the first layer of neurons.
                self.spk_track (list): List to store spike activity at each timestep.
            """
        self.camera_res_x = CAMERA_RES_X
        self.camera_res_y = CAMERA_RES_Y
        self.layer1_n_neurons_x = LAYER1_N_NEURONS_X
        self.layer1_n_neurons_y = CAMERA_RES_Y

        #initialize torch with cuda
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #init input and layer1
        self.input_data = torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y), device=self.device)
        self.l1_lif = snn.Leaky(beta = 0.9).to(self.device) # Is the overall input layer and the off-centre edge map
        self.mem1 = None #torch.zeros((self.layer1_n_neurons_x, self.layer1_n_neurons_y), device=self.device) #stores membrane potentials of layer1


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
        x_end = x_start+self.camera_res_x

        if x_end > len(self.input_data):
            split_idx = len(self.input_data) -x_start
            self.input_data[x_start:] = data[:split_idx]
            self.input_data[:x_end % len(self.input_data)] = data[split_idx]
        else:
            self.input_data[x_start:x_end] = data

        # After updating input_data, automatically feed it into the first layer
        spk = self.feed_input_to_layer()
        return spk


    def feed_input_to_layer(self):
        """
            Feeds the updated input data to the first layer of LIF neurons and stores the spiking activity.
            Side effects:
                - Updates the membrane potential `self.mem` for the first layer of neurons.
                - Appends the spiking activity `spk` to `self.spk_track`.
            Returns:
                spk (torch.Tensor): The spiking activity of the first layer of neurons.
            """

        # Initialize the membrane potential if it's None
        if self.mem1 is None:
            self.mem1 = torch.zeros_like(self.input_data, device=self.device)

        spk, self.mem1 = self.l1_lif(self.input_data, self.mem1)

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

        while t_min < t_end:
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
            yield spikes
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
                self.lif3 = snn.Leaky(beta=0.9, threshold=0.3, spike_grad=surrogate.fast_sigmoid(slope=25))
                # Membrane potentials for the LIF neurons
                self.mem1 = None
                self.mem2 = None
                self.mem3 = None

                # Apply Xavier initialization
                init.xavier_uniform_(self.conv1.weight)
                init.xavier_uniform_(self.conv2.weight)
                init.xavier_uniform_(self.conv3.weight)

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

        # Instantiate the model and move it to the appropriate device
        model = ConvSNNOnCentre().to(self.device)
        return model

    def train_on_centre_SNN_tempo_aligned(self, data_dir, ignore_file, split_ratio=(0.7, 0.15, 0.15), epochs=20,
                            early_stopping=False, patience=5, learning_rate=1e-3, plot_training=True, device_ids=None):
        """
        Trains the convolutional SNN for generating on-centre edge maps.

        Parameters:
            data_dir (str): Path to the dataset directory (where the scene folders are stored).
            ignore_file (str): Path to the 'ignore.txt' file containing scene folders to ignore.
            split_ratio (tuple): Ratio for splitting the dataset into train, validation, and test sets. Default is (0.7, 0.15, 0.15).
            epochs (int): Maximum number of epochs to train. Default is 20.
            early_stopping (bool): If True, enables early stopping based on validation loss. Default is False.
            patience (int): Patience for early stopping (only applies if early_stopping=True). Default is 5.
            learning_rate (float): Learning rate for the Adam optimizer. Default is 1e-3.
            plot_training (bool): If True, saves a plot of the training and validation loss/accuracy. Default is True.
        """
        # Load Dataset and Split into Train/Validation/Test
        print("Loading dataset paths...")
        #torch.autograd.set_detect_anomaly(True)

        train_scenes, val_scenes, test_scenes = self._split_dataset(data_dir, ignore_file, split_ratio)

        # Save the scene lists to text files
        self.save_scene_lists(train_scenes, val_scenes, test_scenes)

        # Initialize the SNN model and optimizer
        model = self.build_convolutional_snn_for_on_centre()
        if device_ids:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        # Loss and performance tracking
        best_val_loss = float('inf')
        patience_counter = 0
        train_loss_history, val_loss_history = [], []
        train_accuracy_history, val_accuracy_history = [], []

        # Training Loop
        for epoch in tqdm(range(epochs), desc="Expochs"):
            model.train()  # Set model to training mode
            train_loss, train_correct, total_train = 0, 0, 0

            # Process training data one scene at a time
            for scene in train_scenes:
                print(scene)
                off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)
                print("Scene loaded")
                #  Accumulate spikes for the on-centre data (binary mask)

                reset_mem=True
                i=0
                for off_spikes, on_spikes in zip(self.update_events_step_by_step(off_centre_spikes),
                                                 self.update_events_step_by_step(on_centre_spikes)):
                    print("Frame ", i)
                    optimizer.zero_grad()
                    model.reset_state()

                    # Forward pass
                    predictions = model(off_spikes.unsqueeze(0).unsqueeze(0), reset_mem=reset_mem)  # Add batch and channel dimensions
                    reset_mem=True
                    # Compare predictions against the spike mask (not individual frames)
                    loss = self._calculate_loss_tempo_aligned(predictions.squeeze(), on_spikes, off_spikes)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()

                    # Accumulate metrics
                    train_loss += loss.item()
                    total_train += on_spikes.numel()
                    # Correct spikes (reward)
                    train_correct += (predictions.squeeze().bool() == on_spikes.bool()).sum().item()
                    i+=1


            # Average training loss and accuracy
            avg_train_loss = train_loss / len(train_scenes)
            train_accuracy = train_correct / total_train
            train_loss_history.append(avg_train_loss)
            train_accuracy_history.append(train_accuracy)

            print(f"Epoch [{epoch + 1}/{epochs}] - Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

            # Step 4: Validation
            model.eval()  # Set model to evaluation mode
            val_loss, val_correct, total_val = 0, 0, 0
            with torch.no_grad():
                for scene in val_scenes:
                    off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)

                    for off_spikes, on_spikes in zip(self.update_events_step_by_step(off_centre_spikes),
                                                     self.update_events_step_by_step(on_centre_spikes)):
                        predictions = model(off_spikes.unsqueeze(0).unsqueeze(0))
                        loss = self._calculate_loss_tempo_aligned(predictions, on_spikes)

                        # Accumulate metrics
                        val_loss += loss.item()
                        total_val += on_spikes.numel()
                        val_correct += (predictions.squeeze().bool() == on_spikes.bool()).sum().item()

            # Average validation loss and accuracy
            avg_val_loss = val_loss / len(val_scenes)
            val_accuracy = val_correct / total_val
            val_loss_history.append(avg_val_loss)
            val_accuracy_history.append(val_accuracy)

            print(f"Epoch [{epoch + 1}/{epochs}] - Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

            # Early stopping logic
            if early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0  # Reset patience counter
                    torch.save(model.state_dict(), "best_model.pth")  # Save the best model
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        torch.save(model.state_dict(), "Temp_alligned_model.pth")

        # Step 5: Testing
        print("Testing model on test set...")
        model.load_state_dict(torch.load("best_model.pth"))  # Load best model
        test_loss, test_correct, total_test = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for scene in test_scenes:
                off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)
                for off_spikes, on_spikes in zip(self.update_events_step_by_step(off_centre_spikes),
                                                 self.update_events_step_by_step(on_centre_spikes)):
                    predictions = model(off_spikes.unsqueeze(0).unsqueeze(0))
                    loss = self._calculate_loss_tempo_aligned(predictions, on_spikes)

                    # Accumulate metrics
                    test_loss += loss.item()
                    total_test += on_spikes.numel()
                    test_correct += torch.sum(predictions*on_spikes).item()

        avg_test_loss = test_loss / len(test_scenes)
        test_accuracy = test_correct / total_test
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Step 6: Plot Training History
        if plot_training:
            self._plot_training_history(train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history)
    def train_on_centre_SNN(self, data_dir, ignore_file, split_ratio=(0.7, 0.15, 0.15), epochs=20,
                            early_stopping=False, patience=5, learning_rate=1e-3, plot_training=True):
        """
        Trains the convolutional SNN for generating on-centre edge maps.

        Parameters:
            data_dir (str): Path to the dataset directory (where the scene folders are stored).
            ignore_file (str): Path to the 'ignore.txt' file containing scene folders to ignore.
            split_ratio (tuple): Ratio for splitting the dataset into train, validation, and test sets. Default is (0.7, 0.15, 0.15).
            epochs (int): Maximum number of epochs to train. Default is 20.
            early_stopping (bool): If True, enables early stopping based on validation loss. Default is False.
            patience (int): Patience for early stopping (only applies if early_stopping=True). Default is 5.
            learning_rate (float): Learning rate for the Adam optimizer. Default is 1e-3.
            plot_training (bool): If True, saves a plot of the training and validation loss/accuracy. Default is True.
        """
        # Load Dataset and Split into Train/Validation/Test
        print("Loading dataset paths...")
        #torch.autograd.set_detect_anomaly(True)

        train_scenes, val_scenes, test_scenes = self._split_dataset(data_dir, ignore_file, split_ratio)

        # Initialize the SNN model and optimizer
        model = self.build_convolutional_snn_for_on_centre()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


        # Loss and performance tracking
        best_val_loss = float('inf')
        patience_counter = 0
        train_loss_history, val_loss_history = [], []
        train_accuracy_history, val_accuracy_history = [], []

        # Training Loop
        for epoch in tqdm(range(epochs), desc="Expochs"):
            model.train()  # Set model to training mode
            train_loss, train_correct, total_train = 0, 0, 0

            # Process training data one scene at a time
            for scene in train_scenes:
                off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)
                print("Scene loaded")
                #  Accumulate spikes for the on-centre data (binary mask)
                spike_mask = self._accumulate_on_centre_spikes(on_centre_spikes)
                reset_mem=True
                i=0
                for off_spikes in self.update_events_step_by_step(off_centre_spikes):
                    print("Frame ", i)
                    optimizer.zero_grad()
                    model.reset_state()

                    # Forward pass
                    predictions = model(off_spikes.unsqueeze(0).unsqueeze(0), reset_mem=reset_mem)  # Add batch and channel dimensions
                    reset_mem=False
                    # Compare predictions against the spike mask (not individual frames)
                    loss = self._calculate_loss(predictions.squeeze(), spike_mask, off_spikes)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    # Accumulate metrics
                    train_loss += loss.item()
                    total_train += spike_mask.numel()
                    # Correct spikes (reward)
                    train_correct += ((predictions.squeeze() > 0.5).bool() & spike_mask.bool()).sum().item()
                    i+=1


            # Average training loss and accuracy
            avg_train_loss = train_loss / len(train_scenes)
            train_accuracy = train_correct / total_train
            train_loss_history.append(avg_train_loss)
            train_accuracy_history.append(train_accuracy)

            print(f"Epoch [{epoch + 1}/{epochs}] - Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

            # Step 4: Validation
            model.eval()  # Set model to evaluation mode
            val_loss, val_correct, total_val = 0, 0, 0
            with torch.no_grad():
                for scene in val_scenes:
                    off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)

                    spike_mask = self._accumulate_on_centre_spikes(on_centre_spikes)

                    for off_spikes in self.update_events_step_by_step(off_centre_spikes):
                        predictions = model(off_spikes.unsqueeze(0).unsqueeze(0))
                        loss = self._calculate_loss(predictions, spike_mask)

                        # Accumulate metrics
                        val_loss += loss.item()
                        total_val += spike_mask.numel()
                        train_correct += ((predictions.squeeze() > 0.5).bool() & spike_mask.bool()).sum().item()

            # Average validation loss and accuracy
            avg_val_loss = val_loss / len(val_scenes)
            val_accuracy = val_correct / total_val
            val_loss_history.append(avg_val_loss)
            val_accuracy_history.append(val_accuracy)

            print(f"Epoch [{epoch + 1}/{epochs}] - Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

            # Early stopping logic
            if early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0  # Reset patience counter
                    torch.save(model.state_dict(), "best_model.pth")  # Save the best model
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Step 5: Testing
        print("Testing model on test set...")
        model.load_state_dict(torch.load("best_model.pth"))  # Load best model
        test_loss, test_correct, total_test = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for scene in test_scenes:
                off_centre_spikes, on_centre_spikes = self._load_scene_data(scene, data_dir)
                for off_spikes, on_spikes in zip(self.update_events_step_by_step(off_centre_spikes),
                                                 self.update_events_step_by_step(on_centre_spikes)):
                    predictions = model(off_spikes.unsqueeze(0).unsqueeze(0))
                    loss = self._calculate_loss(predictions, on_spikes)

                    # Accumulate metrics
                    test_loss += loss.item()
                    total_test += on_spikes.numel()
                    test_correct += (predictions > 0.5).eq(on_spikes).sum().item()

        avg_test_loss = test_loss / len(test_scenes)
        test_accuracy = test_correct / total_test
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Step 6: Plot Training History
        if plot_training:
            self._plot_training_history(train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history)

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
        for on_spikes in self.update_events_step_by_step(on_centre_events):
            spike_mask = torch.max(spike_mask, (on_spikes > 0).float())  # Update spike mask: 1 if spiked at least once

        return spike_mask

    def save_scene_lists(self, train_scenes, val_scenes, test_scenes):
        """
        Saves the train, validation, and test scene lists to text files.
        Files are overwritten if they already exist.
        """
        # Save train scenes
        with open('train_list.txt', 'w') as train_file:
            for scene in train_scenes:
                train_file.write(f"{scene}\n")

        # Save validation scenes
        with open('validation_list.txt', 'w') as val_file:
            for scene in val_scenes:
                val_file.write(f"{scene}\n")

        # Save test scenes
        with open('test_list.txt', 'w') as test_file:
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

    def _calculate_loss(self, predictions, targets, input_spikes):
        """
        Custom loss function to measure performance based on spiking correctness and sparsity.
        - Penalizes false positives (predicted 1 when target is 0).
        - Rewards true positives (predicted 1 when target is 1).
        - Does not punish 'wrong' 0 predictions (false negatives or true negatives).
        - Adds a sparsity penalty to discourage excessive spiking.
        """

        # Apply surrogate_grad
        surrogate_grad = surrogate.fast_sigmoid(slope=25)
        # Reward for correct 1s (True Positives)
        true_positive_reward = surrogate_grad(predictions) * targets

        # Penalize false positives (predicted 1, but target is 0)
        false_positive_penalty = surrogate_grad(predictions) * (1 - targets)


        # Sparsity penalty: Encourage the network to spike less overall (average spike activity)
        sparsity_penalty = torch.mean(predictions)

        weight_true_pos = -1
        weight_false_pos = 5
        weight_sparsity = 0.1
        # Total loss:
        # - Reward for true positives (maximize correct spikes),
        # - Penalty for false positives (minimize incorrect spikes),
        # - Sparsity penalty to avoid excessive spiking.
        total_loss = (
                weight_true_pos * true_positive_reward.sum() / torch.sum(input_spikes) +  # reward for correct spikes
                weight_false_pos * false_positive_penalty.sum() / torch.sum(input_spikes) +  # penalize incorrect spikes
                weight_sparsity * sparsity_penalty  # encourage sparse activity
        )
        print("true pos:", true_positive_reward.sum().item(), weight_true_pos * true_positive_reward.sum().item() / input_spikes.sum().item())
        print("false pos:", false_positive_penalty.sum().item(), weight_false_pos * false_positive_penalty.sum().item() / input_spikes.sum().item())
        print("spartity:" ,sparsity_penalty.item(), weight_sparsity * sparsity_penalty.item())
        print("loss:", total_loss)
        return total_loss

    def _calculate_loss_tempo_aligned(self, predictions, targets, input_spikes):
        """
        Custom loss function to measure performance based on spiking correctness and sparsity.
        - Penalizes false positives (predicted 1 when target is 0).
        - Rewards true positives (predicted 1 when target is 1).
        - Does not punish 'wrong' 0 predictions (false negatives or true negatives).
        - Adds a sparsity penalty to discourage excessive spiking.
        """

        # Apply surrogate_grad
        surrogate_grad = surrogate.fast_sigmoid(slope=25)
        # Reward for correct 1s (True Positives)
        true_positive_reward = surrogate_grad(predictions) * targets

        # Penalize false positives (predicted 1, but target is 0)
        false_positive_penalty = surrogate_grad(predictions) * (1 - targets)


        # Sparsity penalty: Encourage the network to spike less overall (average spike activity)
        sparsity_penalty = torch.mean(predictions)

        weight_true_pos = -1
        weight_false_pos = 0.5
        weight_sparsity = 0.1
        # Total loss:
        # - Reward for true positives (maximize correct spikes),
        # - Penalty for false positives (minimize incorrect spikes),
        # - Sparsity penalty to avoid excessive spiking.
        total_loss = (
                weight_true_pos * true_positive_reward.sum() / torch.sum(input_spikes) +  # reward for correct spikes
                weight_false_pos * false_positive_penalty.sum() / torch.sum(input_spikes) +  # penalize incorrect spikes
                weight_sparsity * sparsity_penalty  # encourage sparse activity
        )
        print("true pos:", true_positive_reward.sum().item(), weight_true_pos * true_positive_reward.sum().item() / input_spikes.sum().item())
        print("false pos:", false_positive_penalty.sum().item(), weight_false_pos * false_positive_penalty.sum().item() / input_spikes.sum().item())
        print("spartity:" ,sparsity_penalty.item(), weight_sparsity * sparsity_penalty.item())
        print("loss:", total_loss)
        return total_loss

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

