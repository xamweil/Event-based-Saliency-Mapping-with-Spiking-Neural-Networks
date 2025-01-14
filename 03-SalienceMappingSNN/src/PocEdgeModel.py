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

class PocEdgeModel:
    def __init__(self, PIX_FRAME_PADDING_X = 150, CAMERA_RES_X = 1280, CAMERA_RES_Y = 720, device = None):

        self.camera_res_x = CAMERA_RES_X
        self.camera_res_y = CAMERA_RES_Y
        self.pix_frame_padding = PIX_FRAME_PADDING_X  # number of pixels on each side padding the actual frame
        self.map_n_neurons_x = self.camera_res_x + 2*self.pix_frame_padding
        self.map_n_neurons_y = CAMERA_RES_Y
        self.camera_FOV_x = 41.4    # Field of view of the camera in degree

        #initialize torch with cuda
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # init input and trainings layer
        self.input_data = torch.zeros((self.map_n_neurons_x, self.map_n_neurons_y), device=self.device)
        self.lif_input_layer = snn.Leaky(beta=0.1, threshold=0.5, reset_mechanism="subtract").to(
            self.device)  # Is the overall input layer and the off-centre edge map
        self.lif_train_output_layer = snn.Leaky(beta=0.9, threshold=1, reset_mechanism="subtract").to(self.device)

        # Two membrane potentials to be able to call update function in parallel for training.
        self.mem_on = None  # stores
        self.mem_off = None  # Stores membrane potentials of input layer

        # input resets for different training modes:
        self.input_data_reset = True


        self.off_centre_file = None
        self.on_centre_file = None
        self.off_centre_pos = 128
        self.bytes_per_event = 4 * 4  # float32 * 4 fields = 16 bytes per event: [x, y, t, p]
        
        self.dpix = (self.camera_res_x*360)/(self.camera_FOV_x*480)     # Pixel shift per frame
        self.frame_padding = int(self.pix_frame_padding*(self.camera_FOV_x*480)/(self.camera_res_x*360))

    


    def feed_input_to_layer(self):
        """
        Feeds the updated input data to the first layer of LIF neurons and stores the spiking activity.
        Side effects:
            - Updates the membrane potential for `mem_off`.
        Returns:
            spk (torch.Tensor): The spiking activity of the specified layer of neurons.
        """
        # Initialize membrane potential i None
        if self.mem_off is None:
            self.mem_off = torch.zeros_like(self.input_data, device=self.device)
        spk, self.mem_off = self.lif_input_layer(self.input_data, self.mem_off)

        return spk

    def get_input_spikes_frame_by_frame(self, train_frame, data_dir, scene):
        if self.on_centre_file==None or self.off_centre_file.name != os.path.join(data_dir, scene, 'off_centre', 'event.bin'):
            self._open_event_files(data_dir, scene)
            
        dt = 4/480
        

        #start_frame = train_frame - self.frame_padding
        t_start = 4/480 * (train_frame-self.frame_padding-(self.camera_res_x/self.dpix))
        t_min, t_max = t_start, t_start+dt
        t_end = t_start + 4/480 * int((self.map_n_neurons_x+self.camera_res_x)/self.dpix)

        t_train_frame = 4/480 * train_frame
        while t_max < t_end:
            # Read events for this interval from off-centre and on-centre files:

            off_centre_events, self.off_centre_pos = self._read_events_for_interval(
                self.off_centre_file, self.off_centre_pos, t_min, t_max
            )

            # Calculate coordinate shift
            shift = self.get_coordinate_shifts(t_train_frame, off_centre_events[:, 2])
            off_centre_events[:, 0] += shift

            # Clip coordinates to valid range
            off_centre_events[:, 0] = off_centre_events[:, 0].clamp(0, self.map_n_neurons_x-1)
            off_centre_events[:, 1] = off_centre_events[:, 1].clamp(0, self.map_n_neurons_y-1)

            if off_centre_events.shape[0] > 0:
                x_coords = off_centre_events[:, 0].long()
                y_coords = off_centre_events[:, 1].long()
                polarities = off_centre_events[:, 3]
                self.input_data[x_coords, y_coords] = polarities



            #off_spikes = self.update_input(data_off, x_start)
            off_spikes = self.feed_input_to_layer()
            # Reset data for next iteration
            self.input_data.fill_(0)

            # Move to next time window 
            t_min = t_max
            t_max += dt
            yield off_spikes

            
    def get_train_output(self, train_frame, data_dir, scene):
        """Creates the training data for the relevant part of the on-centre map,
        by loading the data of the train_frame and converting it to spikes.
        
        Parameters:
            train_frame (int): frame number (between 0 and 480) of the frame one tries to reconstruct.
            data_dir (str): Path to the data-set where the scenes are contained.
            scene (str): Name of the scene.
        
        """
        if self.on_centre_file==None or self.on_centre_file.name != os.path.join(data_dir, scene, 'on_centre', 'event', 'event.bin'):
            self._open_event_files(data_dir, scene)
        dt = 4/480
        t_train_frame = train_frame*dt
        on_centre_events, _ = self._read_events_for_interval(self.on_centre_file, start_pos=128, t_min=t_train_frame-dt/2, t_max=t_train_frame+dt/2)
        data_on = torch.zeros((self.camera_res_x, self.camera_res_y), device=self.device)

        # Store in tensor:
        if on_centre_events.shape[0] > 0:
            x_coords = on_centre_events[:, 0].long()
            y_coords = on_centre_events[:, 1].long()
            polarities = on_centre_events[:, 3]
            data_on[x_coords, y_coords] = polarities

        # Convert event data to spikes
        if self.mem_on is None:
            self.mem_on = torch.zeros_like(self.input_data, device=self.device)
        on_spikes, self.mem_on = self.lif_train_output_layer(data_on, self.mem_on)
        return on_spikes


    def _open_event_files(self, data_dir, scene):
        off_centre_path = os.path.join(data_dir, scene, 'off_centre', 'event', 'event.bin')
        on_centre_path = os.path.join(data_dir, scene, 'on_centre', 'event', 'event.bin')

        # Open files in binary read mode
        if self.off_centre_file is not None:
            self.off_centre_file.close()
        if self.on_centre_file is not None:
            self.on_centre_file.close()

        self.off_centre_file = open(off_centre_path, 'rb')
        self.on_centre_file = open(on_centre_path, 'rb')

        # Skip the first 8 corrupted entries: 8 * 16 bytes = 128 bytes
        self.off_centre_file.seek(128, 0)
        self.on_centre_file.seek(128, 0)
        self.off_centre_pos = 128
        self.on_centre_pos = 128
        
    def get_coordinate_shifts(self, t_train_frame, timestamps):
        """
        Calculates how much the coordinates of the loaded frame needs to be shifted to 
        propperly fit the off-centre map.

        Parameters:
            t_frame (float): time that corresponds to the current off-centre-map frame
            timestamps (torch.Tensor): Time stamps of current buffered event-data.
            

        Returns
            (int): Coordinate shift in the x-direction of the buffered event data with respect to the 
                off-centre-map.            

        """
        dt = 4/480
        shift = (self.pix_frame_padding+self.dpix*(t_train_frame-timestamps)/dt).type(torch.int)
        return shift


    def _read_events_for_interval(self, f, start_pos, t_min, t_max):
        """
        Reads events from file f starting at start_pos, returns events within [t_min, t_max).
        Updates and returns new_pos to reflect how far we read.
        """
        f.seek(start_pos, 0)

        # We'll read in chunks. Depending on the density of events, you may tune this.
        chunk_size = 10000  # number of events to read at once
        selected_events = []
        new_pos = start_pos
        while True:
            # Read a chunk of events
            data = f.read(chunk_size * self.bytes_per_event)
            if len(data) == 0:
                # End of file
                break

            chunk = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
            chunk = chunk.copy()  # Make a writable copy

            # Clip coordinates to valid range
            chunk[:, 0] = np.clip(chunk[:, 0], 0, self.camera_res_x - 1)
            chunk[:, 1] = np.clip(chunk[:, 1], 0, self.camera_res_y - 1)

            # Filter by time
            mask = (chunk[:, 2] >= t_min) & (chunk[:, 2] < t_max)
            selected_events.append(chunk[mask])

            # If any event in this chunk is >= t_max, we can stop reading further
            # because events are time-sorted.
            if np.any(chunk[:, 2] >= t_max):
                # Find the position of the first event >= t_max
                idx = np.searchsorted(chunk[:, 2], t_max, side='left')
                # Calculate how many bytes of this chunk we've actually consumed
                consumed_events = idx
                # Adjust file pointer accordingly:
                new_pos = start_pos + consumed_events * self.bytes_per_event
                f.seek(new_pos, 0)
                break
            else:
                # All events in this chunk are < t_max, so we consumed them all
                start_pos += len(chunk) * self.bytes_per_event
                new_pos = start_pos

        if len(selected_events) > 0:
            events_np = np.concatenate(selected_events, axis=0)
        else:
            events_np = np.empty((0, 4), dtype=np.float32)

        # Convert to torch on device
        events_t = torch.tensor(events_np, device=self.device)
        return events_t, new_pos
    
    def split_dataset(self, data_dir, ignore_file, split_ratio, frames_per_scene = 3):
        """
        Randomly splits the data-set into training, validation and test set.

        Parameters:
        
            data_dir (str): Path to the dataset directory (where the scene folders are stored).
            ignore_file (str):  Path to the 'ignore.txt' file containing scene folders to ignore.
            spilit_ratio (tulple): Ratio for splitting the dataset into train, validation, and test sets. Default is (0.7, 0.15, 0.15).
            frames_per_scene (int): How many training frames are chosen per scene.
            

        Returns:
            train_scene (dict): contains the names of the training scenes as
        """
        def _integer_sampling(minimum_space = self.frame_padding):
            ints = np.zeros(frames_per_scene)
            index = 0
            while index<frames_per_scene:
                nr_frames = int((self.map_n_neurons_x+self.camera_res_x)/self.dpix)-1
                s = np.random.randint(int(nr_frames/2), int(480-nr_frames/2))
                if (np.abs(ints-s)>minimum_space).all():
                    ints[index] = s
                    index+=1
            return ints
        # Load ignored scenes
        with open(ignore_file, 'r') as f:
            ignore_scenes = f.read().splitlines()

        # Collect valid scene folders
        scenes = [d for d in os.listdir(data_dir) if d not in ignore_scenes and os.path.isdir(os.path.join(data_dir, d))]

        # Shuffle and split dataset
        np.random.shuffle(scenes)
        n_train = int(split_ratio[0] * len(scenes))
        n_val = int(split_ratio[1] * len(scenes))

        train_scenes = {scene: _integer_sampling() for scene in scenes[:n_train]}
        val_scenes = {scene: _integer_sampling() for scene in scenes[n_train:n_train+n_val]}
        test_scenes = {scene: _integer_sampling() for scene in scenes[n_train+n_val:]}

        return train_scenes, val_scenes, test_scenes

    def save_scene_lists(self, train_scenes, val_scenes, test_scenes, name):
        """
        Saves the train, validation, and test scene dictionaries to .txt:

        Files are overwritten if they already exist.
        """

        # Save train scenes
        with open(name + '_train_list.txt', 'w') as train_file:
            for scene_name, frames in train_scenes.items():
                frames_str = ",".join(map(str, frames))
                train_file.write(f"{scene_name}::{frames_str}\n")

        # Save validation scenes
        with open(name + '_validation_list.txt', 'w') as val_file:
            for scene_name, frames in val_scenes.items():
                frames_str = ",".join(map(str, frames))
                val_file.write(f"{scene_name}::{frames_str}\n")

        # Save test scenes
        with open(name + '_test_list.txt', 'w') as test_file:
            for scene_name, frames in test_scenes.items():
                frames_str = ",".join(map(str, frames))
                test_file.write(f"{scene_name}::{frames_str}\n")

    def load_scene_lists(self, name):
        """
        Loads and returns the train, validation, and test scene dictionaries
        from .txt files in the dictionary format:

        """

        # Load train scenes
        train_scenes = {}
        with open(name + '_train_list.txt', 'r') as train_file:
            for line in train_file:
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                scene_name, frames_str = line.split("::")
                frames_list = frames_str.split(",") if frames_str else []
                # Convert each frame index to int
                frames_list = list(map(float, frames_list))
                train_scenes[scene_name] = frames_list

        # Load validation scenes
        val_scenes = {}
        with open(name + '_validation_list.txt', 'r') as val_file:
            for line in val_file:
                line = line.strip()
                if not line:
                    continue
                scene_name, frames_str = line.split("::")
                frames_list = frames_str.split(",") if frames_str else []
                frames_list = list(map(float, frames_list))
                val_scenes[scene_name] = frames_list

        # Load test scenes
        test_scenes = {}
        with open(name + '_test_list.txt', 'r') as test_file:
            for line in test_file:
                line = line.strip()
                if not line:
                    continue
                scene_name, frames_str = line.split("::")
                frames_list = frames_str.split(",") if frames_str else []
                frames_list = list(map(float, frames_list))
                test_scenes[scene_name] = frames_list

        return train_scenes, val_scenes, test_scenes

    def POC_build_on_centre_snn(self):
        class OnCentreSNN(torch.nn.Module):
            def __init__(self, device = self.device, pix_frame_padding = self.pix_frame_padding):
                super(OnCentreSNN, self).__init__()
                self.pix_frame_padding = pix_frame_padding
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
                                         conv2d_channels=1, kernel_size=1)

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

                # Second layer: conv2 -> LIF
                x = self.conv2(x)
                x, self.mem2 = self.lif2(x, self.mem2)


                # Spiking ConvLSTM
                x, self.syn1, self.mem3 = self.sconv_lstm1(x, self.syn1, self.mem3)



                # Recurrent spiking output
                self.rspk, self.mem4 = self.rleaky(x, self.rspk, self.mem4)

                if self.pix_frame_padding > 0:
                    out = self.rspk[:, :, self.pix_frame_padding:-self.pix_frame_padding, :]
                else:
                    out = self.rspk
                return out

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
            def get_internal_states(self):
                return self.mem1.clone().detach(), self.mem2.clone().detach(), self.mem3.clone().detach(), self.mem4.clone().detach(), self.syn1.clone().detach(), self.rspk.clone().detach()
            def set_internal_states(self, states):
                self.mem1 = states[0]
                self.mem2 = states[1]
                self.mem3 = states[2]
                self.mem4 = states[3]
                self.syn1 = states[4]
                self.rspk = states[5]



                # Instantiate the model and move it to the appropriate device
        model = OnCentreSNN().to(self.device)
        return model

    def loss_fct(self, prediction, target):
        surrogate_grad = surrogate.fast_sigmoid(slope=1)
        sur_pred = surrogate_grad(prediction)

        neighborhood = 3

        # convoluted target for more forgiving True positives
        kernel = (torch.tensor([[0.25, 0.5, 0.25],
                                [0.5, 1.0, 0.5],
                                [0.25, 0.5, 0.25]]) / 4.0).unsqueeze(0).unsqueeze(0).to(device=self.device)

        target_conv = torch.nn.functional.conv2d(target.unsqueeze(0).unsqueeze(0), kernel, padding=1,
                                                 stride=1).squeeze().squeeze()


        N_tp = target.sum().item()
        N_fp = (1 - target).sum().item()
        N_fn = N_tp
        N_tn = N_fp
        fp = (sur_pred * (1 - target)).sum()
        fn = ((1 - sur_pred) * target).sum()
        tp = (sur_pred * target_conv).sum()
        tn = ((1 - sur_pred) * (1 - target)).sum()

        sparsity_penalty = torch.mean(sur_pred)

        spike_rate = sur_pred.mean()
        # If spike_rate is too low, penalize it:
        anti_silence_penalty = 0.001 / (spike_rate + 1e-4)
        # If everything spikes get penalize
        anti_scream_penalty = 0.001 / (1 - spike_rate + 1e-4)

        weight_true_pos = -1.5
        weight_false_pos = 1.5
        weight_false_neg = 1
        weight_true_neg = -1
        weight_sparsity = 0.3
        # Total loss:
        # - Reward for true positives (maximize correct spikes),
        # - Penalty for false positives (minimize incorrect spikes),
        # - Sparsity penalty to avoid excessive spiking.
        total_loss = (
                weight_true_pos * tp / N_tp +  # reward for correct spikes
                weight_false_pos * fp / N_fp +  # penalize incorrect spikes
                weight_false_neg * fn / N_fn +  # penalize missing spikes
                weight_true_neg * tn / N_tn +  # reward for correct negatives
                weight_sparsity * sparsity_penalty +  # encourage sparse activity
                anti_scream_penalty +anti_silence_penalty
        )

        return total_loss, fp/N_fp, fn/N_fn, tp/N_tp, tn/N_tn, prediction.sum()

    def save_training_history(self, history, file_name, save_dir="training_logs"):
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, "w") as f:
            f.write(repr(history))  # Save the exact structure in a Python-readable format

        print(f"Training history saved in '{save_dir}'")

    def load_training_history(self, file_name):
        with open(file_name, "r") as f:
            return eval(f.read())  # Reconstruct the exact structure using eval()
