import os.path
import pickle

import snntorch as snn
import numpy
import matplotlib.pyplot as plt
import tqdm
import torch.nn as nn
import torch
from EventToSalienceMap import EventToSalienceMap as ets


class ModelValidationEdge:
    def __init__(self, model : nn.Module, event):
        self.event = event
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def Mask_validation(self, scenes, data_dir):
        """
        Takes a set of scenes and translates the training data and the prediction into spike masks to calculate the
        accuracy, false positives, false negatives and the same for the unmapped spikes to establish a base-line.

        Parameters:
            scenes (list: str): contains all scenes to evaluate

            data_dir (str): path to the stored scenes.

        Returns:
            validation (dict): stores the data under the keywords: scene, accuracy, false_pos, false_neg, base_acc, base_false_pos, base_false_neg
        """
        self.model.eval()

        validation = {"scene": [], "accuracy": [], "false_pos": [], "false_neg": [], "base_acc": [], "base_false_pos": [], "base_false_neg": [], "n_spiking": []}

        pbar = tqdm.tqdm(scenes, desc="Calculating mask, Scenes")
        for scene in pbar:
            # Load scenes
            off_centre_spikes, on_centre_spikes = self.event._load_scene_data(scene, data_dir)
            pbar.set_description("Calculating mask1, Scenes")
            prediction_spike_mask = self.get_prediction_spike_mask(off_centre_spikes)
            pbar.set_description("Calculating mask2, Scenes")
            off_centre_mask = self.get_spike_mask(off_centre_spikes)
            pbar.set_description("Calculating mask3, Scenes")
            target_spike_mask = self.get_spike_mask(on_centre_spikes)
            pbar.set_description("Calculating masks done, Scenes")


            #normalization by total number of elements
            N = target_spike_mask.numel()

            validation["scene"].append(scene)

            validation["accuracy"].append((target_spike_mask==prediction_spike_mask).sum().item()/N)

            validation["false_pos"].append((prediction_spike_mask*(1-target_spike_mask)).sum().item()/N)

            validation["false_neg"].append(((1-prediction_spike_mask) * target_spike_mask).sum().item()/N)

            validation["base_acc"].append((target_spike_mask==off_centre_mask).sum().item()/N)

            validation["base_false_pos"].append((off_centre_mask*(1-target_spike_mask)).sum().item()/N)

            validation["base_false_neg"].append(((1-off_centre_mask) * target_spike_mask).sum().item()/N)

            validation["n_spiking"].append([prediction_spike_mask.sum().item()/N, target_spike_mask.sum().item()/N])

            torch.cuda.empty_cache()
        return validation



    def temp_aligned_validation(self, scenes, data_dir, neurons_x = 11130, neurons_y = 720):
        """
        Takes a set of scenes to calculate the accuracy, false positives, false negatives and the same for the
        unmapped spikes to establish a base-line for every frame.

        Parameters:
            scenes (list: str): contains all scenes to evaluate

            data_dir (str): path to the stored scenes.

        Returns:
            validation (dict): stores the data under the keywords: accuracy, false_pos, false_neg, base_acc,
                               base_false_pos, base_false_neg
        """
        self.model.eval()

        validation = {"scene": [], "accuracy": [], "false_pos": [], "false_neg": [], "base_acc": [], "base_false_pos": [],
                      "base_false_neg": [], "n_spiking": []}
        pbar = tqdm.tqdm(scenes, desc="frame=-/480, Scenes")
        with torch.no_grad():
            for scene in pbar:
                pbar.set_description("frame=-/480, Scenes")
                # Load scenes
                off_centre_spikes, on_centre_spikes = self.event._load_scene_data(scene, data_dir)

                validation["scene"].append(scene)
                scene_val = {"accuracy": [], "false_pos": [], "false_neg": [], "base_acc": [], "base_false_pos": [],
                          "base_false_neg": [], "n_spiking": []}
                frame = 0
                for (off_spikes, _), (on_spikes, _) in zip(self.event.update_events_step_by_step(off_centre_spikes),
                                                           self.event.update_events_step_by_step(on_centre_spikes)):
                    pbar.set_description("frame={}/480, Scenes".format(frame))
                    prediction = self.model(off_spikes.unsqueeze(0).unsqueeze(0), reset_mem=True)
                    N = neurons_x * neurons_y
                    scene_val["accuracy"].append((on_spikes == prediction).sum().item() / N)

                    scene_val["false_pos"].append((prediction * (1 - on_spikes)).sum().item() / N)

                    scene_val["false_neg"].append(((1 - prediction) * on_spikes).sum().item() / N)

                    scene_val["base_acc"].append((on_spikes== off_spikes).sum().item() / N)

                    scene_val["base_false_pos"].append((off_spikes * (1 - on_spikes)).sum().item() / N)

                    scene_val["base_false_neg"].append(((1 - off_spikes) * on_spikes).sum().item() / N)

                    scene_val["n_spiking"].append([prediction.sum().item()/N, on_spikes.sum().item()/N])

                    torch.cuda.empty_cache()
                    frame +=1


                validation["accuracy"].append(scene_val["accuracy"])

                validation["false_pos"].append(scene_val["false_pos"])

                validation["false_neg"].append(scene_val["false_neg"])

                validation["base_acc"].append(scene_val["base_acc"])

                validation["base_false_pos"].append(scene_val["base_false_pos"])

                validation["base_false_neg"].append(scene_val["base_false_neg"])

                validation["n_spiking"].append(scene_val["n_spiking"])
        return validation



    def get_spike_mask(self, events, layer1_n_neurons_x = 11130, layer1_n_neurons_y = 720):
        """
        Accumulate spikes for neurons over the entire event sequence to remove temporal alignment.
        Creates a binary spike mask indicating whether each neuron spikes at least once.

        Parameters:
            events (torch.Tensor): Event data to create mask from.

        Returns:
            torch.Tensor: Binary spike mask (1 if neuron spikes at least once, 0 otherwise).
        """
        spike_mask = torch.zeros((layer1_n_neurons_x, layer1_n_neurons_y), device=self.device)

        # Iterate over the event sequence and accumulate spikes
        for target_spikes, _ in self.event.update_events_step_by_step(events):
            spike_mask = torch.max(spike_mask, (target_spikes > 0).float())  # Update spike mask: 1 if spiked at least once

        return spike_mask

    def get_prediction_spike_mask(self, events, layer1_n_neurons_x = 11130, layer1_n_neurons_y = 720):
        """
        Accumulate spikes for neurons over the entire event sequence to remove temporal alignment.
        Creates a binary spike mask from the model prediction indicating whether each neuron
        spikes at least once.

        Parameters:
            events (torch.Tensor): Event data for .

        Returns:
            torch.Tensor: Binary spike mask (1 if neuron spikes at least once, 0 otherwise).
        """
        spike_mask = torch.zeros((layer1_n_neurons_x, layer1_n_neurons_y), device=self.device)
        reset_mem = True
        with torch.no_grad():
            for off_spikes, _ in self.event.update_events_step_by_step(events):
                prediction_spikes = self.model(off_spikes.unsqueeze(0).unsqueeze(0), reset_mem = reset_mem).squeeze()
                reset_mem = False

                spike_mask = torch.max(spike_mask, (prediction_spikes > 0).float())

                del prediction_spikes, off_spikes
                torch.cuda.empty_cache()
        return spike_mask
    def save_validation(self, dir, name, data):
        """
        Saves validation data in .pkl format.
        Parameters:
            dir (str) = path to the directory to save the data in
            name (str) = name of the data to be saves without .pkl
            data (any) = The data to be saved.
        """
        path = os.path.join(dir, name+".pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    model_path = r"Path/To/modelFolder"
    model_name = "ModelName.pth"
    scene_file = "TestSceneFile"
    data_dir = r"Path/to/dataset"

    event = ets()

    scenes = []
    file = open(os.path.join(model_path, scene_file))
    for line in file:
        scenes.append(line.strip())


    # build model
    model = event.build_convolutional_snn_for_on_centre()

    #load trained model
    model.load_state_dict(torch.load(os.path.join(model_path, model_name)))

    mv = ModelValidationEdge(model, event)
    mask_validation = mv.Mask_validation(scenes, data_dir)
    mv.save_validation(model_path, "MaskValidation", mask_validation)

    Temp_aligned_validation = mv.temp_aligned_validation(scenes, data_dir)
    mv.save_validation(model_path, "TemporalAlignedValidation", Temp_aligned_validation)

