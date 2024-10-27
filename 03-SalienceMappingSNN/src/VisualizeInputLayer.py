from EventToSalienceMap import EventToSalienceMap
import os
import numpy as np
import torch

input_folder = r"C:\Users\maxeb\Documents\Radboud\Master\Master Thesis\Dataset tests\MasterThesisDataSet\S1_17DRP5sb8fy\off_centre\event"
event_file = r"event.bin"  # Raw event data file
event_filepath = os.path.join(input_folder, event_file)

event = EventToSalienceMap()

# Get event file
events = event.get_event_file(event_filepath)

# Convert to Tensor on GPU
events = torch.tensor(events, device=event.device)
spk = []
print("File loaded and converted")
for steps, spikes in enumerate(event.update_events_step_by_step(events)):
    spk.append(spikes)
print("Data processed")
event.visualize_layer_spikes_2d(spk, 4, output_file = "input_spike_activity_2d.mp4")
