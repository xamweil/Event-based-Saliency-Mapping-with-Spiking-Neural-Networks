import os
from EventToSalienceMap import EventToSalienceMap
import numpy as np
import torch

# Define input folder and file
input_folder = r"C:\Users\maxeb\Documents\Radboud\Master\Master Thesis\Dataset tests\MasterThesisDataSet\S1_17DRP5sb8fy\off_centre\event"
event_file = r"event.bin"  # Raw event data file
event_filepath = os.path.join(input_folder, event_file)

# Initialize the EventToSalienceMap object
event = EventToSalienceMap()

# Load event data
events = event.get_event_file(event_filepath)
events = torch.tensor(events, device=event.device)

# Build the convolutional SNN for on-centre edge map generation
model = event.build_convolutional_snn_for_on_centre()

spk = []
# Test the untrained network by processing event data step-by-step
print("Testing untrained convolutional SNN...")
with torch.no_grad():  # We don't need gradients for testing
    for step, spikes in enumerate(event.update_events_step_by_step(events)):
        # Feed the spikes (off-centre edge map) into the untrained model
        spikes = spikes.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        on_centre_spikes = model(spikes)

        spk.append(on_centre_spikes[0][0])
        # Output the shape of the predicted on-centre spikes for debugging
        print(f"Step {step + 1}: On-centre spikes shape: {on_centre_spikes.shape}")

        if step>465:
            break


print("Untrained network test complete.")
print(len(spk))
event.visualize_layer_spikes_2d(spk, 4, output_file="untrained_on_centre_map.mp4")

