import os
import argparse
from EventToSalienceMap import EventToSalienceMap


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train SNN on provided dataset directory.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--device_ids', type=str, required=True,
                        help='Comma-separated list of device IDs to use for training (e.g., 0,1).')
    # Parse arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    ignore_file = os.path.join(data_dir, "ignore.txt")

    # Parse device IDs (comma-separated string into a list of integers)
    device_ids = [int(id.strip()) for id in args.device_ids.split(",")]

    # Run the training process
    # Initialize the EventToSalienceMap with the first passed device
    event = EventToSalienceMap(device=f"cuda:{device_ids[0]}")
    event.train_on_centre_SNN_tempo_aligned(data_dir, ignore_file, device_ids=device_ids)


if __name__ == "__main__":
    main()