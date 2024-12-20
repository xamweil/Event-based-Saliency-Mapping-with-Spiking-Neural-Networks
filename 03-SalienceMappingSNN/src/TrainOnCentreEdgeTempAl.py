import os
from EventToSalienceMap import EventToSalienceMap
import torch
from PlotTrainingsProgress import PlotTrainingsProgress
import snntorch as snn
from tqdm import tqdm
import copy

def train_on_centre_SNN_temporal_aligned(ets, data_dir, ignore_file, split_ratio=(0.7, 0.15, 0.15), epochs=10,
                                         early_stopping=False, patience=5, learning_rate=1e-3,
                                         device_ids=None, plot_progress=True,
                                         model_dir=r"C:\Users\maxeb\Documents\Radboud\Master\Master Thesis\SNNsalienceMaps\models\VX_TA"):
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
    ets.lif_input_layer = snn.Leaky(beta=0.1, threshold=0.5, reset_mechanism="subtract").to(ets.device)
    ets.lif_train_output_layer = snn.Leaky(beta=0.1, threshold=0.5, reset_mechanism="subtract").to(ets.device)
    ets.input_data_reset = True
    ets.train_output_data_reset = False

    train_scenes, val_scenes, test_scenes = ets._split_dataset(data_dir, ignore_file, split_ratio)
    ets.save_scene_lists(train_scenes, val_scenes, test_scenes, os.path.join(model_dir, "temp_aligned_"))

    # Initialize the SNN model and optimizer
    model = ets.build_on_centre_snn()
    if device_ids:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss and performance tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []
    val_history = []

    torch.autograd.set_detect_anomaly(True)

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

            off_centre_spikes, on_centre_spikes = ets._load_scene_data(scene, data_dir)
            train_ls, train_tp, train_fp, train_fn, train_n_spk = [], [], [], [], []

            # Reset membrane potentials
            ets.mem_on = None
            ets.mem_off = None
            model.reset_states()
            ets.train_output_data = torch.zeros((ets.layer1_n_neurons_x, ets.layer1_n_neurons_y), device=ets.device)

            # Accumulated loss to preserve internal states of model.
            accumulation_size = 5
            for frame, ((off_spikes, x_start), (on_spikes, _)) in enumerate(
                    zip(ets.update_events_step_by_step(off_centre_spikes),
                        ets.update_train_events_step_by_step(on_centre_spikes))):
                x_start = int(ets.layer1_n_neurons_x - x_start)



                # Forward pass without membrane potential reset
                predictions = model(off_spikes.unsqueeze(0).unsqueeze(0),
                                    reset_mem=False)  # Add batch and channel dimensions


                loss = ets.loss_function_temporal_aligned(predictions.squeeze(), on_spikes, x_start)

                # Update progress Bar
                tp = ets._get_tp_to_frame(predictions.squeeze(), on_spikes, x_start)
                fp = ets.get_fp_out_of_frame(predictions.squeeze(), on_spikes, x_start)
                fn = ets.get_fn_out_of_frame(predictions.squeeze(), on_spikes, x_start)

                pbar.set_description(
                    "Training: TP={tp}, FP={fp}, FN={fn}, loss={l}, Epoch={ep}, scene={sc}, frame={fr}/480, scene_nr".format(
                        tp=round(tp, 4),
                        fp=round(fp, 4),
                        fn=round(fn, 4),
                        l=round(loss.item(), 4), ep=epoch, sc=scene,
                        fr=frame
                    ))

                # Backward pass and optimize
                loss.backward(retain_graph=True)
                if frame % accumulation_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    #model.reset_states()
                    model.detach_states()
                    # empty cache
                    torch.cuda.empty_cache()

                # append statistics
                train_ls.append(loss.item())
                train_tp.append(tp)
                train_fp.append(fp)
                train_fn.append(fn)
                train_n_spk.append([predictions.squeeze().sum().item(), on_spikes.sum().item()])
                if plot_progress:
                    ptp.update_plot(off_spikes, predictions.squeeze(), on_spikes, train_tp, train_fp, train_ls,
                                    os.path.join(model_dir, "TrainingPlots"), scene + "ep_{}".format(epoch))

            # save training statistics per scene
            train_stats["loss"].append([train_ls[:]])
            train_stats["tp"].append([train_tp[:]])
            train_stats["fp"].append([train_fp[:]])
            train_stats["fn"].append([train_fn[:]])
            train_stats["n_spk"].append([train_n_spk[:]])
            # clear memory
            train_ls.clear()
            train_tp.clear()
            train_fp.clear()
            train_fn.clear()
            train_n_spk.clear()

        # Validation
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

                off_centre_spikes, on_centre_spikes = ets._load_scene_data(scene, data_dir)
                val_ls, val_tp, val_fp, val_fn, val_n_spk = [], [], [], [], []

                # Reset input layer membrane potential
                ets.mem_on = None
                ets.mem_off = None
                model.reset_states()
                ets.train_output_data = torch.zeros((ets.layer1_n_neurons_x, ets.layer1_n_neurons_y),
                                                     device=ets.device)
                for frame, ((off_spikes, x_start), (on_spikes, _)) in enumerate(
                        zip(ets.update_events_step_by_step(off_centre_spikes),
                            ets.update_train_events_step_by_step(on_centre_spikes))):

                    x_start = int(ets.layer1_n_neurons_x - x_start)

                    # Forward pass without membrane potential reset
                    predictions = model(off_spikes.unsqueeze(0).unsqueeze(0),
                                        reset_mem=False)  # Add batch and channel dimensions


                    loss = ets.loss_function_temporal_aligned(predictions.squeeze(), on_spikes, x_start)

                    # Update progress Bar
                    tp = ets._get_tp_to_frame(predictions.squeeze(), on_spikes, x_start)
                    fp = ets.get_fp_out_of_frame(predictions.squeeze(), on_spikes, x_start)
                    fn = ets.get_fn_out_of_frame(predictions.squeeze(), on_spikes, x_start)

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
                        ptp.update_plot(off_spikes, predictions.squeeze(), on_spikes, val_tp, val_fp, val_ls,
                                        os.path.join(model_dir, "ValidationPlots"), scene + "ep_{}".format(epoch))

                    # empty cache
                    torch.cuda.empty_cache()

                # save validation statistics  per scene
                validation_stats["loss"].append([val_ls[:]])
                validation_stats["tp"].append([val_tp[:]])
                validation_stats["fp"].append([val_fp[:]])
                validation_stats["fn"].append([val_fn[:]])
                validation_stats["n_spk"].append([val_n_spk[:]])
                # clear memory
                val_ls.clear()
                val_tp.clear()
                val_fp.clear()
                val_fn.clear()
                val_n_spk.clear()

        train_history.append(copy.deepcopy(train_stats))
        val_history.append(copy.deepcopy(validation_stats))

    torch.save(model.state_dict(), os.path.join(model_dir, "Temporal_aligned_model.pth"))

    torch.cuda.empty_cache()

    # Testing
    model.load_state_dict(torch.load(os.path.join(model_dir, "Temp_aligned_model.pth")))
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
            off_centre_spikes, on_centre_spikes = ets._load_scene_data(scene, data_dir)
            test_ls, test_tp, test_fp, test_fn, test_n_spk = [], [], [], [], []

            # Reset input layer membrane potential
            ets.mem_on = None
            ets.mem_off = None
            model.reset_states()
            ets.train_output_data = torch.zeros((ets.layer1_n_neurons_x, ets.layer1_n_neurons_y),
                                                 device=ets.device)
            for frame, ((off_spikes, x_start), (on_spikes, _)) in enumerate(
                    zip(ets.update_events_step_by_step(off_centre_spikes),
                        ets.update_train_events_step_by_step(on_centre_spikes))):

                x_start = int(ets.layer1_n_neurons_x - x_start)

                # Forward pass without membrane potential reset
                predictions = model(off_spikes.unsqueeze(0).unsqueeze(0),
                                    reset_mem=reset_mem)  # Add batch and channel dimensions
                reset_mem = False

                loss = ets.loss_function_temporal_aligned(predictions.squeeze(), on_spikes, x_start)

                # Update progress Bar
                tp = ets._get_tp_to_frame(predictions.squeeze(), on_spikes, x_start)
                fp = ets.get_fp_out_of_frame(predictions.squeeze(), on_spikes, x_start)
                fn = ets.get_fn_out_of_frame(predictions.squeeze(), on_spikes, x_start)

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
                    ptp.update_plot(off_spikes, predictions.squeeze(), on_spikes, test_tp, test_fp, test_ls,
                                    os.path.join(model_dir, "TestPlots"), scene + "ep_{}".format(epoch))
                # empty cache
                torch.cuda.empty_cache()

            # save validation statistics  per scene
            test_history["loss"].append([test_ls[:]])
            test_history["tp"].append([test_tp[:]])
            test_history["fp"].append([test_fp[:]])
            test_history["fn"].append([test_fn[:]])
            test_history["n_spk"].append([test_n_spk[:]])

    ets.save_training_history(train_history, val_history, test_history, save_dir=os.path.join(model_dir, "training_logs"))  # shapes (epoch, dict), (epochs, dict), (dict)
    
if __name__ == "__main__":
    ets = EventToSalienceMap()
    data_dir = r"D:\MasterThesisDataSet"
    ignore_file = os.path.join(data_dir, "ignore.txt")
    model_dir = r"C:\Users\maxeb\Documents\Radboud\Master\Master Thesis\SNNsalienceMaps\models\V2_TA"
    os.makedirs(model_dir, exist_ok=True)
    train_on_centre_SNN_temporal_aligned(ets, data_dir, ignore_file, epochs=1, model_dir=model_dir)