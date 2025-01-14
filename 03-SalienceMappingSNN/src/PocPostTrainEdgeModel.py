import os
from PocEdgeModel import PocEdgeModel
import torch
from PocPlotTrainingsProgress import PocPlotTrainingsProgress
import snntorch as snn
from tqdm import tqdm
import copy

def train_on_centre_edge_map(ets, data_dir, ignore_file, model_name, start_epoch=2, split_ratio=(0.7, 0.15, 0.15), epochs = 10,
                             device_ids=None, plot_progress = True,
                             model_dir=r"C:\Users\maxeb\Documents\Radboud\Master\Master Thesis\SNNsalienceMaps\models\VX_POC"):
    """
        Parameters:
            data_dir (str): Path to the dataset directory (where the scene folders are stored).
            ignore_file (str): Path to the 'ignore.txt' file containing scene folders to ignore.
            split_ratio (tuple): Ratio for splitting the dataset into train, validation, and test sets. Default is (0.7, 0.15, 0.15).
            epochs (int): Maximum number of epochs to train. Default is 20.
            plot_progress (bool): If True, saves a plot of the training and validation loss/accuracy. Default is True.
            device_ids (tulple): Contains devices to train on, if None trains on initialized device. Default is None
        """

    n_frames = int((ets.map_n_neurons_x+ets.camera_res_x)/ets.dpix)-1
    ets.lif_input_layer = snn.Leaky(beta=0.1, threshold=0.5, reset_mechanism="subtract").to(ets.device)
    ets.lif_train_output_layer = snn.Leaky(beta=0.1, threshold=0.5, reset_mechanism="subtract").to(ets.device)
    ets.input_data_reset = True

    train_scenes, val_scenes, test_scenes = ets.load_scene_lists(os.path.join(model_dir, "POC_"))


    if plot_progress:
        ptp = PocPlotTrainingsProgress(ets.map_n_neurons_x, ets.map_n_neurons_y, ets.camera_res_x, ets.camera_res_y, len(train_scenes)*3)

    # Initialize the SNN model and optimizer
    model = ets.POC_build_on_centre_snn()
    model.load_state_dict(torch.load(os.path.join(model_dir,  model_name)))
    if device_ids:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        # Possibly more non-recurrent layers here
        {'params': model.sconv_lstm1.parameters(), 'lr': 1e-4},
        {'params': model.rleaky.parameters(), 'lr': 1e-4}
    ], lr=1e-3)

    train_history = []
    val_history = []

    # Training Loop
    for epoch in range(start_epoch, epochs):
        train_history = []
        val_history = []
        pbar = tqdm(train_scenes, total=len(train_scenes),
                    desc="Training: Epoch={ep}, scene={sc}, frame={fr}/{n_fr}, scene_nr".format(
                     ep=epoch, sc="-", fr="-", n_fr = n_frames))
        model.train()  # Set model to training mode
        train_stats = {"loss": [], "fp": [], "fn": [], "tp": [], "tn": [], "n_spk": []}




        for scene in pbar:
            if plot_progress:
                ptp.update_scene("Training", scene)
            pbar.set_description(
                "Training: Epoch={ep}, scene={sc}, frame={fr}/{n_fr}, scene_nr".format(
                     ep=epoch, sc=scene, fr="-", n_fr=n_frames))

            for train_frame in train_scenes[scene]:

                on_spikes = ets.get_train_output(train_frame, data_dir, scene)

                # Reset membrane potentials
                ets.mem_on = None
                ets.mem_off = None
                model.reset_states()
                for frame, (off_spikes) in enumerate(ets.get_input_spikes_frame_by_frame(train_frame, data_dir, scene)):
                    # Forward pass without membrane potential reset

                    prediction = model(off_spikes.unsqueeze(0).unsqueeze(0),
                                        reset_mem=False)  # Add batch and channel dimensions

                    if plot_progress:
                        ptp.update_plot(off_spikes, prediction.squeeze(), on_spikes, train_stats["tp"], train_stats["fp"], train_stats["loss"],
                                        os.path.join(model_dir, "TrainingPlots"), scene + "ep_{}".format(epoch), frame, train_frame)
                    pbar.set_description(
                        "Training: Epoch={ep}, scene={sc}, frame={fr}/{n_fr}, scene_nr".format(
                            ep=epoch, sc=scene, fr=frame, n_fr=n_frames))
                loss, fp, fn, tp, tn, num_spk = ets.loss_fct(prediction, on_spikes)
                # save training statistics per scene
                train_stats["loss"].append(loss.item())
                train_stats["fp"].append(fp.item())
                train_stats["fn"].append(fn.item())
                train_stats["tp"].append(tp.item())
                train_stats["tn"].append(tn.item())
                train_stats["n_spk"].append(num_spk.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        torch.save(model.state_dict(), os.path.join(model_dir, "POC_model_epoch_{}.pth".format(epoch)))

        # Validation
        model.eval()
        with torch.no_grad():
            pbar = tqdm(val_scenes, total=len(val_scenes),
                        desc="Validation: Epoch={ep}, scene={sc}, frame={fr}/{n_fr}, scene_nr".format(
                            ep=epoch, sc="-", fr="-", n_fr=n_frames))
            validation_stats = {"loss": [], "fp": [], "fn": [], "tp": [], "tn": [], "n_spk": []}

            for scene in pbar:
                if plot_progress:
                    ptp.update_scene("Validation", scene)
                pbar.set_description(
                    "Validation: Epoch={ep}, scene={sc}, frame={fr}/{n_fr}, scene_nr".format(
                        ep=epoch, sc=scene, fr="-", n_fr=n_frames))

                for train_frame in val_scenes[scene]:
                    on_spikes = ets.get_train_output(train_frame, data_dir, scene)

                    # Reset membrane potentials
                    ets.mem_on = None
                    ets.mem_off = None
                    model.reset_states()

                    for frame, (off_spikes) in enumerate(
                            ets.get_input_spikes_frame_by_frame(train_frame, data_dir, scene)):
                        # Forward pass without membrane potential reset

                        prediction = model(off_spikes.unsqueeze(0).unsqueeze(0),
                                           reset_mem=False)  # Add batch and channel dimensions

                        if plot_progress:
                            ptp.update_plot(off_spikes, prediction.squeeze(), on_spikes, validation_stats["tp"],
                                            validation_stats["fp"], validation_stats["loss"],
                                            os.path.join(model_dir, "ValidationPlots"), scene + "ep_{}".format(epoch), frame, train_frame)
                        pbar.set_description(
                            "Validation: Epoch={ep}, scene={sc}, frame={fr}/{n_fr}, scene_nr".format(
                                ep=epoch, sc=scene, fr=frame, n_fr=n_frames))

                    loss, fp, fn, tp, tn, num_spk = ets.loss_fct(prediction, on_spikes)
                    # save valudation statistics per scene
                    validation_stats["loss"].append(loss.item())
                    validation_stats["fp"].append(fp.item())
                    validation_stats["fn"].append(fn.item())
                    validation_stats["tp"].append(tp.item())
                    validation_stats["tn"].append(tn.item())
                    validation_stats["n_spk"].append(num_spk.item())

        train_history.append(copy.deepcopy(train_stats))
        val_history.append(copy.deepcopy(validation_stats))
        ets.save_training_history(train_history, "train_history_epoch{}".format(epoch),
                                  save_dir=os.path.join(model_dir, "training_logs"))
        ets.save_training_history(val_history, "val_history_epoch{}".format(epoch),
                                  save_dir=os.path.join(model_dir, "training_logs"))

    torch.save(model.state_dict(), os.path.join(model_dir, "POC_final_model.pth"))
    torch.cuda.empty_cache()

    # Testing
    model.load_state_dict(torch.load(os.path.join(model_dir,  "POC_final_model.pth")))
    model.eval()
    with torch.no_grad():
        pbar = tqdm(test_scenes, total=len(test_scenes),
                    desc="Testing: scene={sc}, frame={fr}/{n_fr}, scene_nr".format(
                        sc="-", fr="-", n_fr=n_frames))

        test_stats = {"loss": [], "fp": [], "fn": [], "tp": [], "tn": [], "n_spk": []}

        for scene in pbar:
            if plot_progress:
                ptp.update_scene("Testing", scene)
            pbar.set_description(
                "Testing: scene={sc}, frame={fr}/{n_fr}, scene_nr".format(
                    sc=scene, fr="-", n_fr=n_frames))
            for train_frame in test_scenes[scene]:
                on_spikes = ets.get_train_output(train_frame, data_dir, scene)

                # Reset membrane potentials
                ets.mem_on = None
                ets.mem_off = None
                model.reset_states()

                for frame, (off_spikes) in enumerate(
                        ets.get_input_spikes_frame_by_frame(train_frame, data_dir, scene)):
                    # Forward pass without membrane potential reset

                    prediction = model(off_spikes.unsqueeze(0).unsqueeze(0),
                                       reset_mem=False)  # Add batch and channel dimensions
                    if plot_progress:
                        ptp.update_plot(off_spikes, prediction.squeeze(), on_spikes, test_stats["tp"],
                                        test_stats["fp"], test_stats["loss"], os.path.join(model_dir, "TestPlots"),
                                        scene + "frame_{}".format(train_frame), frame, train_frame)
                    pbar.set_description(
                        "Testing: scene={sc}, frame={fr}/{n_fr}, scene_nr".format(
                            sc=scene, fr=frame, n_fr=n_frames))

                loss, fp, fn, tp, tn, num_spk = ets.loss_fct(prediction, on_spikes)
                # save valudation statistics per scene
                test_stats["loss"].append(loss.item())
                test_stats["fp"].append(fp.item())
                test_stats["fn"].append(fn.item())
                test_stats["tp"].append(tp.item())
                test_stats["tn"].append(tn.item())
                test_stats["n_spk"].append(num_spk.item())
    test_history = []
    test_history.append(test_stats)
    ets.save_training_history(test_history, "test_history", save_dir=os.path.join(model_dir, "training_logs"))


if __name__ == "__main__":
    ets = PocEdgeModel()
    data_dir = r"D:\MasterThesisDataSet"
    ignore_file = os.path.join(data_dir, "ignore.txt")
    model_dir = r"C:\Users\maxeb\Documents\Radboud\Master\Master Thesis\SNNsalienceMaps\models\V0_POC"
    os.makedirs(model_dir, exist_ok=True)
    train_on_centre_edge_map(ets, data_dir, ignore_file, model_name="POC_model_epoch_2.pth", start_epoch=3, epochs=8, model_dir=model_dir)






