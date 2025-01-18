import os.path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from OffOnCentreDataset import OffOnCentreDataset
from EdgeModel import EdgeModel
import torch
from ProgressPlotting import ProgessPlotting
import utils


def train_on_centre_edge_map(model_dir, epochs, resume_training: bool, dataset_dir = r"D:\SpikeDataSet",  split_ratio = (0.7, 0.15, 0.15), plot_progress = True, model_name=None):
    model = EdgeModel('cuda')
    model.to(model.device)

    dataset = OffOnCentreDataset(dataset_dir, model.device)
    # When no defined split of the dataset is provided it is split here and the correspinding indices are saves:
    if resume_training:
        train_indices, val_indices, test_indices = dataset.load_splits(os.path.join(model_dir, "dataset_split.txt"))
        if not utils.check_for_overwrites(model_dir, epochs):
            return
        start_epoch, stop_epoch = epochs
        model.load_state_dict(torch.load(os.path.join(model_dir, "EdgeModel_epoch_74.pth")))

    elif not resume_training:
        train_indices, val_indices, test_indices = dataset.split_dataset(split_ratio)
        dataset.save_splits(os.path.join(model_dir, "dataset_split.txt"), train_indices, val_indices, test_indices)
        start_epoch, stop_epoch = 0, epochs
    else:
        return

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=0)
    if plot_progress:
        ptp = ProgessPlotting(nr_scenes=len(train_loader))
    # Define Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        # Possibly more non-recurrent layers here
        {'params': model.sconv_lstm1.parameters(), 'lr': 1e-7},
        {'params': model.rleaky.parameters(), 'lr': 1e-7}
    ], lr=1e-6)
    # Handles underflow protection for gradients
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(start_epoch, stop_epoch):
        # Training:
        pbar = tqdm(train_loader, total=len(train_loader),
                    desc="Training: Epoch={ep}, Step".format(ep=epoch))
        model.train()
        stats = {"loss": [], "fp": [], "fn": [], "tp": [], "tn": [], "n_spk": []}
        for batch in pbar:

            if plot_progress:
                ptp.update_scene("Training", gradient_norm=get_total_gradient_norm(model))

            optimizer.zero_grad()
            model.reset_states()

            with torch.amp.autocast(device_type=model.device, dtype=torch.float16):
                for frame in range(len(batch["input"][0])):

                    prediction = model(batch["input"][:, frame].to(torch.float))

                    if frame%9==0 and frame!=0:
                        # Use of truncated backprop
                        # Intermediate loss approx to keep graph small
                        pix_end_frame = int(len(batch["label"][0,0]) - (frame/len(batch["input"][0]) * len(batch["label"][0,0])))

                        loss, fp, fn, tp, tn, num_spk = model.loss_fct(prediction[:, :, :, pix_end_frame:], batch["label"][:, :, pix_end_frame:].unsqueeze(1).to(torch.float))
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        model.detach_states()
                loss, fp, fn, tp, tn, num_spk = model.loss_fct(prediction, batch["label"].unsqueeze(1).to(torch.float))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


            stats["loss"].append(loss.item())
            stats["fp"].append(fp)
            stats["fn"].append(fn)
            stats["tp"].append(tp)
            stats["tn"].append(tn)
            stats["n_spk"].append(num_spk)

            if plot_progress:
                ptp.update_plot(prediction[0].squeeze(), batch["label"][0, :, :], stats["tp"], stats["fp"], stats["loss"],
                                        os.path.join(model_dir, "TrainingPlots"), batch["scene_name"][0] + "ep_{}".format(epoch))
            torch.cuda.empty_cache()

        utils.save_training_history(stats, f"train_stats_{epoch}", os.path.join(model_dir, "training_logs"))

        torch.save(model.state_dict(), os.path.join(model_dir, "EdgeModel_epoch_{}.pth".format(epoch)))

        # Validation
        pbar = tqdm(val_loader, total=len(val_loader), desc="Validation: Epoch={ep}, Step".format(ep=epoch))
        stats = {"loss": [], "fp": [], "fn": [], "tp": [], "tn": [], "n_spk": []}
        model.eval()
        with torch.no_grad():
            for batch in pbar:
                if plot_progress:
                    ptp.update_scene("Validation")
                model.reset_states()
                for frame in range(len(batch["input"][0])):

                    prediction = model(batch["input"][:, frame].to(torch.float))

                loss, fp, fn, tp, tn, num_spk = model.loss_fct(prediction, batch["label"].unsqueeze(1).to(torch.float))

                stats["loss"].append(loss.item())
                stats["fp"].append(fp)
                stats["fn"].append(fn)
                stats["tp"].append(tp)
                stats["tn"].append(tn)
                stats["n_spk"].append(num_spk)

                if plot_progress:
                    ptp.update_plot(prediction[0].squeeze(), batch["label"][0, :, :], stats["tp"], stats["fp"],
                                    stats["loss"], os.path.join(model_dir, "ValidationPlots"),
                                    batch["scene_name"][0] + "ep_{}".format(epoch))
                utils.save_training_history(stats, f"val_stats_{epoch}", os.path.join(model_dir, "training_logs"))

    # Testing
    pbar = tqdm(test_loader, total=len(test_loader), desc="Testing:  Step")
    stats = {"loss": [], "fp": [], "fn": [], "tp": [], "tn": [], "n_spk": []}
    model.eval()
    with torch.no_grad():
        for batch in pbar:
            if plot_progress:
                ptp.update_scene("Testing")
            model.reset_states()
            for frame in range(len(batch["input"][0])):

                prediction = model(batch["input"][:, frame].to(torch.float))

            loss, fp, fn, tp, tn, num_spk = model.loss_fct(prediction, batch["label"].unsqueeze(1).to(torch.float))

            stats["loss"].append(loss.item())
            stats["fp"].append(fp)
            stats["fn"].append(fn)
            stats["tp"].append(tp)
            stats["tn"].append(tn)
            stats["n_spk"].append(num_spk)

            if plot_progress:
                ptp.update_plot(prediction[0].squeeze(), batch["label"][0, :, :], stats["tp"], stats["fp"],
                                stats["loss"], os.path.join(model_dir, "TestPlots"),
                                batch["scene_name"][0])
    utils.save_training_history(stats, f"test_stats", os.path.join(model_dir, "training_logs"))

def get_total_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

if __name__ == "__main__":
    model_dir = os.path.join(r"Path/to/model/directory", "EdgeModel_V0")
    os.makedirs(model_dir, exist_ok=True)
    train_on_centre_edge_map(model_dir=model_dir, resume_training=True, epochs=(75, 100))



