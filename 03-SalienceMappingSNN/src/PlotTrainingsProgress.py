import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


import torch


class PlotTrainingsProgress:
    def __init__(self):
        self.fig, self.ax = plt.subplots(5)
        self.down_sampling = 100
        self.input_layer = self.ax[0].imshow(np.zeros(11130*720).reshape((11130, 720)).T, cmap='gray', vmin=0, vmax=1)
        self.prediction_layer = self.ax[1].imshow(np.zeros(11130*720).reshape((11130, 720)).T, cmap='gray', vmin=0, vmax=1)
        self.train_layer = self.ax[2].imshow(np.zeros(11130*720).reshape((11130, 720)).T, cmap='gray', vmin=0, vmax=1)

        self.graph_tp, = self.ax[3].plot([], [], label = "True Prositive")
        self.graph_fp, = self.ax[3].plot([], [], label = "False Positive")
        self.graph_ls, = self.ax[4].plot([], [], label = "Loss")

        self.ax[0].title.set_text("Input Layer")
        self.ax[1].title.set_text("Prediction Layer")
        self.ax[2].title.set_text("Trainings Layer")
        self.ax[3].title.set_text("Progress")

        self.ax[3].set_xlim([0,480])
        self.ax[3].set_ylim([0,1])
        self.ax[3].legend()

        self.ax[4].set_xlim([0, 480])
        self.ax[4].set_ylim([0, 1])
        self.ax[4].set_xlabel("Frame")
        self.ax[4].title.set_text("Loss")
        self.ax[4].legend()


        self.fig.set_size_inches(10, 7)
        self.fig.tight_layout()
        plt.ion()  # Turn on interactive mode
        self.fig.show()
        self.fig.canvas.flush_events()

    def update_scene(self, mode, scene):
        self.ax[3].title.set_text(mode +" Progress Scenen:{}".format(scene))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()




    def update_plot(self, input, prediction, target, tp, fp, ls, dir, scene):
        # Directly update image layers
        self.input_layer.set_data(input.detach().cpu().numpy().T)
        self.input_layer.set_clim(vmin=0, vmax=1)
        self.prediction_layer.set_data(prediction.detach().cpu().numpy().T)
        self.prediction_layer.set_clim(vmin=0, vmax = 1)
        self.train_layer.set_data(target.detach().cpu().numpy().T)
        self.train_layer.set_clim(vmin=0, vmax=1)

        # Update graph data
        self.graph_tp.set_data(np.arange(len(tp)), tp)
        self.graph_fp.set_data(np.arange(len(tp)), fp)


        self.ax[4].set_ylim([min(ls)-abs(min(ls))*0.2, max(ls)])
        self.graph_ls.set_data(np.arange(len(tp)), ls)

        # Force canvas update
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Save plot
        self.save_plot(dir, scene, frame = len(tp))


    def save_plot(self, dir, scene, frame):
        """
           Save the plot to a file with a fixed size without affecting the display size.

           Parameters:
           - fig: The Matplotlib figure object to save.
           - path: Path to save the figure.
           - width: Desired width of the saved figure in inches.
           - height: Desired height of the saved figure in inches.
           - dpi: Resolution in dots per inch (default 300).
           """
        os.makedirs(dir, exist_ok=True)
        # Backup current size
        original_size = self.fig.get_size_inches()

        # Temporarily set the size
        self.fig.set_size_inches(10, 7)

        # Save the figure
        self.fig.savefig(os.path.join(dir, "{sc}_{fr}.pdf".format(sc = scene, fr = frame)), format="pdf")

        # Restore original size
        self.fig.set_size_inches(original_size)



if __name__ == "__main__":
    down_sampling = 100
    ptp = PlotTrainingsProgress()
    inp, pre, tar = torch.randint(0, 2, (11130, 720)), torch.randint(0, 2, (11130, 720)), torch.randint(0, 2, (11130, 720))
    tp, fp, ls = np.linspace(0, 0.5, 100), np.linspace(0, 0.75, 100), np.linspace(0, 1, 100)
    plt.pause(1)
    ptp.update_scene("Training", "S2_afgadsf")
    plt.pause(1)

    ptp.update_plot(inp, pre, tar, acc, fp, ls, "trainPlots", "S2_afasf")
    plt.pause(3)
"""
import matplotlib.pyplot as plt
import numpy as np
import time

fig, ax = plt.subplots(3)
img = ax[0].imshow(np.zeros((100, 10)).T, cmap='gray')
im2 = ax[1].imshow(np.zeros((100, 10)).T, cmap="gray")

plt.ion()
fig.show()

for _ in range(5):
    data = np.random.rand(100, 10)
    img.set_data(data.T)
    img.set_clim(vmin=data.min(), vmax=data.max())
    im2.set_data(data.T)
    im2.set_clim(vmin=data.min(), vmax=data.max())
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)
"""