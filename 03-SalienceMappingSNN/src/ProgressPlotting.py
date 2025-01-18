import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

class ProgessPlotting:
    def __init__(self, nr_scenes, output_size_x =1280, output_size_y = 720):
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 1, figure=self.fig, height_ratios=[1, 1, 1])
        gs_row0 = gs[0].subgridspec(1, 2)  # Nested GridSpec for first row


        ax1 = self.fig.add_subplot(gs_row0[0, 0])  # Second subplot
        ax2 = self.fig.add_subplot(gs_row0[0, 1])  # Third subplot

        ax3 = self.fig.add_subplot(gs[1, 0])
        ax4 = self.fig.add_subplot(gs[2, 0])

        self.ax = ["Don't ask", ax1, ax2, ax3, ax4]

        #self.input_layer = self.ax[0].imshow(np.zeros(input_size_x*input_size_y).reshape((input_size_x, input_size_y)).T, cmap='gray', vmin=0, vmax=1)
        self.prediction_layer = self.ax[1].imshow(np.zeros(output_size_x*output_size_y).reshape((output_size_x, output_size_y)).T, cmap='gray', vmin=0, vmax=1)
        self.train_layer = self.ax[2].imshow(np.zeros(output_size_x*output_size_y).reshape((output_size_x, output_size_y)).T, cmap='gray', vmin=0, vmax=1)

        self.graph_tp, = self.ax[3].plot([], [], label="True Prositive")
        self.graph_fp, = self.ax[3].plot([], [], label="False Positive")
        self.graph_ls, = self.ax[4].plot([], [], label="Loss")

        #self.ax[0].title.set_text("Input Layer")
        self.ax[1].title.set_text("Prediction Layer")
        self.ax[2].title.set_text("Trainings Layer")
        self.ax[3].title.set_text("Progress")

        self.ax[3].set_xlim([0, nr_scenes])
        self.ax[3].set_ylim([0, 1])
        self.ax[3].legend()

        self.ax[4].set_xlim([0, nr_scenes])
        self.ax[4].set_ylim([0, 1])
        self.ax[4].set_xlabel("Scenes")
        self.ax[4].title.set_text("Loss")
        self.ax[4].legend()

        self.fig.set_size_inches(10, 7)
        self.fig.tight_layout()
        plt.ion()  # Turn on interactive mode
        self.fig.show()
        self.fig.canvas.flush_events()

    def update_scene(self, mode, gradient_norm=None):
        if gradient_norm:
            self.ax[3].title.set_text(f"Training Progress: Last gradient norm={gradient_norm}")
        else:
            self.ax[3].title.set_text(mode + " Progress")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot(self, prediction, target, tp, fp, ls, dir, scene):
        # Directly update image layers
        self.prediction_layer.set_data(prediction.detach().cpu().numpy())
        self.prediction_layer.set_clim(vmin=0, vmax=1)
        self.train_layer.set_data(target.detach().cpu().numpy())
        self.train_layer.set_clim(vmin=0, vmax=1)

        # Update graph data
        self.graph_tp.set_data(np.arange(len(tp)), tp)
        self.graph_fp.set_data(np.arange(len(tp)), fp)
        if ls:
            self.ax[4].set_ylim([min(ls) - abs(min(ls)) * 0.2, max(ls)])
            self.graph_ls.set_data(np.arange(len(tp)), ls)

        # Force canvas update
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Save plot
        self.save_plot(dir, scene)

    def save_plot(self, dir, scene):
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
        self.fig.savefig(os.path.join(dir, "{sc}.pdf".format(sc=scene)), format="pdf")

        # Restore original size
        self.fig.set_size_inches(original_size)