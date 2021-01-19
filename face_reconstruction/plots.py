import matplotlib.pyplot as plt
from face_reconstruction.utils.io import create_directories

from env import PLOTS_PATH


class PlotManager:

    def __init__(self, plot_group: str):
        self.plot_group = plot_group

    def save_current_plot(self, plot_name: str):
        plot_name = plot_name.lower().strip().replace(' ', '_')
        file_path = f"{PLOTS_PATH}/{self.plot_group}/{plot_name}"
        create_directories(file_path)
        plt.savefig(file_path)
