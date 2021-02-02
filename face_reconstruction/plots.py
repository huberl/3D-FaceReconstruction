import matplotlib.pyplot as plt
from face_reconstruction.utils.io import create_directories, generate_run_name

from env import PLOTS_PATH


class PlotManager:

    def __init__(self, plot_group: str):
        self.plot_group = plot_group

    def save_current_plot(self, plot_name: str):
        plot_name = plot_name.lower().strip().replace(' ', '_')
        file_path = f"{PLOTS_PATH}/{self.plot_group}/{plot_name}"
        create_directories(file_path)
        plt.savefig(file_path)

    @staticmethod
    def new_run(plot_group: str, prefix: str = 'run'):
        run_name = generate_run_name(f"{PLOTS_PATH}/{plot_group}", prefix)
        return PlotManager(f"{plot_group}/{run_name}")
