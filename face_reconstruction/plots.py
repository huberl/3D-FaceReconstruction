import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from env import PLOTS_PATH
from face_reconstruction.utils.io import create_directories, generate_run_name


class PlotManager:

    def __init__(self, plot_group: str):
        self.plot_group = plot_group
        self.folder = f"{PLOTS_PATH}//{self.plot_group}"

    def save_current_plot(self, plot_name: str):
        plot_name = plot_name.lower().strip().replace(' ', '_')
        file_path = f"{self.folder}/{plot_name}"
        create_directories(file_path)
        plt.savefig(file_path)

    @staticmethod
    def new_run(plot_group: str, prefix: str = 'run'):
        run_name = generate_run_name(f"{PLOTS_PATH}/{plot_group}", prefix)
        return PlotManager(f"{plot_group}/{run_name}")

    def generate_video(self, prefix, suffix, video_name='sequence.mp4', fps=5):
        img_array = []
        for filename in glob.glob(f"{self.folder}/{prefix}*{suffix}"):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(f"{self.folder}/{video_name}", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def plot_params(params):
    params_count = np.count_nonzero(params)
    params_stripped = params[:params_count]
    langs = [i for i in range(params_count)]
    plt.bar(langs, params_stripped)
    plt.show()


def plot_reconstruction_error(image_depth, model_depth):
    height, length = np.shape(image_depth)
    error_image = np.empty((height, length, 3))
    color_error = cm.get_cmap('Reds', 50)
    for i in range(height):
        for j in range(length):
            if model_depth[i][j] != 0:
                error = np.abs(image_depth[i][j] * 1000 - model_depth[i][j])
                error_image[i][j] = color_error(error)[:3]
            else:
                error_image[i][j] = (255, 255, 255)
    return error_image
