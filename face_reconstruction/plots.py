import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from env import PLOTS_PATH
from face_reconstruction.model import BaselFaceModel
from face_reconstruction.optim import BFMOptimizationParameters
from face_reconstruction.utils.io import create_directories, generate_run_name, save_pickled, load_pickled


class PlotManager:

    def __init__(self, plot_group: str):
        self.plot_group = plot_group
        self.folder = f"{PLOTS_PATH}/{self.plot_group}"

    def cd(self, sub_dir: str):
        return PlotManager(f"{self.plot_group}/{sub_dir}")

    def save_current_plot(self, plot_name: str):
        plot_name = plot_name.lower().strip().replace(' ', '_')
        file_path = f"{self.folder}/{plot_name}"
        create_directories(file_path)
        plt.savefig(file_path)

    def save_image(self, img: np.ndarray, plot_name: str):
        plot_name = plot_name.lower().strip().replace(' ', '_')
        file_path = f"{self.folder}/{plot_name}"
        create_directories(file_path)
        plt.imsave(file_path, img)

    def save(self, obj, file_name: str):
        save_pickled(obj, f"{self.folder}/{file_name}")

    def load(self, file_name: str):
        return load_pickled(f"{self.folder}/{file_name}")

    def save_params(self, params: BFMOptimizationParameters, file_name: str):
        params.dump(f"{self.folder}/{file_name}")

    def load_params(self, file_name: str, bfm: BaselFaceModel):
        return BFMOptimizationParameters.load(f"{self.folder}/{file_name}", bfm)

    def save_param_history(self, param_history, file_name: str):
        bfm = param_history[0].optimization_manager.bfm if len(param_history) > 0 else None
        param_history_without_bfm = []
        for params in param_history:
            params.optimization_manager.bfm = None
            param_history_without_bfm.append(params)
        save_pickled(param_history_without_bfm, f"{self.folder}/{file_name}")
        for params in param_history:
            params.optimization_manager.bfm = bfm

    def load_param_history(self, file_name: str, bfm: BaselFaceModel):
        param_history = load_pickled(f"{self.folder}/{file_name}")
        for param in param_history:
            param.optimization_manager.bfm = bfm
        return param_history

    @staticmethod
    def new_run(plot_group: str, prefix: str = 'run'):
        run_name = generate_run_name(f"{PLOTS_PATH}/{plot_group}", prefix)
        return PlotManager(f"{plot_group}/{run_name}")

    def generate_video(self, prefix, suffix, video_name='sequence.mp4', fps=5):
        img_array = []
        file_names = glob.glob(f"{self.folder}/{prefix}*{suffix}")
        assert len(file_names), f"No files found in {self.folder}"
        for filename in file_names:
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(f"{self.folder}/{video_name}", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def plot_params(params, **kwargs):
    params_count = np.count_nonzero(params)
    params_stripped = params[:params_count]
    langs = [i for i in range(params_count)]
    plt.bar(langs, params_stripped, **kwargs)


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
