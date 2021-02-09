import matplotlib.pyplot as plt
from matplotlib import gridspec

import argparse
from env import PLOTS_PATH
from face_reconstruction.pipeline import BFMPreprocessor
from face_reconstruction.plots import PlotManager, plot_params
from face_reconstruction.utils.io import list_file_numbering

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
args = parser.parse_args()


def load_params(frame_id):
    return plot_manager.load_params(f"params_{frame_id:04d}", preprocessor.bfm)


def load_param_history(frame_id):
    return plot_manager.load_param_history(f"param_history_{frame_id:04d}", preprocessor.bfm)


def plot_reconstruction_error(frame_id):
    preprocessor.load_frame(frame_id)
    preprocessor.to_3d()
    error = preprocessor.plot_reconstruction_error(load_params(frame_id))
    plt.xlabel(f"Mean Reconstruction Error: {error:.3f}")


def plot_rgb(frame_id):
    preprocessor.load_frame(frame_id)
    plt.imshow(preprocessor.img)


def plot_depth(frame_id):
    preprocessor.load_frame(frame_id)
    plt.imshow(preprocessor.depth_img)


def plot_mask(frame_id):
    preprocessor.load_frame(frame_id)
    preprocessor.to_3d()
    img_with_mask = preprocessor.render_onto_img(load_params(frame_id))
    plt.imshow(img_with_mask)


def generate_param_history_video(frame_id):
    ph = load_param_history(frame_id)
    preprocessor.load_frame(frame_id)
    preprocessor.to_3d()
    preprocessor.store_param_history(plot_manager, f"param_history/{frame_id}/", ph)
    plot_manager.cd(f"param_history/{frame_id}/").generate_video('iteration_', '.jpg')


def get_frames():
    return list_file_numbering(f"{PLOTS_PATH}/video_reconstruction/{args.run_name}", 'frame_', '.jpg')


if __name__ == '__main__':
    plot_manager = PlotManager(f"video_reconstruction/{args.run_name}")
    preprocessor = BFMPreprocessor()


    gs = gridspec.GridSpec(2, 4, height_ratios=[3, 1])
    gs2 = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

    for frame_id in get_frames():
        fig = plt.figure(figsize=(20, 10))

        fig.add_subplot(gs[0])
        plt.title('RGB Input')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                        labelleft=False)
        plot_rgb(frame_id)

        fig.add_subplot(gs[1])
        plt.title('Depth Input')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                        labelleft=False)
        plot_depth(frame_id)

        fig.add_subplot(gs[2])
        plt.title('Fitted Mask')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                        labelleft=False)
        plot_mask(frame_id)

        fig.add_subplot(gs[3])
        plt.title('Reconstruction Error')
        plt.xlim(0, 150)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                        labelleft=False)
        plot_reconstruction_error(frame_id)

        fig.add_subplot(gs2[2])
        plt.title("Shape Coefficients")
        plt.ylim(-10, 10)
        plot_params(load_params(frame_id).shape_coefficients)

        fig.add_subplot(gs2[3])
        plt.title("Expression Coefficients")
        plt.ylim(-10, 10)
        plot_params(load_params(frame_id).expression_coefficients, color='orange')

        plot_manager.save_current_plot(f"final/frame_{frame_id:04d}.jpg")

        # plt.show()
        plt.close()

    plot_manager.cd("final").generate_video('frame_', '.jpg')
