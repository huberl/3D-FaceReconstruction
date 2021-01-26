import numpy as np
from imageio import imread

from env import BIWI_DATASET_PATH
from face_reconstruction.utils.io import list_file_numbering


class BiwiDataLoader:
    """
    Class for easy loading of BIWI Kinect Dataset files.
    A loader corresponds to a single run which in turn consists of several frames.
    For each run, the camera intrinsics and extrinsics of both depth and color cameras can be obtained.
    """

    def __init__(self, run_id: int):
        self.dataset_path = BIWI_DATASET_PATH
        self.run_id = run_id

        depth_intrinsics = np.zeros((3, 3))
        with open(f"{self.get_run_path()}/depth.cal", 'r') as f:
            depth_intrinsics[0] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
            depth_intrinsics[1] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
            depth_intrinsics[2] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
            for _ in range(9):
                f.readline()
            width, height = [int(number) for number in f.readline().rstrip(' \n').split(' ')]

        self.depth_intrinsics = depth_intrinsics

        rgb_extrinsic_rotation = np.zeros((3, 3))
        rgb_intrinsics = np.zeros((3, 3))
        with open(f"{self.get_run_path()}/rgb.cal", 'r') as f:
            rgb_intrinsics[0] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
            rgb_intrinsics[1] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
            rgb_intrinsics[2] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
            for _ in range(3):
                f.readline()
            rgb_extrinsic_rotation[0] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
            rgb_extrinsic_rotation[1] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
            rgb_extrinsic_rotation[2] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
            f.readline()
            extrinsic_translation = np.array([float(number) for number in f.readline().rstrip(' \n').split(' ')])

        self.rgb_intrinsics = rgb_intrinsics
        self.rgb_extrinsic_rotation = rgb_extrinsic_rotation
        self.rgb_extrinsic_translation = extrinsic_translation
        rgb_extrinsics = np.eye(4)
        rgb_extrinsics[0:3, 0:3] = rgb_extrinsic_rotation
        rgb_extrinsics[0:3, 3] = extrinsic_translation
        self.rgb_extrinsics = rgb_extrinsics

        self.image_width = width
        self.image_height = height

    def get_frame(self, frame_id):
        return BiwiFrame(self.get_run_path(), frame_id)

    def get_frame_ids(self):
        return list_file_numbering(f"{self.get_run_path()}", 'frame_', '_rgb.png')

    def get_run_path(self):
        return f"{self.dataset_path}/{self.run_id:02d}"

    def get_image_width(self):
        return self.image_width

    def get_image_height(self):
        return self.image_height

    def get_depth_intrinsics(self):
        return self.depth_intrinsics

    def get_rgb_intrinsics(self):
        return self.rgb_intrinsics

    def get_rgb_extrinsics(self):
        return self.rgb_extrinsics


class BiwiFrame:
    """
    Represents a single frame of the BIWI Kinect Dataset. A frame consists of a color and a depth image.
    """

    def __init__(self, run_path: str, frame_id: int):
        self.color_image = imread(f"{run_path}/frame_{frame_id:05d}_rgb.png")
        # Depth values are given in millimeters
        with open(f"{run_path}/frame_{frame_id:05d}_depth.bin", 'rb') as f:
            width = int.from_bytes(f.read(4), byteorder='little')
            height = int.from_bytes(f.read(4), byteorder='little')

            depth_image = np.zeros(height * width)
            p = 0
            while (p < width * height):
                num_empty = int.from_bytes(f.read(4), byteorder='little')
                p += num_empty
                num_full = int.from_bytes(f.read(4), byteorder='little')
                for i in range(num_full):
                    depth_image[p + i] = float(int.from_bytes(f.read(2), byteorder='little'))
                p += num_full

        self.depth_image = depth_image.reshape((height, width))

    def get_depth_image(self):
        return np.array(self.depth_image)

    def get_color_image(self):
        return np.array(self.color_image)
