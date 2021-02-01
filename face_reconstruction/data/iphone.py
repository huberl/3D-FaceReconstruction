from env import IPHONE_DATASET_PATH
import numpy as np
from PIL import Image

from face_reconstruction.utils.io import list_file_numbering


class IPhoneDataLoader:

    def __init__(self):
        self.dataset_path = IPHONE_DATASET_PATH

        # depth_intrinsics = np.zeros((3, 3))
        # with open(f"{self.get_run_path()}/depth.cal", 'r') as f:
        #     depth_intrinsics[0] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
        #     depth_intrinsics[1] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
        #     depth_intrinsics[2] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
        #     for _ in range(9):
        #         f.readline()
        #     width, height = [int(number) for number in f.readline().rstrip(' \n').split(' ')]
        #
        # self.depth_intrinsics = depth_intrinsics
        #
        # rgb_extrinsic_rotation = np.zeros((3, 3))
        # rgb_intrinsics = np.zeros((3, 3))
        # with open(f"{self.get_run_path()}/rgb.cal", 'r') as f:
        #     rgb_intrinsics[0] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
        #     rgb_intrinsics[1] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
        #     rgb_intrinsics[2] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
        #     for _ in range(3):
        #         f.readline()
        #     rgb_extrinsic_rotation[0] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
        #     rgb_extrinsic_rotation[1] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
        #     rgb_extrinsic_rotation[2] = [float(number) for number in f.readline().rstrip(' \n').split(' ')]
        #     f.readline()
        #     extrinsic_translation = np.array([float(number) for number in f.readline().rstrip(' \n').split(' ')])
        #
        # self.rgb_intrinsics = rgb_intrinsics
        # self.rgb_extrinsic_rotation = rgb_extrinsic_rotation
        # self.rgb_extrinsic_translation = extrinsic_translation
        # rgb_extrinsics = np.eye(4)
        # rgb_extrinsics[0:3, 0:3] = rgb_extrinsic_rotation
        # rgb_extrinsics[0:3, 3] = extrinsic_translation
        # self.rgb_extrinsics = rgb_extrinsics

        image = Image.open(f"{self.dataset_path}/images/image_0.jpg")
        width, height = image.size

        self.image_width = width
        self.image_height = height

    def get_frame(self, frame_id):
        return IPhoneFrame(self.dataset_path, frame_id)

    def get_frame_ids(self):
        return list_file_numbering(f"{self.dataset_path}/images", 'image_', '.jpg')

    def get_image_width(self):
        return self.image_width

    def get_image_height(self):
        return self.image_height


class IPhoneFrame:
    def __init__(self, dataset_path: str, frame_id: int):
        color_image = Image.open(f"{dataset_path}/images/image_{frame_id}.jpg")
        self.color_image = np.asarray(color_image)

        self.depth_image = np.load(f"{dataset_path}/depth/depth_{frame_id}.npy")

        self.intrinsics = np.load(f"{dataset_path}/intrinsics/intrinsic_{frame_id}.npy")

    def get_depth_image(self):
        return np.array(self.depth_image)

    def get_color_image(self):
        return np.array(self.color_image)

    def get_intrinsics(self):
        return np.array(self.intrinsics)
