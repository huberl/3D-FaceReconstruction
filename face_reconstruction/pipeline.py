import numpy as np
import matplotlib.pyplot as plt
import open3d
import pyrender
from sklearn.neighbors import NearestNeighbors
from scipy import optimize
from face_reconstruction.data.iphone import IPhoneDataLoader
from face_reconstruction.model import BaselFaceModel
from face_reconstruction.landmarks import load_bfm_landmarks, detect_landmarks
from face_reconstruction.graphics import draw_pixels_to_image, register_rgb_depth, backproject_points, \
    interpolate_around, SimpleImageRenderer, setup_standard_scene, get_perspective_camera, backproject_image
from face_reconstruction.optim import BFMOptimization, run_icp, NearestNeighborMode, DistanceType, nearest_neighbors
from face_reconstruction.utils.math import add_column, geometric_median
from math import sin, cos
from numpy.linalg import det


class Pipeline:

    def __init__(self, frame_id):
        loader = IPhoneDataLoader()
        frame = loader.get_frame(frame_id)
        self.img = frame.get_color_image()
        self.depth_img = frame.get_depth_image()
        self.img_width = loader.get_image_width()
        self.img_height = loader.get_image_height()
        self.intrinsics = frame.get_intrinsics()
        depth_threshold = 0.5  # Drop all points behind that threshold

        self.intrinsics = frame.get_intrinsics()
        points = backproject_image(self.intrinsics, self.depth_img)
        points_to_render = points[:, :3]
        points_to_render *= 1000  # meter to millimeter
        self.colors = self.img.reshape(-1, 3)  # Just flatten color image

        foreground_mask = self.depth_img.reshape(-1) < depth_threshold
        self.pointcloud = points_to_render[foreground_mask]
        self.colors = self.colors[foreground_mask]
        self.bfm = BaselFaceModel.from_h5("model2019_face12.h5")

    def landmark_and_sparse_reconstruction(
            self,
            n_params_shape_sparse,
            n_params_expression_sparse,
            weight_shape_params_sparse,
            weight_expression_params_sparse,
            l2_regularization_sparse,
            initial_params):

        print("###  Landmark detection ###")
        bfm_landmarks = load_bfm_landmarks("model2019_face12_landmarks_v2")
        bfm_landmark_indices = np.array(list(bfm_landmarks.values()))
        n_shape_coefficients = self.bfm.get_n_shape_coefficients()
        n_expression_coefficients = self.bfm.get_n_expression_coefficients()
        n_color_coefficients = self.bfm.get_n_color_coefficients()

        landmarks_img, face_pos = detect_landmarks(self.img, return_face_pos=True)
        face_pos = face_pos[0]  # Assume there is only one face
        rgb_depth_img = self.depth_img
        # As RGB and depth channels are not aligned, we might not have exact depth information for every pixel in the color channel. Hence, we have to interpolate
        interpolation_size = 1
        rgb_depth_values = [interpolate_around(rgb_depth_img, pixel, interpolation_size) for pixel in landmarks_img]
        landmark_points_3d = backproject_points(self.intrinsics, rgb_depth_values, landmarks_img)
        landmark_points_3d_render = np.array(landmark_points_3d)
        landmark_points_3d_render[:, 2] = -landmark_points_3d_render[:,
                                           2]  # Invert z-coordinate for easier rendering (landmarks will be right in front of camera)
        landmark_points_3d_render *= 1000  # meter to millimeter
        landmark_points_3d_median = geometric_median(landmark_points_3d_render)
        distances_from_median = np.linalg.norm(landmark_points_3d_render - landmark_points_3d_median, axis=1)
        threshold_landmark_deviation = 500  # It can happen that depth information is bad and back-projected landmark points are far away from the other. These should be ignored
        valid_landmark_points_3d = \
            np.where((np.array(rgb_depth_values) != 0) & (distances_from_median < threshold_landmark_deviation))[0]
        pixels_without_depth = 68 - len(valid_landmark_points_3d)
        if pixels_without_depth > 0:
            print(f"There are {pixels_without_depth} pixels without depth information.")
        face_depth_values = []
        face_pixels = []
        for x in range(face_pos.left(), face_pos.right() + 1):
            for y in range(face_pos.top(), face_pos.bottom() + 1):
                pixel = [x, y]
                face_depth_value = interpolate_around(rgb_depth_img, pixel, interpolation_size)
                if face_depth_value > 0:
                    face_depth_values.append(face_depth_value)
                    face_pixels.append(pixel)

        self.face_pointcloud = backproject_points(self.intrinsics, face_depth_values, face_pixels)
        self.face_pointcloud[:, 2] = -self.face_pointcloud[:, 2]
        self.face_pointcloud_colors = np.array([self.img[y, x] for x, y in face_pixels])
        self.face_pointcloud *= 1000  # Meters to Millimeters
        body_depth_values = []
        body_pixels = []
        for x in range(self.img_width):
            for y in range(self.img_height):
                if (x < face_pos.left() or x > face_pos.right()) or (y < face_pos.top() or y > face_pos.bottom()):
                    pixel = [x, y]
                    body_depth_value = interpolate_around(rgb_depth_img, pixel, interpolation_size)
                    if body_depth_value > 0:
                        body_depth_values.append(body_depth_value)
                        body_pixels.append(pixel)
        self.body_pointcloud = backproject_points(self.intrinsics, body_depth_values, body_pixels)
        self.body_pointcloud[:, 2] = -self.body_pointcloud[:, 2]
        self.body_pointcloud_colors = np.array([self.img[y, x] for x, y in body_pixels])
        self.body_pointcloud *= 1000  # Meters to Millimeters

        print("###  Sparse reconstruction ###")
        sparse_optimizer = BFMOptimization(self.bfm,
                                           n_params_shape=n_params_shape_sparse,
                                           n_params_expression=n_params_expression_sparse,
                                           weight_shape_params=weight_shape_params_sparse,
                                           weight_expression_params=weight_expression_params_sparse)
        self.initial_camera_pose = np.eye(4)  # position camera just in front of face
        # if initial params are not set, set all to zero
        if initial_params == None:
            initial_params = sparse_optimizer.create_parameters(
                [0 for _ in range(n_shape_coefficients)],
                [0 for _ in range(n_expression_coefficients)],
                self.initial_camera_pose)
        sparse_loss = sparse_optimizer.create_sparse_loss_3d(bfm_landmark_indices[valid_landmark_points_3d],
                                                             landmark_points_3d_render[valid_landmark_points_3d],
                                                             regularization_strength=l2_regularization_sparse)
        sparse_context = sparse_optimizer.create_optimization_context(sparse_loss, initial_params)
        result = sparse_context.run_optimization(sparse_loss, initial_params)
        params_sparse = sparse_context.create_parameters_from_theta(result.x)
        return params_sparse

    def dense_reconstruction(
            self,
            initial_params,
            nn_mode,  # FACE_VERTICES: every face vertex will be assigned its nearest neighbor in pointcloud
            # POINTCLOUD: every point in pointcloud will be assigned its nearest neighbor in face model
            distance_type,
            icp_iterations,
            optimization_steps_per_iteration,
            l2_regularization_dense,  # 100
            n_params_shape_dense,  # 20
            n_params_expression_dense,  # 10
            weight_shape_params_dense,  # 10000
            weight_expression_params_dense,  # 1000
    ):
        dense_optimizer = BFMOptimization(self.bfm,
                                          n_params_shape=n_params_shape_dense,
                                          n_params_expression=n_params_expression_dense,
                                          weight_shape_params=weight_shape_params_dense,
                                          weight_expression_params=weight_expression_params_dense)
        params, distances, _ = run_icp(dense_optimizer,
                                       self.face_pointcloud,
                                       self.bfm,
                                       initial_params.with_new_manager(dense_optimizer),
                                       max_iterations=icp_iterations,
                                       nearest_neighbor_mode=nn_mode,
                                       distance_type=distance_type,
                                       max_nfev=optimization_steps_per_iteration,
                                       l2_regularization=l2_regularization_dense)
        return params

    def render_face(self, params_render):
        face_mesh = self.bfm.draw_sample(
            shape_coefficients=params_render.shape_coefficients,
            expression_coefficients=params_render.expression_coefficients,
            color_coefficients=params_render.color_coefficients)
        return face_mesh

    def setup_scene(self, face_mesh, camera_pose, show_pointcloud, show_mask, show_pointcloud_face, cut_around_face):
        bfm_vertices = camera_pose @ add_column(face_mesh.vertices, 1).T
        distances, indices = nearest_neighbors(self.pointcloud, bfm_vertices[:3, :].T)
        pointcloud_mask = distances > cut_around_face

        perspective_camera = get_perspective_camera(self.intrinsics, self.img_width, self.img_height)
        scene = setup_standard_scene(perspective_camera)
        if show_pointcloud and show_pointcloud_face:
            scene.add(pyrender.Mesh.from_points(self.pointcloud[pointcloud_mask], colors=self.colors[pointcloud_mask]),
                      pose=self.initial_camera_pose)
        if show_mask:
            scene.add(pyrender.Mesh.from_trimesh(self.bfm.convert_to_trimesh(face_mesh)), pose=camera_pose)
        if not show_pointcloud and show_pointcloud_face:
            scene.add(pyrender.Mesh.from_points(self.face_pointcloud, colors=self.face_pointcloud_colors),
                      pose=self.initial_camera_pose)
        if show_pointcloud and not show_pointcloud_face:
            scene.add(pyrender.Mesh.from_points(self.body_pointcloud, colors=self.body_pointcloud_colors),
                      pose=self.initial_camera_pose)
        return scene


class BFMPreprocessor:

    def __init__(self):
        self.bfm = BaselFaceModel.from_h5("model2019_face12.h5")
        bfm_landmarks = load_bfm_landmarks("model2019_face12_landmarks_v2")
        self.bfm_landmark_indices = np.array(list(bfm_landmarks.values()))

        self.n_shape_coefficients = self.bfm.get_n_shape_coefficients()
        self.n_expression_coefficients = self.bfm.get_n_expression_coefficients()
        self.n_color_coefficients = self.bfm.get_n_color_coefficients()

        self.loader = IPhoneDataLoader()

    def load_frame(self, frame_id: int):

        # loader = BiwiDataLoader(run_id)
        frame = self.loader.get_frame(frame_id)

        self.img = frame.get_color_image()
        self.depth_img = frame.get_depth_image()
        self.img_width = self.loader.get_image_width()
        self.img_height = self.loader.get_image_height()
        self.intrinsics = frame.get_intrinsics()

        return self.img, self.depth_img, self.intrinsics

    def to_3d(self, img=None, depth_img=None, intrinsics=None):

        if img is None:
            img = self.img
        if depth_img is None:
            depth_img = self.depth_img
        if intrinsics is None:
            intrinsics = self.intrinsics

        depth_threshold = 0.5  # Drop all points behind that threshold

        points = backproject_image(intrinsics, depth_img)
        points_to_render = points[:, :3]
        points_to_render *= 1000  # meter to millimeter
        colors = img.reshape(-1, 3)  # Just flatten color image

        foreground_mask = depth_img.reshape(-1) < depth_threshold
        pointcloud = points_to_render[foreground_mask]
        colors = colors[foreground_mask]
        self.colors = colors

        pointcloud[:, 2] = -pointcloud[:,
                            2]  # Invert z-coordinate for easier rendering (point cloud will be right in front of camera)

        pc = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(pointcloud))
        pc.estimate_normals()
        pointcloud_normals = np.asarray(pc.normals)
        self.pointcloud = pointcloud

        return pointcloud, pointcloud_normals, colors

    def detect_landmarks(self, img, depth_img, intrinsics):
        landmarks_img, face_pos = detect_landmarks(img, return_face_pos=True)
        face_pos = face_pos[0]  # Assume there is only one face
        rgb_depth_img = depth_img

        # As RGB and depth channels are not aligned, we might not have exact depth information for every pixel in the color channel. Hence, we have to interpolate
        interpolation_size = 1
        rgb_depth_values = [interpolate_around(rgb_depth_img, pixel, interpolation_size) for pixel in landmarks_img]

        landmark_points_3d = backproject_points(intrinsics, rgb_depth_values, landmarks_img)
        landmark_points_3d_render = np.array(landmark_points_3d)
        landmark_points_3d_render[:, 2] = -landmark_points_3d_render[:,
                                           2]  # Invert z-coordinate for easier rendering (landmarks will be right in front of camera)

        landmark_points_3d_render *= 1000  # meter to millimeter

        landmark_points_3d_median = geometric_median(landmark_points_3d_render)
        distances_from_median = np.linalg.norm(landmark_points_3d_render - landmark_points_3d_median, axis=1)

        threshold_landmark_deviation = 500  # It can happen that depth information is bad and back-projected landmark points are far away from the other. These should be ignored
        valid_landmark_points_3d = \
            np.where((np.array(rgb_depth_values) != 0) & (distances_from_median < threshold_landmark_deviation))[0]

        pixels_without_depth = 68 - len(valid_landmark_points_3d)
        if pixels_without_depth > 0:
            print(f"There are {pixels_without_depth} pixels without depth information.")

        face_depth_values = []
        face_pixels = []
        for x in range(face_pos.left(), face_pos.right() + 1):
            for y in range(face_pos.top(), face_pos.bottom() + 1):
                pixel = [x, y]
                face_depth_value = interpolate_around(rgb_depth_img, pixel, interpolation_size)
                if face_depth_value > 0:
                    face_depth_values.append(face_depth_value)
                    face_pixels.append(pixel)

        face_pointcloud = backproject_points(intrinsics, face_depth_values, face_pixels)
        face_pointcloud[:, 2] = -face_pointcloud[:, 2]
        self.face_pointcloud_colors = np.array([img[y, x] for x, y in face_pixels])
        face_pointcloud *= 1000  # Meters to Millimeters
        self.face_pointcloud = face_pointcloud

        body_depth_values = []
        body_pixels = []
        for x in range(self.img_width):
            for y in range(self.img_height):
                if (x < face_pos.left() or x > face_pos.right()) or (y < face_pos.top() or y > face_pos.bottom()):
                    pixel = [x, y]
                    body_depth_value = interpolate_around(rgb_depth_img, pixel, interpolation_size)
                    if body_depth_value > 0:
                        body_depth_values.append(body_depth_value)
                        body_pixels.append(pixel)

        body_pointcloud = backproject_points(intrinsics, body_depth_values, body_pixels)
        body_pointcloud[:, 2] = -body_pointcloud[:, 2]
        self.body_pointcloud_colors = np.array([img[y, x] for x, y in body_pixels])
        body_pointcloud *= 1000  # Meters to Millimeters
        self.body_pointcloud = body_pointcloud

        return landmark_points_3d_render[valid_landmark_points_3d], self.bfm_landmark_indices[
            valid_landmark_points_3d], face_pointcloud, self.face_pointcloud_colors

    def get_initial_params(self, optimizer):
        initial_camera_pose = np.eye(4)
        initial_camera_pose[2, 3] = -100  # position face already on pointcloud

        if optimizer.rotation_mode == 'lie':
            theta = 0.0001
            initial_camera_pose[:3, :3] = np.array(
                [[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])
            assert abs(det(initial_camera_pose) - 1.0) < 0.00001

        initial_params = optimizer.create_parameters(
            [0 for _ in range(self.n_shape_coefficients)],
            [0 for _ in range(self.n_expression_coefficients)],
            initial_camera_pose
        )
        return initial_params

    def setup_scene(self, params_render, show_pointcloud=True, show_mask=True, show_pointcloud_face=False,
                    cut_around_face=4):
        face_mesh = self.bfm.draw_sample(
            shape_coefficients=params_render.shape_coefficients,
            expression_coefficients=params_render.expression_coefficients,
            color_coefficients=params_render.color_coefficients)

        bfm_vertices = params_render.camera_pose @ add_column(face_mesh.vertices, 1).T
        if cut_around_face is not None:
            distances, indices = nearest_neighbors(self.pointcloud, bfm_vertices[:3, :].T)
            pointcloud_mask = distances > cut_around_face
        else:
            pointcloud_mask = bfm_vertices is not None

        perspective_camera = get_perspective_camera(self.intrinsics, self.img_width, self.img_height)
        scene = setup_standard_scene(perspective_camera)
        if show_pointcloud and show_pointcloud_face:
            scene.add(pyrender.Mesh.from_points(self.pointcloud[pointcloud_mask], colors=self.colors[pointcloud_mask]))
        if show_mask:
            scene.add(pyrender.Mesh.from_trimesh(self.bfm.convert_to_trimesh(face_mesh)),
                      pose=params_render.camera_pose)
        if not show_pointcloud and show_pointcloud_face:
            scene.add(pyrender.Mesh.from_points(self.face_pointcloud, colors=self.face_pointcloud_colors))
        if show_pointcloud and not show_pointcloud_face:
            scene.add(pyrender.Mesh.from_points(self.body_pointcloud, colors=self.body_pointcloud_colors))
        return scene

    def render_onto_img(self, params):
        scene = self.setup_scene(params, show_pointcloud=False, show_mask=True)

        r = pyrender.OffscreenRenderer(self.img_width, self.img_height)
        color, depth = r.render(scene)
        r.delete()

        img_with_mask = np.array(self.img)
        img_with_mask[depth != 0] = color[depth != 0]
        return img_with_mask

    def plot_reconstruction_error(self, params):
        scene = self.setup_scene(params, show_pointcloud=False, show_mask=True)
        r = pyrender.OffscreenRenderer(self.img_width, self.img_height)
        color, depth = r.render(scene)
        r.delete()

        diff_img = np.abs(self.depth_img * 1000 - depth)
        diff_img[depth == 0] = None

        plt.imshow(diff_img)
        plt.colorbar()
        plt.clim(0, 50)

        return (diff_img[depth != 0] * diff_img[depth != 0]).mean()

    def store_param_history(self, plot_manager, folder, param_history):
        for i, params in enumerate(param_history):
            face_mesh = self.bfm.draw_sample(
                shape_coefficients=params.shape_coefficients,
                expression_coefficients=params.expression_coefficients,
                color_coefficients=[0 for _ in range(self.n_color_coefficients)])
            translation = np.zeros((4, 4))
            translation[2, 3] = -150

            perspective_camera = get_perspective_camera(self.intrinsics, self.img_width, self.img_height)
            scene = setup_standard_scene(perspective_camera)
            scene.add(pyrender.Mesh.from_points(self.pointcloud, colors=self.colors), pose=np.eye(4) + translation)
            scene.add(pyrender.Mesh.from_trimesh(self.bfm.convert_to_trimesh(face_mesh)),
                      pose=params.camera_pose + translation)

            r = pyrender.OffscreenRenderer(self.img_width * 2, self.img_height * 2)
            color, depth = r.render(scene)
            r.delete()

            plt.figure(figsize=(8, 12))
            plt.imshow(color)
            plot_manager.save_current_plot(f"{folder}/iteration_{i:05d}.jpg")
            # plt.show()
            plt.close()
