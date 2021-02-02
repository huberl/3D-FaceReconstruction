import numpy as np
import matplotlib.pyplot as plt
import pyrender
from sklearn.neighbors import NearestNeighbors
from scipy import optimize
from face_reconstruction.data.iphone import IPhoneDataLoader
from face_reconstruction.model import BaselFaceModel
from face_reconstruction.landmarks import load_bfm_landmarks, detect_landmarks
from face_reconstruction.graphics import draw_pixels_to_image, register_rgb_depth, backproject_points, interpolate_around, SimpleImageRenderer, setup_standard_scene, get_perspective_camera, backproject_image
from face_reconstruction.optim import BFMOptimization, run_icp, NearestNeighborMode, DistanceType, nearest_neighbors
from face_reconstruction.utils.math import add_column, geometric_median

class Pipeline:

    def __init__(self, frame_id ):
        loader = IPhoneDataLoader()
        frame = loader.get_frame(frame_id)
        self.img = frame.get_color_image()
        self.depth_img = frame.get_depth_image()
        self.img_width = loader.get_image_width()
        self.img_height = loader.get_image_height()
        self.intrinsics = frame.get_intrinsics()
        depth_threshold = 0.5 # Drop all points behind that threshold
    
        self.intrinsics = frame.get_intrinsics()
        points = backproject_image(self.intrinsics, self.depth_img)
        points_to_render = points[:, :3]
        points_to_render *= 1000 # meter to millimeter
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
        face_pos = face_pos[0] # Assume there is only one face
        rgb_depth_img = self.depth_img
        # As RGB and depth channels are not aligned, we might not have exact depth information for every pixel in the color channel. Hence, we have to interpolate
        interpolation_size = 1
        rgb_depth_values = [interpolate_around(rgb_depth_img, pixel, interpolation_size) for pixel in landmarks_img]
        landmark_points_3d = backproject_points(self.intrinsics, rgb_depth_values, landmarks_img)
        landmark_points_3d_render = np.array(landmark_points_3d)
        landmark_points_3d_render[:,2] = -landmark_points_3d_render[:,2]  # Invert z-coordinate for easier rendering (landmarks will be right in front of camera)
        landmark_points_3d_render *= 1000  # meter to millimeter
        landmark_points_3d_median = geometric_median(landmark_points_3d_render)
        distances_from_median = np.linalg.norm(landmark_points_3d_render - landmark_points_3d_median, axis=1)   
        threshold_landmark_deviation = 500  # It can happen that depth information is bad and back-projected landmark points are far away from the other. These should be ignored
        valid_landmark_points_3d = np.where((np.array(rgb_depth_values) != 0) & (distances_from_median < threshold_landmark_deviation))[0]
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
        self.initial_camera_pose = np.eye(4) # position camera just in front of face
        # if initial params are not set, set all to zero
        if initial_params==None: 
            initial_params=sparse_optimizer.create_parameters(
                [0 for _ in range(n_shape_coefficients)],
                [0 for _ in range(n_expression_coefficients)],
                self.initial_camera_pose)
        sparse_loss = sparse_optimizer.create_sparse_loss_3d(bfm_landmark_indices[valid_landmark_points_3d], landmark_points_3d_render[valid_landmark_points_3d], regularization_strength=l2_regularization_sparse)
        sparse_context = sparse_optimizer.create_optimization_context(sparse_loss, initial_params)
        result = sparse_context.run_optimization(sparse_loss, initial_params)
        params_sparse = sparse_context.create_parameters_from_theta(result.x)
        return params_sparse


    def dense_reconstruction(
        self,
        initial_params,
        nn_mode, # FACE_VERTICES: every face vertex will be assigned its nearest neighbor in pointcloud
                                                # POINTCLOUD: every point in pointcloud will be assigned its nearest neighbor in face model
    distance_type,
    icp_iterations,
    optimization_steps_per_iteration,
    l2_regularization_dense, # 100
    n_params_shape_dense, # 20
    n_params_expression_dense, # 10
    weight_shape_params_dense ,# 10000
    weight_expression_params_dense ,# 1000
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


    def render_face(self,params_render):
        face_mesh = self.bfm.draw_sample(
            shape_coefficients=params_render.shape_coefficients, 
            expression_coefficients=params_render.expression_coefficients, 
            color_coefficients=params_render.color_coefficients)
        return face_mesh

    def setup_scene(self,face_mesh,camera_pose,show_pointcloud, show_mask, show_pointcloud_face, cut_around_face):
        bfm_vertices = camera_pose @ add_column(face_mesh.vertices, 1).T
        distances, indices = nearest_neighbors(self.pointcloud, bfm_vertices[:3, :].T)
        pointcloud_mask = distances > cut_around_face
        
        perspective_camera = get_perspective_camera(self.intrinsics, self.img_width, self.img_height)
        scene = setup_standard_scene(perspective_camera)
        if show_pointcloud and show_pointcloud_face:
            scene.add(pyrender.Mesh.from_points(self.pointcloud[pointcloud_mask], colors=self.colors[pointcloud_mask]), pose=self.initial_camera_pose)
        if show_mask:
            scene.add(pyrender.Mesh.from_trimesh(self.bfm.convert_to_trimesh(face_mesh)), pose=camera_pose)
        if not show_pointcloud and show_pointcloud_face:
            scene.add(pyrender.Mesh.from_points(self.face_pointcloud, colors=self.face_pointcloud_colors), pose=self.initial_camera_pose)
        if show_pointcloud and not show_pointcloud_face:
            scene.add(pyrender.Mesh.from_points(self.body_pointcloud, colors=self.body_pointcloud_colors), pose=self.initial_camera_pose)
        return scene
