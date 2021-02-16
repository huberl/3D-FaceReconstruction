from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Union
from deprecated import deprecated

import trimesh
from scipy import optimize
import numpy as np
from pyquaternion import Quaternion
from numpy.linalg import det

from face_reconstruction.graphics import SimpleImageRenderer
from face_reconstruction.model import BaselFaceModel
from face_reconstruction.optim.nn import NearestNeighborMode
from face_reconstruction.utils.math import add_column
from face_reconstruction.optim.lie import SE3_to_se3, se3_to_SE3

"""
Optimization is split into 3 modules:
    - Optimization manager (BFMOptimization):
        facilitates creating the parameter and loss modules needed for optimization.
        The typical workflow is:
            1) Initialize optimization manager by specifying which parameters you want to optimize for
            2) Create initial parameter module
            3) Use `parameter.to_theta()` to get the list of parameters to optimize for
            4) Create loss module
            5) Start optimizing wrt loss and initial parameters
            6) In the loss function you can use the `create_parameters_from_theta()` method to get easier access
               to all the parameters in the theta parameter list
    
    - Loss (BFMOptimizationLoss):
        Encapsulates the loss calculation. Any external dependencies such as landmark indices etc. are injected in the
        constructor. The loss takes a list of parameters to optimize for (theta) and returns a list of residuals
        
    - Parameters (BFMOptimizationParameters):
        Syntactic sugar. Provides an interface to translate between a list of parameters (theta) and an object of 
        meaningful attributes, e.g., divided into shape coefficients, expression coefficients and camera pose
        
"""


class DistanceType(Enum):
    POINT_TO_POINT = auto()
    POINT_TO_PLANE = auto()


# =========================================================================
# Optimization Manager
# =========================================================================

class BFMOptimization:
    def __init__(self,
                 bfm: BaselFaceModel,
                 n_params_shape,
                 n_params_expression,
                 fix_camera_pose=False,
                 weight_shape_params=1.0,
                 weight_expression_params=1.0,
                 rotation_mode='lie'):
        """

        :param bfm:
            the basel face model
        :param n_params_shape:
            Specifies that only the first `n_params_shape` parameters of the shape model will be optimized for.
            These are the parameters that have the biggest impact on the face model.
            The remaining coefficients will be held constant to 0.
        :param n_params_expression:
            Specifies that only the first `n_params_expression` parameters of the expression model will be optimized for.
            These are the parameters that have the biggest impact on the face model.
            The remaining coefficients will be held constant to 0.
        :param fix_camera_pose:
            Whether the camera pose should be optimized for
        :param weight_shape_params:
            Specifies how much more changing a shape coefficient parameter will impact the loss
        :param weight_expression_params:
            Specifies how much more changing an expression coefficient parameter will impact the loss
        """
        self.bfm = bfm
        self.n_params_shape = n_params_shape
        self.n_params_expression = n_params_expression
        self.n_params_color = 0  # Currently, optimizing for color is not supported
        self.fix_camera_pose = fix_camera_pose
        self.weight_shape_params = weight_shape_params
        self.weight_expression_params = weight_expression_params

        assert rotation_mode in ['quaternion', 'lie'], f'Rotation mode has to be either lie or quaternion. ' \
                                                       f'You gave {rotation_mode}'
        self.rotation_mode = rotation_mode

        self.n_shape_coefficients = bfm.get_n_shape_coefficients()
        self.n_expression_coefficients = bfm.get_n_expression_coefficients()
        self.n_color_coefficients = bfm.get_n_color_coefficients()

        lower_bounds = []
        lower_bounds.extend([-float('inf') for _ in range(n_params_shape)])
        lower_bounds.extend([-float('inf') for _ in range(n_params_expression)])
        lower_bounds.extend([-1, -1, -1, -1, -float('inf'), -float('inf'), -float('inf')])
        self.lower_bounds = np.array(lower_bounds)

        upper_bounds = []
        upper_bounds.extend([float('inf') for _ in range(n_params_shape)])
        upper_bounds.extend([float('inf') for _ in range(n_params_expression)])
        upper_bounds.extend([1, 1, 1, 1, float('inf'), float('inf'), float('inf')])
        self.upper_bounds = np.array(upper_bounds)

    @deprecated("Use BFMOptimizationContext to run the optimization. "
                "Otherwise, you might loose fixed/initial params as you lack the context object.")
    def run_optimization(self, loss, initial_params, max_nfev=100, verbose=2, x_scale='jac'):
        context = self.create_optimization_context(loss, initial_params, max_nfev=max_nfev, verbose=verbose,
                                                   x_scale=x_scale)
        result = context.run_optimization()
        return result

    @deprecated("Use BFMOptimizationContext to create parameters from theta. "
                "Otherwise, you might loose fixed/initial params.")
    def create_parameters_from_theta(self, theta: np.ndarray):
        return BFMOptimizationParameters.from_theta(self, theta)

    @staticmethod
    def create_optimization_context(loss, initial_params, max_nfev=100, verbose=2, x_scale='jac'):
        context = BFMOptimizationContext(loss, initial_params, max_nfev=max_nfev, verbose=verbose, x_scale=x_scale)
        return context

    def create_parameters(self,
                          shape_coefficients: np.ndarray = None,
                          expression_coefficients: np.ndarray = None,
                          camera_pose: np.ndarray = None
                          ):
        return BFMOptimizationParameters(
            self,
            shape_coefficients=shape_coefficients,
            expression_coefficients=expression_coefficients,
            camera_pose=camera_pose)

    def create_parameters_from_other(self, parameters):
        return parameters.with_new_manager(self)

    def create_sparse_loss(self,
                           renderer,
                           bfm_landmark_indices,
                           img_landmark_pixels,
                           fixed_camera_pose: np.ndarray = None,
                           fixed_shape_coefficients: np.ndarray = None,
                           fixed_expression_coefficients: np.ndarray = None,
                           regularization_strength: float = None
                           ):
        return SparseOptimizationLoss(
            self,
            renderer=renderer,
            bfm_landmark_indices=bfm_landmark_indices,
            img_landmark_pixels=img_landmark_pixels,
            fixed_camera_pose=fixed_camera_pose,
            fixed_shape_coefficients=fixed_shape_coefficients,
            fixed_expression_coefficients=fixed_expression_coefficients,
            regularization_strength=regularization_strength)

    def create_sparse_loss_3d(self,
                              bfm_landmark_indices,
                              img_landmark_points_3d,
                              regularization_strength: float = None):
        return SparseOptimizationLoss3D(
            optimization_manager=self,
            bfm_landmark_indices=bfm_landmark_indices,
            img_landmark_points_3d=img_landmark_points_3d,
            regularization_strength=regularization_strength
        )

    def create_dense_loss_3d(self,
                             pointcloud: np.ndarray,
                             nearest_neighbors: np.ndarray,
                             nearest_neighbor_mode: NearestNeighborMode,
                             distance_type: DistanceType,
                             regularization_strength: float = None,
                             pointcloud_normals: np.ndarray = None):
        return DenseOptimizationLoss3D(self, pointcloud=pointcloud, nearest_neighbors=nearest_neighbors,
                                       nearest_neighbor_mode=nearest_neighbor_mode,
                                       distance_type=distance_type, regularization_strength=regularization_strength,
                                       pointcloud_normals=pointcloud_normals)

    def create_combined_loss_3d(self,
                                bfm_landmark_indices,
                                img_landmark_points_3d,
                                pointcloud: np.ndarray,
                                nearest_neighbors: np.ndarray,
                                nearest_neighbor_mode: NearestNeighborMode,
                                distance_type: DistanceType,
                                weight_sparse_term: float = 1,
                                regularization_strength: float = None,
                                pointcloud_normals: np.ndarray = None
                                ):
        return CombinedLoss3D(self, bfm_landmark_indices=bfm_landmark_indices,
                              img_landmark_points_3d=img_landmark_points_3d,
                              pointcloud=pointcloud, nearest_neighbors=nearest_neighbors,
                              nearest_neighbor_mode=nearest_neighbor_mode,
                              distance_type=distance_type, weight_sparse_term=weight_sparse_term,
                              regularization_strength=regularization_strength, pointcloud_normals=pointcloud_normals)


class BFMOptimizationRGB(BFMOptimization):
    def create_dense_loss_3d(
        self,
        pointcloud: np.ndarray,
        nearest_neighbors: np.ndarray,
        nearest_neighbor_mode: NearestNeighborMode,
        distance_type: DistanceType,
        regularization_strength: float = None,
        pointcloud_normals: np.ndarray = None,
    ):
        return RGBLoss3D(
            self,
            pointcloud=pointcloud,
            nearest_neighbors=nearest_neighbors,
            nearest_neighbor_mode=nearest_neighbor_mode,
            distance_type=distance_type,
            regularization_strength=regularization_strength,
            pointcloud_normals=pointcloud_normals,
        )

# =========================================================================
# Optimization Context
# =========================================================================

class BFMOptimizationContext:
    """
    Encapsulates the context of a single optimization run. Needed in order to hold the initial parameters as
    the theta list that is communicated to the optimizer only contains the parameters that are to be optimized.
    Fixed parameters, i.e., those specified as initial parameters that are outside the n_shape_coefficients or
    n_expression_coefficients bounds, can then be obtained from this context wrapper.
    """

    def __init__(self, loss, initial_params, max_nfev=100, verbose=2, x_scale='jac'):
        self.loss = loss
        self.initial_params = initial_params
        self.max_nfev = max_nfev
        self.verbose = verbose
        self.x_scale = x_scale
        self.param_history = []
        self._previous_theta = None

    def run_optimization(self, *args, **kwargs):
        self.loss.set_context(self)
        self.param_history = []
        self._previous_theta = None
        result = optimize.least_squares(self.loss,
                                        self.initial_params.to_theta(),
                                        max_nfev=self.max_nfev,
                                        verbose=self.verbose,
                                        x_scale=self.x_scale)
        return result

    def log_param(self, parameters):
        theta = parameters.to_theta()

        # Detect Optimization iterations by comparing the number of variables changed to the previous loss function
        # call. scipy optimize usually tries all the parameters once separately and then updates all of them.
        # Only when all of them are updated, the parameters are added to the history
        if self._previous_theta is not None and np.sum((theta - self._previous_theta) != 0) == len(
                self._previous_theta):
            self.param_history.append(parameters)
        self._previous_theta = theta

    def get_param_history(self):
        return list(self.param_history)

    def create_parameters_from_theta(self, theta):
        return BFMOptimizationParameters.from_theta(self, theta)


# =========================================================================
# Loss functions
# =========================================================================

class BFMOptimizationLoss(ABC):
    def __init__(self, optimization_manager: BFMOptimization, regularization_strength):
        self.optimization_manager = optimization_manager
        self.regularization_strength = regularization_strength
        self.context = None

    def __call__(self, *args, **kwargs):
        return self.loss(args[0], args[1:], kwargs)

    def set_context(self, context: BFMOptimizationContext):
        self.context = context

    def create_parameters_from_theta(self, theta):
        if self.context is None:
            print("Loss function should be used from within OptimizationRunner")
            return self.optimization_manager.create_parameters_from_theta(theta)
        else:
            return self.context.create_parameters_from_theta(theta)

    @abstractmethod
    def loss(self, theta, *args, **kwargs):
        pass

    def _apply_params_to_model(self, theta):
        parameters = self.create_parameters_from_theta(theta)
        if self.context is not None:
            self.context.log_param(parameters)

        shape_coefficients = parameters.shape_coefficients
        expression_coefficients = parameters.expression_coefficients
        camera_pose = parameters.camera_pose

        face_mesh = self.optimization_manager.bfm.draw_sample(
            shape_coefficients=shape_coefficients,
            expression_coefficients=expression_coefficients,
            color_coefficients=[0 for _ in range(self.optimization_manager.n_color_coefficients)])
        bfm_vertices = add_column(np.array(face_mesh.vertices), 1)
        bfm_vertices = camera_pose @ bfm_vertices.T
        return bfm_vertices.T, face_mesh

    def _compute_regularization_terms(self, params):
        regularization_terms = []
        if self.regularization_strength is not None:
            # Regularization on shape coefficients
            n_params_shape = self.optimization_manager.n_params_shape
            if n_params_shape > 0:
                shape_coefficients_variance = self.optimization_manager.bfm.get_shape_coefficients_variance()
                shape_regularization = params.shape_coefficients[:n_params_shape]
                shape_regularization /= np.sqrt(shape_coefficients_variance[:n_params_shape])  # Divide by standard dev
                regularization_terms.extend(shape_regularization)

            # Regularization on expression coefficients
            n_params_expression = self.optimization_manager.n_params_expression
            if n_params_expression > 0:
                expression_coefficients_variance = self.optimization_manager.bfm.get_expression_coefficients_variance()
                expression_regularization = params.expression_coefficients[:n_params_expression]
                expression_regularization /= np.sqrt(
                    expression_coefficients_variance[:n_params_expression])  # Divide by standard dev
                regularization_terms.extend(expression_regularization)

            # Regularization on color coefficients
            n_params_color = self.optimization_manager.n_params_color
            if n_params_color > 0:
                color_coefficients_variance = self.optimization_manager.bfm.get_color_coefficients_variance()
                color_regularization = params.color_coefficients[:n_params_color]
                color_regularization /= np.sqrt(
                    color_coefficients_variance[:n_params_color])  # Divide by standard dev
                regularization_terms.extend(color_regularization)

            regularization_terms = self.regularization_strength * np.array(regularization_terms)
        return regularization_terms


class SparseOptimizationLoss(BFMOptimizationLoss):

    def __init__(
            self,
            optimization_manager: BFMOptimization,
            renderer: SimpleImageRenderer,
            bfm_landmark_indices: np.ndarray,
            img_landmark_pixels: np.ndarray,
            fixed_camera_pose: np.ndarray = None,
            fixed_shape_coefficients: np.ndarray = None,
            fixed_expression_coefficients: np.ndarray = None,
            regularization_strength=None):
        """
        :param optimization_manager:
            the optimization manager that specifies how the optimization should be performed
        :param renderer:
            a renderer object that is used to project the 3D face model landmarks to raster space
        :param bfm_landmark_indices:
            a list of Basel Face Model vertex indices that describe the landmarks
        :param img_landmark_pixels:
            a list of 2D pixels that describe where the corresponding landmarks are in the image
        """
        super(SparseOptimizationLoss, self).__init__(optimization_manager, regularization_strength)
        self.renderer = renderer
        self.bfm_landmark_indices = bfm_landmark_indices
        self.img_landmark_pixels = img_landmark_pixels

        assert not optimization_manager.fix_camera_pose or fixed_camera_pose is not None, \
            "If camera pose is fixed, it has to be provided to the loss"
        self.fixed_camera_pose = fixed_camera_pose

        self.fixed_shape_coefficients = fixed_shape_coefficients
        self.fixed_expression_coefficients = fixed_expression_coefficients

    def loss(self, theta, *args, **kwargs):
        parameters = self.create_parameters_from_theta(theta)

        if self.fixed_shape_coefficients is None:
            shape_coefficients = parameters.shape_coefficients
        else:
            shape_coefficients = self.fixed_shape_coefficients

        if self.fixed_expression_coefficients is None:
            expression_coefficients = parameters.expression_coefficients
        else:
            expression_coefficients = self.fixed_expression_coefficients

        if self.optimization_manager.fix_camera_pose:
            camera_pose = self.fixed_camera_pose
        else:
            camera_pose = parameters.camera_pose

        face_mesh = self.optimization_manager.bfm.draw_sample(
            shape_coefficients=shape_coefficients,
            expression_coefficients=expression_coefficients,
            color_coefficients=[0 for _ in range(self.optimization_manager.n_color_coefficients)])
        landmark_points = np.array(face_mesh.vertices)[self.bfm_landmark_indices]
        face_landmark_pixels = self.renderer.project_points(camera_pose, landmark_points)
        residuals = face_landmark_pixels - self.img_landmark_pixels

        # residuals = np.linalg.norm(residuals, axis=1)
        residuals = residuals.reshape(-1)

        if self.regularization_strength is not None:
            regularization_terms = self._compute_regularization_terms(parameters)
            residuals = np.hstack((residuals, regularization_terms))
        # residuals = (residuals ** 2).sum()

        return residuals


class SparseOptimizationLoss3D(BFMOptimizationLoss):
    """
    This loss optimizes a basel face model to fit a given set of 3D landmarks
    """

    def __init__(
            self,
            optimization_manager: BFMOptimization,
            bfm_landmark_indices: np.ndarray,
            img_landmark_points_3d: np.ndarray,
            regularization_strength: float = None):
        super(SparseOptimizationLoss3D, self).__init__(optimization_manager, regularization_strength)
        self.bfm_landmark_indices = bfm_landmark_indices
        self.img_landmark_points_3d = img_landmark_points_3d

    def loss(self, theta, *args, **kwargs):
        bfm_vertices, _ = self._apply_params_to_model(theta)
        landmark_points = bfm_vertices[self.bfm_landmark_indices]

        # Simple point-to-point distance of 3D landmarks
        residuals = landmark_points[:, :3] - self.img_landmark_points_3d
        residuals = residuals.reshape(-1)

        if self.regularization_strength is not None:
            regularization_terms = self._compute_regularization_terms(
                self.create_parameters_from_theta(theta))
            residuals = np.hstack((residuals, regularization_terms))

        return residuals

    def __call__(self, *args, **kwargs):
        return self.loss(args[0], args[1:], kwargs)


class DenseOptimizationLoss3D(BFMOptimizationLoss):
    def __init__(
            self,
            optimization_manager: BFMOptimization,
            pointcloud: np.ndarray,
            nearest_neighbors: np.ndarray,
            nearest_neighbor_mode: NearestNeighborMode,
            distance_type: DistanceType,
            regularization_strength: float = None,
            pointcloud_normals: np.ndarray = None):
        super(DenseOptimizationLoss3D, self).__init__(optimization_manager, regularization_strength)

        # assert distance_type == DistanceType.POINT_TO_PLANE or nearest_neighbor_mode.FACE_VERTICES, \
        #     f"point-to-plane distance only works when computing nearest neighbors for face vertices"

        self.pointcloud = pointcloud
        self.nearest_neighbors = nearest_neighbors
        self.nearest_neighbor_mode = nearest_neighbor_mode
        self.distance_type = distance_type
        self.pointcloud_normals = pointcloud_normals

    def loss(self, theta, *args, **kwargs):
        bfm_vertices, face_mesh = self._apply_params_to_model(theta)
        bfm_vertices = bfm_vertices[:, :3]

        residuals = []
        if self.distance_type == DistanceType.POINT_TO_PLANE:
            face_trimesh = trimesh.Trimesh(
                vertices=bfm_vertices,
                faces=face_mesh.tvi)

            if self.nearest_neighbor_mode == NearestNeighborMode.FACE_VERTICES:
                distances = self.pointcloud[self.nearest_neighbors] - bfm_vertices
                residuals.extend(np.sum(face_trimesh.vertex_normals * distances, axis=1))
                if self.pointcloud_normals is not None:
                    # Can do symmetric point-to-plane
                    residuals.extend(np.sum(self.pointcloud_normals[self.nearest_neighbors] * distances, axis=1))
            elif self.nearest_neighbor_mode == NearestNeighborMode.POINTCLOUD:
                distances = self.pointcloud - bfm_vertices[self.nearest_neighbors]
                residuals.extend(np.sum(face_trimesh.vertex_normals[self.nearest_neighbors] * distances, axis=1))
                if self.pointcloud_normals is not None:
                    # Can do symmetric point-to-plane
                    residuals.extend(np.sum(self.pointcloud_normals * distances, axis=1))
            else:
                raise ValueError(f"Unknown nearest_neighbor_mode: {self.nearest_neighbor_mode}")
        elif self.distance_type == DistanceType.POINT_TO_POINT:
            # Simple point-to-point distance of vertices and corresponding nearest neighbors
            if self.nearest_neighbor_mode == NearestNeighborMode.FACE_VERTICES:
                residuals = bfm_vertices - self.pointcloud[self.nearest_neighbors]
            elif self.nearest_neighbor_mode == NearestNeighborMode.POINTCLOUD:
                residuals = bfm_vertices[self.nearest_neighbors] - self.pointcloud
            else:
                raise ValueError(f"Unknown nearest_neighbor_mode: {self.nearest_neighbor_mode}")
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

        residuals = np.array(residuals).reshape(-1)
        if self.regularization_strength is not None:
            regularization_terms = self._compute_regularization_terms(
                self.create_parameters_from_theta(theta))
            residuals = np.hstack((residuals, regularization_terms))

        return residuals


class CombinedLoss3D(BFMOptimizationLoss):

    def __init__(self, optimization_manager: BFMOptimization,
                 bfm_landmark_indices: np.ndarray,
                 img_landmark_points_3d: np.ndarray,
                 pointcloud: np.ndarray,
                 nearest_neighbors: np.ndarray,
                 nearest_neighbor_mode: NearestNeighborMode,
                 distance_type: DistanceType,
                 weight_sparse_term: float = 1,
                 regularization_strength: float = None,
                 pointcloud_normals: np.ndarray = None):
        super(CombinedLoss3D, self).__init__(optimization_manager, regularization_strength)
        self.bfm_landmark_indices = bfm_landmark_indices
        self.img_landmark_points_3d = img_landmark_points_3d
        self.pointcloud = pointcloud
        self.nearest_neighbors = nearest_neighbors
        self.nearest_neighbor_mode = nearest_neighbor_mode
        self.distance_type = distance_type
        self.pointcloud_normals = pointcloud_normals
        self.weight_sparse_term = weight_sparse_term

    def loss(self, theta, *args, **kwargs):
        bfm_vertices, face_mesh = self._apply_params_to_model(theta)
        bfm_vertices = bfm_vertices[:, :3]

        residuals = []
        if self.distance_type == DistanceType.POINT_TO_PLANE:
            face_trimesh = trimesh.Trimesh(
                vertices=bfm_vertices,
                faces=face_mesh.tvi)

            if self.nearest_neighbor_mode == NearestNeighborMode.FACE_VERTICES:
                distances = self.pointcloud[self.nearest_neighbors] - bfm_vertices
                residuals.extend(np.sum(face_trimesh.vertex_normals * distances, axis=1))
                if self.pointcloud_normals is not None:
                    # Can do symmetric point-to-plane
                    residuals.extend(np.sum(self.pointcloud_normals[self.nearest_neighbors] * distances, axis=1))
            elif self.nearest_neighbor_mode == NearestNeighborMode.POINTCLOUD:
                distances = self.pointcloud - bfm_vertices[self.nearest_neighbors]
                residuals.extend(np.sum(face_trimesh.vertex_normals[self.nearest_neighbors] * distances, axis=1))
                if self.pointcloud_normals is not None:
                    # Can do symmetric point-to-plane
                    residuals.extend(np.sum(self.pointcloud_normals * distances, axis=1))
            else:
                raise ValueError(f"Unknown nearest_neighbor_mode: {self.nearest_neighbor_mode}")
        elif self.distance_type == DistanceType.POINT_TO_POINT:
            # Simple point-to-point distance of vertices and corresponding nearest neighbors
            if self.nearest_neighbor_mode == NearestNeighborMode.FACE_VERTICES:
                residuals.extend(bfm_vertices - self.pointcloud[self.nearest_neighbors])
            elif self.nearest_neighbor_mode == NearestNeighborMode.POINTCLOUD:
                residuals.extend(bfm_vertices[self.nearest_neighbors] - self.pointcloud)
            else:
                raise ValueError(f"Unknown nearest_neighbor_mode: {self.nearest_neighbor_mode}")
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

        # Sparse residuals
        landmark_points = bfm_vertices[self.bfm_landmark_indices]
        residuals.extend(self.weight_sparse_term * (landmark_points[:, :3] - self.img_landmark_points_3d))

        residuals = np.array(residuals).reshape(-1)
        if self.regularization_strength is not None:
            regularization_terms = self._compute_regularization_terms(
                self.create_parameters_from_theta(theta))
            residuals = np.hstack((residuals, regularization_terms))

        return residuals


class RGBLoss3D(DenseOptimizationLoss3D):
    """
    """
    def loss(self, theta, *args, **kwargs):
        bfm_vertices, face_mesh = self._apply_params_to_model(theta)
        bfm_vertices = bfm_vertices[:, :3]

        residuals = []
        # currently passing only:
        # nearest neighbor mode: POINTCLOUD & distance: POINT_TO_POINT
        residuals = np.array(face_mesh.colors)[self.nearest_neighbors] - (
            self.pointcloud/255.0)
        residuals = np.array(residuals).reshape(-1)
        if self.regularization_strength is not None:
            regularization_terms = self._compute_regularization_terms(
                self.create_parameters_from_theta(theta),
            )
            residuals = np.hstack((residuals, regularization_terms))
        return residuals


# =========================================================================
# Optimization Parameters
# =========================================================================

class BFMOptimizationParameters:

    def __init__(self,
                 optimization_manager: BFMOptimization,
                 shape_coefficients: np.ndarray,
                 expression_coefficients: np.ndarray,
                 camera_pose: np.ndarray):
        """
        Defines all the parameters that will be optimized for
        :param optimization_manager:
            Specifies how many shape and coefficients parameters the optimization will use
        :param shape_coefficients:
            The part of the parameters that describes the shape coefficients
        :param expression_coefficients:
            The part of the parameters that describes the expression coefficients
        :param camera_pose:
            The part of the parameters that describes the 4x4 camera pose matrix
        """
        self.optimization_manager = optimization_manager

        n_shape_coefficients = optimization_manager.n_shape_coefficients
        n_expression_coefficients = optimization_manager.n_expression_coefficients
        n_color_coefficients = optimization_manager.n_color_coefficients
        n_params_shape = optimization_manager.n_params_shape
        n_params_expression = optimization_manager.n_params_expression

        assert shape_coefficients is not None or n_params_shape == 0, "If n_params_shape > 0 then shape coefficients have to be provided"
        if shape_coefficients is None:
            shape_coefficients = []

        assert expression_coefficients is not None or n_params_expression == 0, "If n_params_expression > 0 then expression coefficients have to be provided"
        if expression_coefficients is None:
            expression_coefficients = []
        # Shape and expression coefficients are multiplied by their weight to enforce that changing them
        # will have a higher impact depending on the weight
        self.shape_coefficients = np.hstack(
            [shape_coefficients,
             np.zeros((n_shape_coefficients - len(shape_coefficients)))])
        self.expression_coefficients = np.hstack([expression_coefficients,
                                                  np.zeros((n_expression_coefficients - len(expression_coefficients)))])
        self.color_coefficients = np.zeros(n_color_coefficients)

        assert camera_pose is not None or optimization_manager.fix_camera_pose, "Camera pose may only be None if it is fixed"
        if camera_pose is not None:
            self.camera_pose = np.array(camera_pose, dtype=np.float)

    @staticmethod
    def from_theta(optimization_manager: Union[BFMOptimization, BFMOptimizationContext], theta: np.ndarray):
        """
        :param optimization_manager:
            Specifies how the optimization should be done
        :param theta:
            Contains a list of parameters that are interpreted as follows.
            The 1st `n_shape_params` are shape coefficients
            The next `n_expression_params` are expression coefficients
            The final 7 parameters are the quaternion defining the camera rotation (4 params) and the translation (3 params)
        """
        context = None
        if isinstance(optimization_manager, BFMOptimizationContext):
            context = optimization_manager
            optimization_manager = context.loss.optimization_manager

        n_params_shape = optimization_manager.n_params_shape
        n_params_expression = optimization_manager.n_params_expression

        if context is None:
            # No access to initial or fixed parameters
            # Just reconstruct coefficients from what is there in the theta list
            shape_coefficients = theta[:n_params_shape] * optimization_manager.weight_shape_params
            expression_coefficients = theta[n_params_shape:n_params_shape + n_params_expression] \
                                      * optimization_manager.weight_expression_params
        else:
            # Access to initial or fixed parameters
            # Combine initial parameters and what is there in the theta list
            shape_coefficients = context.initial_params.shape_coefficients
            shape_coefficients[:n_params_shape] = theta[:n_params_shape] * optimization_manager.weight_shape_params
            expression_coefficients = context.initial_params.expression_coefficients
            expression_coefficients[:n_params_expression] = theta[n_params_shape:n_params_shape + n_params_expression] \
                                                            * optimization_manager.weight_expression_params
        i = n_params_shape + n_params_expression

        if optimization_manager.fix_camera_pose:
            camera_pose = None
        else:
            mode = optimization_manager.rotation_mode
            # TODO: Enforcing unity at Quaternion does not yet yield desired effect
            if mode == 'quaternion':
                q = Quaternion(*theta[i:i + 4]).unit  # Important that we only allow unit quaternions
                camera_pose = q.transformation_matrix
                i = i + 4
                camera_pose[:3, 3] = theta[i:i + 3]
                i = i + 3
                camera_pose[:3, :3] *= theta[i]
            elif mode == 'lie':
                w = theta[i:i + 3]
                v = theta[i + 3:i + 6]
                camera_pose = se3_to_SE3(w, v)

        return BFMOptimizationParameters(
            optimization_manager=optimization_manager,
            shape_coefficients=shape_coefficients,
            expression_coefficients=expression_coefficients,
            camera_pose=camera_pose)

    def with_new_manager(self, optimization_manager: BFMOptimization):
        return BFMOptimizationParameters(
            optimization_manager=optimization_manager,
            shape_coefficients=self.shape_coefficients,
            expression_coefficients=self.expression_coefficients,
            camera_pose=self.camera_pose
        )

    def to_theta(self):
        theta = []
        # To translate the parameters back into a theta list, shape and expression coefficients have to be divided
        # again by their weights
        theta.extend(self.shape_coefficients[:self.optimization_manager.n_params_shape]
                     / self.optimization_manager.weight_shape_params)
        theta.extend(self.expression_coefficients[:self.optimization_manager.n_params_expression]
                     / self.optimization_manager.weight_expression_params)

        if not self.optimization_manager.fix_camera_pose:
            mode = self.optimization_manager.rotation_mode
            if mode == 'quaternion':
                scaling_factor = np.linalg.norm(self.camera_pose[0, :3])
                rotation_matrix = np.array(self.camera_pose[:3, :3])
                rotation_matrix /= scaling_factor
                if det(rotation_matrix) < 0:
                    print('IT HAS HAPPENED!')
                    rotation_matrix = -rotation_matrix
                q = Quaternion(matrix=rotation_matrix)
                theta.extend(q)
                theta.extend(self.camera_pose[:3, 3])
                theta.append(scaling_factor)
            elif mode == 'lie':
                w, v = SE3_to_se3(self.camera_pose)
                theta.extend(w)
                theta.extend(v)
            else:
                raise NotImplementedError(f'Rotation mode {mode} is not implemented!')

        return np.array(theta)
