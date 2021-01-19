import numpy as np
from pyquaternion import Quaternion

from face_reconstruction.graphics import SimpleImageRenderer
from face_reconstruction.model import BaselFaceModel

"""
Optimization is split into 3 modules:
    - Optimization manager (SparseOptimization):
        facilitates creating the parameter and loss modules needed for optimization.
        The typical workflow is:
            1) Initialize optimization manager by specifying which parameters you want to optimize for
            2) Create initial parameter module
            3) Use `parameter.to_theta()` to get the list of parameters to optimize for
            4) Create loss module
            5) Start optimizing wrt loss and initial parameters
            6) In the loss function you can use the `create_parameters_from_theta()` method to get easier access
               to all the parameters in the theta parameter list
    
    - Loss (SparseOptimizationLoss):
        Encapsulates the loss calculation. Any external dependencies such as landmark indices etc. are injected in the
        constructor. The loss takes a list of parameters to optimize for (theta) and returns a list of residuals
        
    - Parameters (SparseOptimizationParameters):
        Syntactic sugar. Provides an interface to translate between a list of parameters (theta) and an object of 
        meaningful attributes, e.g., divided into shape coefficients, expression coefficients and camera pose
        
"""


class SparseOptimization:
    def __init__(self,
                 bfm: BaselFaceModel,
                 n_params_shape,
                 n_params_expression,
                 weight_shape_params=1.0,
                 weight_expression_params=1.0):
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
        :param weight_shape_params:
            Specifies how much more changing a shape coefficient parameter will impact the loss
        :param weight_expression_params:
            Specifies how much more changing an expression coefficient parameter will impact the loss
        """
        self.bfm = bfm
        self.n_params_shape = n_params_shape
        self.n_params_expression = n_params_expression
        self.weight_shape_params = weight_shape_params
        self.weight_expression_params = weight_expression_params

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

    def create_parameters_from_theta(self, theta: np.ndarray):
        return SparseOptimizationParameters.from_theta(self, theta)

    def create_parameters(self,
                          shape_coefficients: np.ndarray,
                          expression_coefficients: np.ndarray,
                          camera_pose: np.ndarray):
        return SparseOptimizationParameters(
            self,
            shape_coefficients=shape_coefficients,
            expression_coefficients=expression_coefficients,
            camera_pose=camera_pose)

    def create_loss(self, renderer, bfm_landmark_indices, img_landmark_pixels):
        return SparseOptimizationLoss(
            self,
            renderer=renderer,
            bfm_landmark_indices=bfm_landmark_indices,
            img_landmark_pixels=img_landmark_pixels)


class SparseOptimizationLoss:

    def __init__(
            self,
            optimization_manager: SparseOptimization,
            renderer: SimpleImageRenderer,
            bfm_landmark_indices: np.ndarray,
            img_landmark_pixels: np.ndarray):
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
        self.optimization_manager = optimization_manager
        self.renderer = renderer
        self.bfm_landmark_indices = bfm_landmark_indices
        self.img_landmark_pixels = img_landmark_pixels

    def __call__(self, *args, **kwargs):
        return self.loss(args[0], args[1:], kwargs)

    def loss(self, theta, *args, **kwargs):
        # global residuals_before
        parameters = self.optimization_manager.create_parameters_from_theta(theta)
        # print(parameters.camera_pose)
        # print(parameters.shape_coefficients)
        face_mesh = self.optimization_manager.bfm.draw_sample(
            shape_coefficients=parameters.shape_coefficients,
            expression_coefficients=parameters.expression_coefficients,
            color_coefficients=[0 for _ in range(self.optimization_manager.n_color_coefficients)])
        landmark_points = np.array(face_mesh.vertices)[self.bfm_landmark_indices]
        face_landmark_pixels = self.renderer.project_points(parameters.camera_pose, landmark_points)
        residuals = face_landmark_pixels - self.img_landmark_pixels

        # residuals = np.linalg.norm(residuals, axis=1)
        residuals = residuals.reshape(-1)
        # residuals = (residuals ** 2).sum()
        return residuals


class SparseOptimizationParameters:

    def __init__(self,
                 optimization_manager: SparseOptimization,
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
        n_params_shape = optimization_manager.n_params_shape
        n_params_expression = optimization_manager.n_params_expression

        # Shape and expression coefficients are multiplied by their weight to enforce that changing them
        # will have a higher impact depending on the weight
        self.shape_coefficients = np.hstack(
            [shape_coefficients[:optimization_manager.n_params_shape],
             np.zeros((n_shape_coefficients - n_params_shape))])
        self.expression_coefficients = np.hstack([expression_coefficients[:n_params_expression],
                                                  np.zeros((n_expression_coefficients - n_params_expression))])
        self.camera_pose = camera_pose

    @staticmethod
    def from_theta(optimization_manager: SparseOptimization, theta: np.ndarray):
        """
        :param theta:
            Contains a list of parameters that are interpreted as follows.
            The 1st `n_shape_params` are shape coefficients
            The next `n_expression_params` are expression coefficients
            The final 7 parameters are the quaternion defining the camera rotation (4 params) and the translation (3 params)
        """
        n_params_shape = optimization_manager.n_params_shape
        n_params_expression = optimization_manager.n_params_expression

        shape_coefficients = theta[:n_params_shape] * optimization_manager.weight_shape_params
        expression_coefficients = theta[
                                  n_params_shape:n_params_shape + n_params_expression] * optimization_manager.weight_expression_params
        i = n_params_shape + n_params_expression

        # TODO: Enforcing unity at Quaternion does not yet yield desired effect
        q = Quaternion(*theta[i:i + 4]).unit  # Important that we only allow unit quaternions
        camera_pose = q.transformation_matrix
        i = i + 4
        camera_pose[:3, 3] = theta[i:i + 3]

        return SparseOptimizationParameters(
            optimization_manager=optimization_manager,
            shape_coefficients=shape_coefficients,
            expression_coefficients=expression_coefficients,
            camera_pose=camera_pose)

    def to_theta(self):
        theta = []
        # To translate the parameters back into a theta list, shape and expression coefficients have to be divided
        # again by their weights
        theta.extend(self.shape_coefficients[:self.optimization_manager.n_params_shape]
                     / self.optimization_manager.weight_shape_params)
        theta.extend(self.expression_coefficients[:self.optimization_manager.n_params_expression]
                     / self.optimization_manager.weight_expression_params)
        q = Quaternion(matrix=self.camera_pose)
        theta.extend(q)
        theta.extend(self.camera_pose[:3, 3])

        return np.array(theta)
