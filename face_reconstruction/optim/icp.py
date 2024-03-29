from typing import List

import numpy as np

from face_reconstruction.model import BaselFaceModel
from face_reconstruction.optim import KeyframeOptimizationParameters
from face_reconstruction.optim.nn import NearestNeighborMode, nearest_neighbors
from face_reconstruction.optim.bfm import BFMOptimization, BFMOptimizationParameters, DistanceType
from face_reconstruction.utils.math import add_column


def run_icp(optimizer: BFMOptimization,
            pointcloud: np.ndarray,
            face_model: BaselFaceModel,
            initial_params: BFMOptimizationParameters,
            max_iterations=20,
            max_nfev=20,
            tolerance=0.001,
            nearest_neighbor_mode=NearestNeighborMode.FACE_VERTICES,
            distance_type=DistanceType.POINT_TO_POINT,
            l2_regularization: float = None,
            pointcloud_normals: np.ndarray = None,
            return_cost_history=False):
    """
    Performs ICP to fit the given face_model as closely as possible to the given pointcloud.
    ICP consists of two phases:
        1) Finding correspondences between point cloud and face model vertices by searching for nearest neighbors
        2) Optimizing parameters to minimize distance between corresponding points
    The whole procedure is repeated at most max_nfev times.
    For phase 1) there are two options:
        either for every face mesh vertex its nearest neighbor in the pointcloud is searched (NearestNeighborMode.FACE_VERTICES)
        or for every point in the pointcloud its nearest neighbor among the face mesh vertices is searched (NearestNeighborMode.POINTCLOUD)
    For phase 2) there are two options as well:
        either the point-to-point distance between corresponding points is minimized (DistanceType.POINT_TO_POINT)
        or the point-to-plane distance between the pointcloud point and the face mesh vertex is minimized (DistanceType.POINT_TO_PLANE)
    Additionally, L2-Regularization may be passed to the underlying loss function.

    Parameters
    ----------
        optimizer:
            optimization manager that handles the Dense 3D Face Reconstruction
        pointcloud:
            (n, 3) list of points that constitute the optimization target
        face_model:
            the basel face model from which face meshes can be sampled conditioned on coefficients
        initial_params:
            the initial parameter guess
        max_iterations:
            how many iterations of ICP should be performed.
        max_nfev:
            how many steps one optimization in phase 2 may take at most
        tolerance:
            lower bound to change in error. ICP is stopped if change falls below
        nearest_neighbor_mode:
            specifies how nearest neighbors are found in phase 1
        distance_type:
            specifies how distances are calculated in phase 2
        l2_regularization:
            specifies the extent of L2-regularization that will be imposed on the face model coefficients in phase 2
        pointcloud_normals:
            if provided and distance_type = POINT_TO_PLANE, then symmetric point-to-plane will be computed for
            the dense reconstruction, i.e., both the face mesh plane as well as planes on the pointcloud will be used
        return_cost_history:
            Whether the cost history of the full ICP optimization should be returned. The cost history is a list of
            floats representing the cost of the Optimization logged every time the optimizer evaluates the loss function

    Returns
    -------
        the parameters after optimization
        the final distances
        the number of iterations needed

    """
    loss_factory = lambda indices: \
        optimizer.create_dense_loss_3d(pointcloud,
                                       indices,
                                       nearest_neighbor_mode=nearest_neighbor_mode,
                                       distance_type=distance_type,
                                       regularization_strength=l2_regularization,
                                       pointcloud_normals=pointcloud_normals)
    return _run_icp(optimizer=optimizer,
                    loss_factory=loss_factory,
                    pointcloud=pointcloud,
                    face_model=face_model,
                    initial_params=initial_params,
                    max_iterations=max_iterations,
                    max_nfev=max_nfev,
                    tolerance=tolerance,
                    nearest_neighbor_mode=nearest_neighbor_mode,
                    return_cost_history=return_cost_history)


def run_icp_combined(optimizer: BFMOptimization,
                     bfm_landmark_indices,
                     img_landmark_points_3d,
                     pointcloud: np.ndarray,
                     face_model: BaselFaceModel,
                     initial_params: BFMOptimizationParameters,
                     weight_sparse_term=1,
                     max_iterations=20,
                     max_nfev=20,
                     tolerance=0.001,
                     nearest_neighbor_mode=NearestNeighborMode.FACE_VERTICES,
                     distance_type=DistanceType.POINT_TO_POINT,
                     l2_regularization: float = None,
                     pointcloud_normals: np.ndarray = None,
                     return_cost_history=False):
    """
    Runs ICP with a combined loss function (Sparse + Dense term)
    """
    loss_factory = lambda indices: \
        optimizer.create_combined_loss_3d(bfm_landmark_indices, img_landmark_points_3d,
                                          pointcloud,
                                          indices,
                                          nearest_neighbor_mode=nearest_neighbor_mode,
                                          distance_type=distance_type,
                                          weight_sparse_term=weight_sparse_term,
                                          regularization_strength=l2_regularization,
                                          pointcloud_normals=pointcloud_normals)
    return _run_icp(optimizer=optimizer,
                    loss_factory=loss_factory,
                    pointcloud=pointcloud,
                    face_model=face_model,
                    initial_params=initial_params,
                    max_iterations=max_iterations,
                    max_nfev=max_nfev,
                    tolerance=tolerance,
                    nearest_neighbor_mode=nearest_neighbor_mode,
                    return_cost_history=return_cost_history)


def _run_icp(optimizer: BFMOptimization,
             loss_factory,
             pointcloud: np.ndarray,
             face_model: BaselFaceModel,
             initial_params: BFMOptimizationParameters,
             max_iterations=20,
             max_nfev=20,
             tolerance=0.001,
             nearest_neighbor_mode=NearestNeighborMode.FACE_VERTICES,
             return_cost_history=False):
    params = initial_params

    prev_error = 0
    distances = None
    param_history = []
    cost_history = []
    for i in range(max_iterations):
        face_mesh = face_model.draw_sample(
            shape_coefficients=params.shape_coefficients,
            expression_coefficients=params.expression_coefficients,
            color_coefficients=params.color_coefficients)

        # find the nearest neighbors between the current source and destination points
        transformed_face_vertices = params.camera_pose @ add_column(face_mesh.vertices, 1).T
        transformed_face_vertices = transformed_face_vertices.T[:, :3]
        if nearest_neighbor_mode == NearestNeighborMode.FACE_VERTICES:
            distances, indices = nearest_neighbors(transformed_face_vertices, pointcloud)
        elif nearest_neighbor_mode == NearestNeighborMode.POINTCLOUD:
            distances, indices = nearest_neighbors(pointcloud, transformed_face_vertices)
        else:
            raise ValueError(f"Unknown nearest_neighbor_mode: {nearest_neighbor_mode}")

        # Create 3D loss function
        loss = loss_factory(indices)

        # Optimize distance between face mesh vertices and their nearest neighbors
        context = optimizer.create_optimization_context(loss, params, max_nfev=max_nfev)
        result = context.run_optimization(loss, params, max_nfev=max_nfev)
        params = context.create_parameters_from_theta(result.x)
        param_history.extend(context.get_param_history())
        cost_history.extend(context.get_cost_history())

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    if return_cost_history:
        return params, distances, param_history, cost_history
    else:
        return params, distances, param_history


def run_icp_keyframes(optimizer: BFMOptimization,
                      pointclouds: List[np.ndarray],
                      face_model: BaselFaceModel,
                      initial_params,
                      max_iterations=20,
                      max_nfev=20,
                      tolerance=0.001,
                      nearest_neighbor_mode=NearestNeighborMode.FACE_VERTICES,
                      distance_type=DistanceType.POINT_TO_POINT,
                      l2_regularization: float = None,
                      pointcloud_normals: List[np.ndarray] = None):
    params = initial_params

    prev_error = 0
    distances = None
    param_history = []
    for i in range(max_iterations):
        nearest_neighbors_list = []
        for j in range(len(pointclouds)):
            face_mesh = face_model.draw_sample(
                shape_coefficients=params.shape_coefficients,
                expression_coefficients=params.expression_coefficients,
                color_coefficients=params.color_coefficients)

            # find the nearest neighbors between the current source and destination points
            transformed_face_vertices = params.camera_poses[j] @ add_column(face_mesh.vertices, 1).T
            transformed_face_vertices = transformed_face_vertices.T[:, :3]
            if nearest_neighbor_mode == NearestNeighborMode.FACE_VERTICES:
                distances, indices = nearest_neighbors(transformed_face_vertices, pointclouds[j])
            elif nearest_neighbor_mode == NearestNeighborMode.POINTCLOUD:
                distances, indices = nearest_neighbors(pointclouds[j], transformed_face_vertices)
            else:
                raise ValueError(f"Unknown nearest_neighbor_mode: {nearest_neighbor_mode}")
            nearest_neighbors_list.append(indices)

        # Create 3D loss function
        keyframe_loss = optimizer.create_dense_keyframe_loss(pointclouds, nearest_neighbors_list, nearest_neighbor_mode,
                                                             distance_type,
                                                             regularization_strength=l2_regularization,
                                                             pointcloud_normals_list=pointcloud_normals)

        # Optimize distance between face mesh vertices and their nearest neighbors
        context = optimizer.create_optimization_context(keyframe_loss, params, max_nfev=max_nfev)
        result = context.run_optimization(keyframe_loss, max_nfev=max_nfev)
        params = KeyframeOptimizationParameters.from_theta(context, theta=result.x)
        param_history.extend(context.get_param_history())

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return params, distances, param_history
