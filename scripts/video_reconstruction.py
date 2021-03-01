from face_reconstruction.optim import NearestNeighborMode, DistanceType, run_icp, BFMOptimization, \
    KeyframeOptimizationParameters, run_icp_keyframes, BFMOptimizationParameters, run_icp_combined
from face_reconstruction.pipeline import BFMPreprocessor
from face_reconstruction.plots import PlotManager

preprocessor = BFMPreprocessor()

reuse_previous_frame = False
keyframes = [0, 1, 2, 3, 4, 5]

# -------------------------------------------------------------------------
# Sparse Reconstruction Params
# -------------------------------------------------------------------------

n_params_shape_sparse = 30
n_params_expression_sparse = 30
weight_shape_params_sparse = 1000
weight_expression_params_sparse = 1000
l2_regularization_sparse = 1000

# -------------------------------------------------------------------------
# Dense Reconstruction Params
# -------------------------------------------------------------------------

nn_mode = NearestNeighborMode.FACE_VERTICES  # FACE_VERTICES: every face vertex will be assigned its nearest neighbor in pointcloud
# POINTCLOUD: every point in pointcloud will be assigned its nearest neighbor in face model
distance_type = DistanceType.POINT_TO_POINT
icp_iterations = 3
optimization_steps_per_iteration = 10
l2_regularization_dense = 4000  # 10000 for Lie

n_params_shape_dense = 30  # 20
n_params_expression_dense = 30  # 10
weight_shape_params_dense = 100  # 10000, 10000000000 for POINT_TO_PLANE
weight_expression_params_dense = 100  # 1000, 10000000000 for POINT_TO_PLANE

weight_sparse_term = 10


def preprocess(frame_id):
    img, depth_img, intrinsics = preprocessor.load_frame(frame_id)
    pointcloud, pointcloud_normals, colors = preprocessor.to_3d(img, depth_img, intrinsics)
    landmark_points, bfm_landmark_indices, face_pointcloud, face_pointcloud_colors = \
        preprocessor.detect_landmarks(img,
                                      depth_img,
                                      intrinsics,
                                      threshold_landmark_deviation=200,
                                      ignore_jawline=True)
    return bfm_landmark_indices, landmark_points, face_pointcloud, pointcloud_normals


def run_sparse_optimization(sparse_optimizer, bfm_landmark_indices, landmark_points, initial_params):
    sparse_loss = sparse_optimizer.create_sparse_loss_3d(bfm_landmark_indices, landmark_points,
                                                         regularization_strength=l2_regularization_sparse)
    sparse_context = sparse_optimizer.create_optimization_context(sparse_loss, initial_params)
    result = sparse_context.run_optimization(sparse_loss, initial_params)
    return sparse_context.create_parameters_from_theta(result.x)


def run_dense_optimization(dense_optimizer, bfm_landmark_indices, landmark_points, face_pointcloud, pointcloud_normals,
                           params_sparse):
    params, distances, dense_param_history = run_icp_combined(dense_optimizer,
                                                              bfm_landmark_indices,
                                                              landmark_points,
                                                              face_pointcloud,
                                                              preprocessor.bfm,
                                                              params_sparse.with_new_manager(dense_optimizer),
                                                              max_iterations=icp_iterations,
                                                              nearest_neighbor_mode=nn_mode,
                                                              distance_type=distance_type,
                                                              max_nfev=optimization_steps_per_iteration,
                                                              l2_regularization=l2_regularization_dense,
                                                              pointcloud_normals=pointcloud_normals,
                                                              weight_sparse_term=weight_sparse_term)
    return params, dense_param_history


if __name__ == "__main__":
    sparse_optimizer = BFMOptimization(preprocessor.bfm,
                                       n_params_shape=n_params_shape_sparse,
                                       n_params_expression=n_params_expression_sparse,
                                       weight_shape_params=weight_shape_params_sparse,
                                       weight_expression_params=weight_expression_params_sparse,
                                       rotation_mode='lie')

    dense_optimizer = BFMOptimization(preprocessor.bfm,
                                      n_params_shape=n_params_shape_dense,
                                      n_params_expression=n_params_expression_dense,
                                      weight_shape_params=weight_shape_params_dense,
                                      weight_expression_params=weight_expression_params_dense,
                                      rotation_mode='lie')

    plot_manager = PlotManager.new_run("video_reconstruction")

    initial_params = preprocessor.get_initial_params(sparse_optimizer)

    if keyframes:
        print(f"===== Estimating shape coefficients from keyframes =====")

        # Only optimize for shape coefficients
        sparse_optimizer = BFMOptimization(preprocessor.bfm,
                                           n_params_shape=n_params_shape_sparse,
                                           n_params_expression=0,
                                           weight_shape_params=weight_shape_params_sparse,
                                           weight_expression_params=weight_expression_params_sparse,
                                           rotation_mode='lie')

        dense_optimizer = BFMOptimization(preprocessor.bfm,
                                          n_params_shape=n_params_shape_dense,
                                          n_params_expression=0,
                                          weight_shape_params=weight_shape_params_dense,
                                          weight_expression_params=weight_expression_params_dense,
                                          rotation_mode='lie')

        img_landmark_points = []
        pointclouds = []
        pointcloud_normals_list = []
        bfm_landmark_indices_list = []
        for frame_id in keyframes:
            bfm_landmark_indices, landmark_points, face_pointcloud, pointcloud_normals = preprocess(frame_id)
            bfm_landmark_indices_list.append(bfm_landmark_indices)
            img_landmark_points.append(landmark_points)
            pointclouds.append(face_pointcloud)
            pointcloud_normals_list.append(pointcloud_normals)

        initial_params_keyframe = KeyframeOptimizationParameters(sparse_optimizer,
                                                                 [0 for _ in range(n_params_shape_sparse)],
                                                                 [initial_params.camera_pose for _ in
                                                                  range(len(keyframes))])
        sparse_keyframe_loss = sparse_optimizer.create_sparse_keyframe_loss(bfm_landmark_indices_list,
                                                                            img_landmark_points,
                                                                            regularization_strength=l2_regularization_sparse)
        sparse_context = sparse_optimizer.create_optimization_context(sparse_keyframe_loss, initial_params_keyframe)
        result = sparse_context.run_optimization()

        initial_params_keyframe_dense = \
            KeyframeOptimizationParameters.from_theta(sparse_context, result.x).with_new_manager(dense_optimizer)
        params_dense, distances, dense_param_history = run_icp_keyframes(dense_optimizer,
                                                                         pointclouds,
                                                                         preprocessor.bfm,
                                                                         initial_params_keyframe_dense,
                                                                         max_iterations=icp_iterations,
                                                                         nearest_neighbor_mode=nn_mode,
                                                                         distance_type=distance_type,
                                                                         max_nfev=optimization_steps_per_iteration,
                                                                         l2_regularization=l2_regularization_dense,
                                                                         pointcloud_normals=pointcloud_normals_list)

        # Don't optimize for shape coefficients anymore
        sparse_optimizer = BFMOptimization(preprocessor.bfm,
                                           n_params_shape=0,
                                           n_params_expression=n_params_expression_sparse,
                                           weight_shape_params=weight_shape_params_sparse,
                                           weight_expression_params=weight_expression_params_sparse,
                                           rotation_mode='lie')

        dense_optimizer = BFMOptimization(preprocessor.bfm,
                                          n_params_shape=0,
                                          n_params_expression=n_params_expression_dense,
                                          weight_shape_params=weight_shape_params_dense,
                                          weight_expression_params=weight_expression_params_dense,
                                          rotation_mode='lie')

        initial_params = BFMOptimizationParameters(sparse_optimizer, params_dense.shape_coefficients,
                                                   initial_params.expression_coefficients, initial_params.camera_pose)
        plot_manager.save_params(initial_params, "keyframe_params")

    for frame_id in preprocessor.loader.get_frame_ids():
        print(f"===== Frame {frame_id} ======")
        bfm_landmark_indices, landmark_points, face_pointcloud, pointcloud_normals = preprocess(frame_id)
        print(f"  --- Sparse Reconstruction ---")
        params_sparse = run_sparse_optimization(sparse_optimizer, bfm_landmark_indices, landmark_points, initial_params)
        print(f"  --- Dense Reconstruction ---")
        params_dense, param_history = run_dense_optimization(dense_optimizer, bfm_landmark_indices, landmark_points,
                                                             face_pointcloud, pointcloud_normals, params_sparse)

        img_with_mask = preprocessor.render_onto_img(params_dense)
        plot_manager.save_image(img_with_mask, f"frame_{frame_id:04d}.jpg")
        plot_manager.save_params(params_dense, f"params_{frame_id:04d}")
        plot_manager.save_param_history(param_history, f"param_history_{frame_id:04d}")

        if reuse_previous_frame:
            initial_params = params_dense.with_new_manager(sparse_optimizer)

        print()
        print()
