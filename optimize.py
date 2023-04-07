import g2o
import numpy as np

def bundle_adjustment(K, poses, points_3D, observations):
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    for i, (R, t) in enumerate(poses):
        pose = g2o.Isometry3d(R, t)
        se3quat = g2o.SE3Quat(pose.rotation(), pose.translation())

        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(i)
        vertex.set_estimate(se3quat)

        if i == 0:
            vertex.set_fixed(True)  # Fix the first pose to ensure gauge freedom

        optimizer.add_vertex(vertex)
    # Add point vertices
    for i, point in enumerate(points_3D.T):
        vertex = g2o.VertexPointXYZ()
        vertex.set_id(len(poses) + i)
        vertex.set_estimate(point)
        vertex.set_marginalized(True)
        optimizer.add_vertex(vertex)

    # Add observation edges
    for pose_id, point_id, xy in observations:
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, optimizer.vertex(point_id + len(poses)))
        edge.set_vertex(1, optimizer.vertex(pose_id))
        edge.set_measurement(xy)
        edge.set_information(np.eye(2))
        edge.set_robust_kernel(g2o.RobustKernelHuber())

        # Set camera intrinsics
        camera = g2o.CameraParameters(K[0, 0], np.array([K[0, 2], K[1, 2]]), 0)
        camera.set_id(0)
        optimizer.add_parameter(camera)
        edge.set_parameter_id(0, camera.id())

        optimizer.add_edge(edge)

    # Optimize the graph
    optimizer.initialize_optimization()
    optimizer.optimize(1)

    # Update poses and points
    optimized_poses = []
    for i in range(len(poses)):
        vertex = optimizer.vertex(i)
        pose = vertex.estimate().matrix()[:3]
        optimized_poses.append((pose[:, :3], pose[:, 3]))

    optimized_points = []
    for i in range(points_3D.shape[1]):
        vertex = optimizer.vertex(len(poses) + i)
        point = vertex.estimate()
        optimized_points.append(point)

    return optimized_poses, np.array(optimized_points).T
