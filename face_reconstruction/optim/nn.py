from enum import Enum, auto
import numpy as np
from sklearn.neighbors import NearestNeighbors


class NearestNeighborMode(Enum):
    """
    Defines from which perspective nearest neighbors are found.
        - FACE_VERTICES: for every vertex of the face model, a nearest neighbor in the pointcloud will be assigned
        - POINTCLOUD: for every point in the pointcloud, a nearest neighbor in the face model will be assigned
    """
    FACE_VERTICES = auto()
    POINTCLOUD = auto()


def nearest_neighbors(pointcloud_src: np.ndarray, pointcloud_dst: np.ndarray):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in pointcloud_src.

    Parameters
    ----------
        pointcloud_src:
            (n, 3) list of source points
        pointcloud_dst
            (m, 3) list of destination points

    Returns
    -------
        (n, 3) list containing the distance to its nearest neighbor for every point in pointcloud_src
        (n,) list containing the index to its nearest neighbor in pointcloud_dst for every point in pointcloud_src
    """
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(pointcloud_dst)
    distances, indices = nn.kneighbors(pointcloud_src, return_distance=True)
    return distances.ravel(), indices.ravel()
