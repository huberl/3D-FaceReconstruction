import numpy as np
from numpy import trace
from numpy.linalg import norm, det
from math import sin, cos, acos


def skew(w):
    '''
    Creates a skew symmetric matrix from a lie group so(3).
    :param w: The lie group so(3).
    :return:
    '''

    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def _rodriguez(w):
    '''
    Applies the Rodriguez formula to the so(3) group to yield the rotation SO(3).
    Rodriguez formula: e^(w_skew) = I + sin(w_hat) * w_hat / norm(w) + w_hat^2 / norm(w)^2 * (1 - cos(norm(w)))
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    :param w: The lie group so(3).
    :return: The rotation matrix from space SO(3).
    '''

    SO3 = np.eye(3) + sin(norm(w)) * skew(w) / norm(w) + (1 - cos(norm(w))) * skew(w) @ skew(w) / norm(w)**2
    assert abs(det(SO3) - 1.0) < 0.00001, f'Determinant of rotation matrix should be +1, but is {det(SO3)} instead!'
    return SO3


def se3_to_SE3(w, v):
    '''
    Transforms the lie algebra se(3) to the special Euclidian group SE(3)
    :param w: Vector of dim (3,) describing the rotation.
    :param v: Vector of dim (3,) describing the translation.
    :return: The rotation matrix from SE(3).
    '''
    so3 = _rodriguez(w)
    trans = ((np.eye(3) - _rodriguez(w)) @ skew(w) @ v + np.outer(w, w) @ v) / norm(w)**2

    # Sanity check: This should give the same translation
    #J = np.eye(3) + (1 - cos(norm(w))) * skew(w) / norm(w)**2 + (norm(w) - sin(norm(w))) * skew(w) @ skew(w) / norm(w)**3
    #trans = J @ v

    rot_mat = np.eye(4)
    rot_mat[:3, :3] = so3
    rot_mat[:3, 3] = trans

    return rot_mat


def SE3_to_se3(T):
    '''
    Computes the lie algebra se(3) from a transformation matrix SE(3)
    :param R: The rotation matrix from SE(3)
    :return: The vectors w, v of the lie algebra
    '''
    R = T[:3, :3]
    t = T[:3, 3]
    w_norm = acos((trace(R) - 1) / 2)

    w = w_norm / (2 * sin(w_norm)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    J_inv = np.eye(3) - 0.5 * skew(w) + (1 / w_norm**2 - (1 + cos(w_norm)) / 2 * w_norm * sin(w_norm)) * skew(w) @ skew(w)
    v = J_inv @ t

    return w, v


def rot2eul(mat):
    '''
    Computes the euler angles from a rotation matrix or transformation matrix
    :param R: The rotation matrix
    :return: Angles alpha, beta, gamma
    '''
    assert mat.shape == (3, 3) or mat.shape == (4, 4)

    # Transform Transformation matrix to rotation matrix if needed
    R = mat[:3, :3]

    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1]/np.cos(beta), R[2, 2]/np.cos(beta))
    gamma = np.arctan2(R[1, 0]/np.cos(beta), R[0, 0]/np.cos(beta))
    return np.array([alpha, beta, gamma])


if __name__ == '__main__':
    w = np.array([1.7, 2.0, 1.0])
    v = np.array([1.0, 1.0, 1.0])

    T = se3_to_SE3(w, v)
    w_new, v_new = SE3_to_se3(T)

    print(f'Error of inverse w: {abs(w_new - w)}')
    print(f'Error of inverse v: {abs(v_new - v)}')

    print(f'Euler angles original: {rot2eul(se3_to_SE3(w, v))}')
    print(f'Euler angles inverse: {rot2eul(T)}')

