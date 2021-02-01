import numpy as np
from numpy.linalg import norm, det
from math import sin, cos


def skew(w):
    '''
    Creates a skew symmetric matrix from a lie group so(3).
    :param w: The lie group so(3).
    :return:
    '''
    assert w.shape == (3,), 'w has to have shape of (3,)'

    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def _rodriguez(w):
    '''
    Applies the Rodriguez formula to the so(3) group to yield the rotation SO(3).
    Rodriguez formula: e^(w_skew) = I + sin(w_hat) * w_hat / norm(w) + w_hat^2 / norm(w)^2 * (1 - cos(norm(w)))
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    :param w: The lie group so(3).
    :return: The rotation matrix from space SO(3).
    '''

    SO3 =  np.eye(3) + sin(norm(w)) * skew(w) / norm(w) + (1 - cos(norm(w))) * skew(w) @ skew(w) / norm(w)**2
    assert abs(det(SO3) - 1.0) < 0.00001, f'Determinant of rotation matrix should be +1, but is {det(SO3)} instead!'
    return SO3


def se3_to_SE3(w, v):
    so3 = _rodriguez(w)
    trans = (np.eye(3) - _rodriguez(w)) @ skew(w) @ v + np.outer(w, w) @ v

    rot_mat = np.eye(4)
    rot_mat[:3, :3] = so3
    rot_mat[:3, 3] = trans
    return rot_mat


if __name__ == '__main__':
    w = np.array([1, 2, 3])
    v = np.array([1, 2, 4])

    rot = se3_to_SE3(w, v)
    print(rot)
    print(f'Determinant of Rotation Matrix: {det(rot)}')
