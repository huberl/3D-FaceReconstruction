from env import BASEL_FACE_MODEL_PATH, MODELS_PATH
from face_reconstruction.utils.io import load_pickled, save_pickled
import dlib
import numpy as np


def load_bfm_landmarks(landmarks_file: str):
    return load_pickled(f"{BASEL_FACE_MODEL_PATH}/{landmarks_file}")


def save_bfm_landmarks(landmarks, landmarks_file: str):
    save_pickled(landmarks, f"{BASEL_FACE_MODEL_PATH}/{landmarks_file}")


def detect_landmarks(img):
    model_path = f"{MODELS_PATH}/Keypoint Detection/shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    rect = detector(img)
    shape = predictor(img, rect[0])

    coords = np.zeros((68, 2), dtype='int')
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords
