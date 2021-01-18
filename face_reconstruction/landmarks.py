from env import BASEL_FACE_MODEL_PATH
from face_reconstruction.utils.io import load_pickled, save_pickled


def load_landmarks(landmarks_file: str):
    return load_pickled(f"{BASEL_FACE_MODEL_PATH}/{landmarks_file}")


def save_landmarks(landmarks, landmarks_file: str):
    save_pickled(landmarks, f"{BASEL_FACE_MODEL_PATH}/{landmarks_file}")
