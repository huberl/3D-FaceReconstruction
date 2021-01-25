from env import BASEL_FACE_MODEL_PATH, MODELS_PATH
from face_reconstruction.utils.io import load_pickled, save_pickled
import dlib
import numpy as np
from scipy import spatial


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

def detect_pcd_landmarks(img, pcd,K):
    #intrinsic matrix from camera_info.yaml
    #K= [1052.667867276341, 0, 962.4130834944134, 0, 1052.020917785721, 536.2206151001486, 0, 0, 1]
    #pcd = open3d.io.read_point_cloud(pcd_path)
    pcd_points=np.array(pcd.points)[:,:2]
    #for faster searching the neighest neighboring point
    pcd_spatial = spatial.KDTree(pcd_points)
    #img = cv2.imread(img_path)
    landmarks_img = detect_landmarks(img)
    def screenToCamera(x,y):
        p_x = (x - K[2]) / K[0]
        p_y = (y - K[5]) / K[4]
        return p_x,p_y
    idx_landmarks_pcd=[]
    for l_point in landmarks_img:
        landmark_pcd = screenToCamera(l_point[0],l_point[1])
        index=pcd_spatial.query(landmark_pcd)[1]
        idx_landmarks_pcd.append(index)
        #annotate landmark in pointcloud
        #pcd.colors[index]=[255,0,0]
    return idx_landmarks_pcd
    


    

