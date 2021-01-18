# 3D-FaceReconstruction

## 1. Setup

### 1.1. Basel Face Model 2019

Obtain the Basel Face Model from these links:
 - https://faces.dmi.unibas.ch/bfm/bfm2019/restricted/model2019_fullHead.h5 (full head model, the statistics for the back of the head based on little data and heuristics)
 - https://faces.dmi.unibas.ch/bfm/bfm2019/restricted/model2019_bfm.h5 (model masked similar to the original Basel Face Model mask)
 - https://faces.dmi.unibas.ch/bfm/bfm2019/restricted/model2019_face12.h5 (model cropped to the face region)

Put the model files into a data folder and update the paths in `env.py` accordingly.

### 1.2 Facial Keypoint detection

You can download the pretrained facial keypoint detector here:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

### 1.3 Environment variables

Change the `DATA_PATH` and `MODEL_PATH` environment variables in `env.py` to point to the locations where you stored 
the above files.
If you commit changes, ensure to temporarily stash `env.py` in order to not push your local paths to the repository.
