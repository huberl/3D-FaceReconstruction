# 3D-FaceReconstruction

## 1. Setup

### 1.1. Basel Face Model 2019

Obtain the Basel Face Model from these links:
 - https://faces.dmi.unibas.ch/bfm/bfm2019/restricted/model2019_fullHead.h5 (full head model, the statistics for the back of the head based on little data and heuristics)
 - https://faces.dmi.unibas.ch/bfm/bfm2019/restricted/model2019_bfm.h5 (model masked similar to the original Basel Face Model mask)
 - https://faces.dmi.unibas.ch/bfm/bfm2019/restricted/model2019_face12.h5 (model cropped to the face region)

Put the model files into a data folder and update the paths in `env.py` accordingly.