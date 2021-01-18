import eos
import h5py
import numpy as np
import trimesh

from env import BASEL_FACE_MODEL_PATH


class BaselFaceModel:

    def __init__(self, shape_model, expression_model, color_model=None):
        self.shape_model = shape_model
        self.expression_model = expression_model
        self.color_model = color_model
        self.face_model = eos.morphablemodel.MorphableModel(
            shape_model=shape_model,
            expression_model=expression_model,
            color_model=color_model)

    @staticmethod
    def from_h5(h5_file_name: str, use_color_model=True):
        h5 = h5py.File(f"{BASEL_FACE_MODEL_PATH}/{h5_file_name}")

        shape_model = BaselFaceModel.load_pca_model(h5, 'shape')
        expression_model = BaselFaceModel.load_pca_model(h5, 'expression')
        if use_color_model:
            color_model = BaselFaceModel.load_pca_model(h5, 'color')
            return BaselFaceModel(shape_model, expression_model, color_model)
        else:
            return BaselFaceModel(shape_model, expression_model)

    @staticmethod
    def load_pca_model(h5_model_file, model_type):
        return eos.morphablemodel.PcaModel(
            np.expand_dims(h5_model_file[f'{model_type}/model/mean'][()], 1),
            h5_model_file[f'{model_type}/model/pcaBasis'][()],
            np.expand_dims(h5_model_file[f'{model_type}/model/pcaVariance'][()], 1),
            h5_model_file[f'{model_type}/representer/cells'][()].T.tolist())

    def get_n_shape_coefficients(self):
        return self.face_model.get_shape_model().get_num_principal_components()

    def get_n_expression_coefficients(self):
        return self.face_model.get_expression_model().get_num_principal_components()

    def get_n_color_coefficients(self):
        return self.face_model.get_color_model().get_num_principal_components()

    def get_face_model(self):
        return self.face_model

    def draw_sample(self, shape_coefficients, expression_coefficients, color_coefficients=None):
        return self.face_model.draw_sample(shape_coefficients=shape_coefficients,
                                           expression_coefficients=expression_coefficients,
                                           color_coefficients=color_coefficients)

    @staticmethod
    def convert_to_trimesh(face_mesh):
        return trimesh.Trimesh(
            vertices=face_mesh.vertices,
            faces=face_mesh.tvi,
            vertex_colors=face_mesh.colors,
            face_colors=face_mesh.colors,
            process=False)
