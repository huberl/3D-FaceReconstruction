from distutils.core import setup

setup(
    name='3D-FaceReconstruction',
    version='0.1',
    packages=['face_reconstruction'],
    url='',
    license='',
    author='Tobias Kirschstein',
    author_email='kirschto@in.tum.de',
    description='',
    install_requires=['h5py', 'eos-py', 'pyrender', 'numpy', 'opencv-python', 'imutils', 'dlib', 'pyquaternion', 'tqdm',
                      'open3d']
)
