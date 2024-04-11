from setuptools import setup, find_packages

setup(
    name='GenerativeRL',
    version='0.0.1',
    description='PyTorch implementations of generative reinforcement learning algorithms',
    author='zjowowen',

    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        'gym',
        'numpy',
        'torch>=2.2.0',
        'opencv-python',
        'tensordict',
        'matplotlib',
        'wandb',
        'rich',
        'mujoco_py',
        'easydict',
        'Cython<3.0',
        'tqdm',
        'torchdyn',
        'torchode',
        'torchsde',
        'scipy',
        'diffusers',
        'timm',
        'av',
        'moviepy',
        'imageio[ffmpeg]',
    ]
)
