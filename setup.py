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
        'di-treetensor',
        'matplotlib',
        'wandb',
        'rich',
        'easydict',
        'tqdm',
        'torchdyn',
        'torchode',
        'torchsde',
        'scipy',
        'beartype',
        'diffusers',
        'timm',
        'av',
        'moviepy',
        'imageio[ffmpeg]',
    ],
    extras_require={
        'd4rl': [
            'gym==0.23.1',
            'mujoco_py',
            'Cython<3.0',
        ],
        'DI-engine': [
            'DI-engine',
        ]
    }
)
