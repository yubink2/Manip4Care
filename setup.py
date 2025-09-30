from setuptools import setup, find_packages

setup(
    name="manip4care",
    version="0.1.0",
    packages=find_packages(
        include=["manip4care", "manip4care.*"]
    ),
    python_requires=">=3.8,<3.9",
    install_requires=[
        "numpy==1.23.5",
        "trimesh",
        "autolab-core",
        "h5py",
        "pyrender",
        "pyglet==1.5.27",
        "python-fcl",
        "open3d",
        "pybullet",
        "scipy",
        "pytorch-kinematics",
        "ipywidgets",
        "wheel",
    ],
)
