from setuptools import find_packages, setup


setup(
    name="nanoowl",
    version="1.0.0",
    description="NanoOWL is a project that optimizes OWL-ViT to run in real-time on NVIDIA Jetson Orin Platforms with NVIDIA TensorRT.",
    author="Originally authored by NVIDIA, modifications made by Independent Robotics.",
    maintainer="Michael Fulton",
    maintainer_email="michael.fulton@independentrobotics.com",
    license="Apache 2.0",
    packages=find_packages()
)