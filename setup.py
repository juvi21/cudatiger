from setuptools import setup, find_packages

setup(
    name="cudatiger",
    version="0.0.1", 
    description="About An accelerated implementation of the Tiger optimizer for PyTorch, supercharged with Triton for enhanced CUDA GPU efficiency",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="juvi21",
    author_email="juv121@skiff.com",
    url="https://github.com/juvi21/tritonizedTiger",
    install_requires=[
        "torch",
        "triton"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires='>=3.10.12',
)
