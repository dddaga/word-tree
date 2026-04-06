from setuptools import setup, find_packages

setup(
    name="mps-sparse",
    version="0.1.0",
    description="Efficient sparse matrix multiplication for PyTorch on Apple MPS",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)
