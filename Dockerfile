# Use the official PyTorch image as the base image
FROM pytorch/pytorch:latest

# Update and install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    libopenblas-dev \
    libblas-dev \
    libatlas-base-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages
RUN pip install \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter

# Install PyTorch with GPU support
RUN pip install torch

# Set the working directory
WORKDIR /workspace

# Specify CUDA compatibility (adjust based on your CUDA version)
ENV TORCH_CUDA_ARCH_LIST "7.5"

