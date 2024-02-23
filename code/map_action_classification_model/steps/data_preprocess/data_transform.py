import torch
from torchvision.transforms import v2 as T
from typing import Annotated, Optional, Tuple
from zenml import step, pipeline

@step
def get_transform(train):
    """
    Get image transformations based on whether it's for training or not.

    Args:
        train (bool): True if transformations are for training, False otherwise.

    Returns:
        torchvision.transforms.Compose: Composition of image transformations.
    """
    # Initialize an empty list to store transformations
    transforms = []

    # Apply transformations for training
    if train:
        # Randomly flip the image horizontally with a probability of 50%
        transforms.append(T.RandomHorizontalFlip(0.5))
        # Randomly resize and crop the image to the specified size with antialiasing
        transforms.append(T.RandomResizedCrop(size=[224, 224], antialias=True))
        # Normalize the image with mean=[0.] (Please replace with actual mean values)
        # transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # Convert the image to torch float and scale the pixel values
    transforms.append(T.ToDtype(torch.float32, scale=True))

    # Combine all transformations into a single composition
    return T.Compose(transforms)

