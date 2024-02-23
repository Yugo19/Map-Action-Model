from torchvision.transforms import functional as F
from torchvision.transforms import transforms

def get_transform(train):
    transforms_list = []

    # Convert the PIL Image to a PyTorch Tensor
    transforms_list.append(transforms.ToTensor())

    if train:
        # Data augmentation for training
        transforms_list.extend([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # Add more augmentation transforms as needed
        ])

    return transforms.Compose(transforms_list)