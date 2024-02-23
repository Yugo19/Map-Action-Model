import os
import torch

from torchvision.io import read_image, ImageReadMode
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class MapActionDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images2"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Mask"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images2", self.imgs[idx])
        mask_path = os.path.join(self.root, "Mask", self.masks[idx])
        img = read_image(img_path,  mode=ImageReadMode.RGB)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)
        
        boxes = []
        for mask in masks:
            pos = torch.where(mask)
            if pos[0].numel() == 0 or pos[1].numel() == 0:
                # Skip masks with no positive pixels
                continue
            xmin = torch.min(pos[1]).item()
            xmax = torch.max(pos[1]).item()
            ymin = torch.min(pos[0]).item()
            ymax = torch.max(pos[0]).item()
            # Ensure positive height and width
            if xmin == xmax or ymin == ymax:
                continue
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes)

        assert len(boxes) > 0, "No valid bounding boxes found."

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="xyxy", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
