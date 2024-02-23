import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO



class MapActionDataset(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotations = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # Containers for multiple objects
        boxes_list = []
        labels_list = []
        img_ids_list = []
        areas_list = []
        iscrowd_list = []

        for annotation in coco_annotations:
            # Bounding boxes for objects
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, the input should be [xmin, ymin, xmax, ymax]
            xmin = annotation['bbox'][0]
            ymin = annotation['bbox'][1]
            xmax = xmin + annotation['bbox'][2]
            ymax = ymin + annotation['bbox'][3]
            boxes_list.append([xmin, ymin, xmax, ymax])

            # Labels based on category_id
            labels_list.append(annotation['category_id'])

            # Tensorise img_id
            img_ids_list.append(img_id)

            # Size of bbox (Rectangular)
            areas_list.append(annotation['area'])

            # Iscrowd
            iscrowd_list.append(0)  # Assuming all objects are not crowd, modify as needed

        # Convert lists to tensors
        boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
        labels = torch.as_tensor(labels_list, dtype=torch.int64)
        img_ids = torch.as_tensor(img_ids_list)
        areas = torch.as_tensor(areas_list, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd_list, dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_ids,
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

