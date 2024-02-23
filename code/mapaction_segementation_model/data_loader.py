import torch
import os
from utility import MapActionDataset
from data_transform import get_transform
from torch.utils.data import DataLoader, Subset


NUM_WORKERS = os.cpu_count()

root = "/home/mapaction/Documents/Exp-data/"

mapaction_dataset = MapActionDataset(root, get_transform(train=True))
mapaction_test_dataset = MapActionDataset(root, get_transform(train=False))


indices = torch.randperm(len(mapaction_dataset)).tolist()
dataset = Subset(mapaction_dataset, indices[:7])
test_dataset = Subset(mapaction_dataset, indices[7:])


def collate_fn(batch):
    return tuple(zip(*batch))



mapction_data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)

mapaction_test_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)