import os
import time
import torch
from tqdm import tqdm
from typing import Tuple, List, Dict
from torch import nn, optim
from torch.utils.data import DataLoader

import sys
sys.path.append('/home/mapaction/mapaction_env/Map-Action-Model/code/vision_dir/')
from engine import evaluate, _get_iou_types





class ModelTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                 optimizer: optim.Optimizer, loss_fn: nn.Module, device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_step(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.train()
        train_loss = 0.0

        for X, y in tqdm(dataloader):
            X = list(image.to(self.device) for image in X)
            y = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in y]
            loss_dict = self.model(X, y)
            losses = sum(loss for loss in loss_dict.values())
            self.optimizer.zero_grad()
            
            losses.backward()
            self.optimizer.step()
            train_loss = losses

        return train_loss

    def test_step(self, dataloader: DataLoader) -> torch.Tensor:
        
        return evaluate(self.model, dataloader, device=self.device)

    def train(self, epochs: int) -> Dict[str, List[float]]:
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []}

        self.model.to(self.device)

        for epoch in tqdm(range(epochs)):
            train_loss = self.train_step(self.train_loader)
            test_loss = self.test_step(self.test_loader)

            print(f"Epoch: {epoch + 1} | "
                  f"train_loss: {train_loss:.4f} | "
                   #f"test_loss: {test_loss} | "
                  )

            results["train_loss"].append(train_loss)
            # results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            # results["test_acc"].append(test_acc)

        return results
