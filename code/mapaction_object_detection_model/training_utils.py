import os
import sys
import time
import torch
import mlflow
from tqdm import tqdm
from typing import Tuple, List, Dict
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
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
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        
        
    def extract_predictions(self, model_output):
        predictions = []
        for prediction in model_output:
            pred_dict = {
                'boxes': prediction['boxes'].detach().cpu().numpy(),
                'labels': prediction['labels'].detach().cpu().numpy(),
                'scores': prediction['scores'].detach().cpu().numpy(),
            }
            predictions.append(pred_dict)
        return predictions

    def train_step(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.train()
        train_loss = 0.0

        for X, y in tqdm(dataloader):
            X = list(image.to(self.device) for image in X)
            y = [{k: v.to(self.device) for k, v in t.items()} for t in y]
            loss_dict = self.model(X, y)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            train_loss = losses

        return train_loss

    def test_step(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(dataloader):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self.model(images)

                # Assuming you have a function to extract bounding boxes, labels, and scores from the model output
                predictions = self.extract_predictions(outputs)

                all_predictions.extend(predictions)
                all_targets.extend(targets)

        # Calculate metrics
   

        # Extract ground truth labels and predicted labels
        true_labels = [t['labels'].cpu().numpy() for t in all_targets]
        predicted_labels = [p['labels'] for p in all_predictions]


        # Flatten the lists
        true_labels = [item for sublist in true_labels for item in sublist]
        predicted_labels = [item for sublist in predicted_labels for item in sublist]


        # Ensure the lengths match
        min_length = min(len(true_labels), len(predicted_labels))
        true_labels = true_labels[:min_length]
        predicted_labels = predicted_labels[:min_length]

        # Calculate precision, recall, and f1 score
        self.precision = precision_score(true_labels, predicted_labels, average='weighted')
        self.recall = recall_score(true_labels, predicted_labels, average='weighted')
        self.f1 = f1_score(true_labels, predicted_labels, average='weighted')
        
        return self.precision, self.recall, self.f1
    mlflow.sklearn.autolog()        
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
                    f"test_loss: {test_loss} | "
                    )

            results["train_loss"].append(train_loss)
                # results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
                # results["test_acc"].append(test_acc)
            mlflow.log_params({
                    "Optimizer":self.optimizer,
                    "Criterion": self.loss_fn,
                    "Epochs": epochs
                    
                })
                
            mlflow.log_metrics({
                    "Precision": self.precision,
                    "recall": self.recall,
                    "f1": self.f1
                })

        return results