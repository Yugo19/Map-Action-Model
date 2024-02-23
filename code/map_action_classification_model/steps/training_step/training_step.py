import torch
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import mlflow.pytorch
from typing import Annotated, Optional, Tuple, List, Dict
from zenml import step, pipeline
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model(model: torch.nn.Module, train_dataloader: DataLoader, epochs: int, loss_fn: torch.nn.Module) -> Tuple[
    Annotated[torch.nn.Module, "model"],
    Annotated[Dict, "results"]
]:
    """
    Train a PyTorch model using the provided training dataloader.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_dataloader (DataLoader): The training dataloader.
        epochs (int): Number of training epochs.
        loss_fn (torch.nn.Module): The loss function for training.

    Returns:
        Tuple[torch.nn.Module, Dict]: Tuple containing the trained model and a dictionary with training results.
            The results dictionary contains 'train_loss' and 'train_acc'.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss":[],
        "test_acc": [],
    }

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    model.to(device)
    mlflow.end_run()
    with mlflow.start_run():
        for epoch in tqdm(range(epochs)):
            model.train()
            train_loss, train_acc = 0, 0

            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                train_loss += loss.cpu().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                train_acc += (y_pred_class == y).sum().item() / len(y_pred)

            train_loss = train_loss / len(train_dataloader)
            train_acc = train_acc / len(train_dataloader)

            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} "
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("train_acc", train_acc)
        
        mlflow.pytorch.log_model(model, "model")
    

    return model, results
