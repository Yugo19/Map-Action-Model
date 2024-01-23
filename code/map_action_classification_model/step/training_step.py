import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict
import mlflow.pytorch
from zenml.steps import step, Output, BaseStepConfig
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

@enable_mlflow
@step(enable_cache=False)
def train_model(model: nn.Module, train_dataloader, optimizer, loss_fn, epochs=20) -> Output(
    model = nn.Module,
    results = List
    
):
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): The neural network model.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        optimizer (Optimizer): The optimizer for model training.
        loss_fn (nn.Module): The loss function.
        epochs (int, optional): Number of training epochs. Defaults to 20.

    Returns:
        Tuple[nn.Module, dict]: Trained model and dictionary containing training results.
    """
    model.cuda()

    results = {
        "train_loss": [],
        "train_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.cuda(), y.cuda()
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
        
    
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(model, "model")
            
    return model, results
