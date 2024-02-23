import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Annotated, Optional, Tuple, List, Dict
from zenml import step, pipeline
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)


@step(enable_cache=True, experiment_tracker="mlflow_tracker")
def test_step(model: nn.Module, test_dataloader: DataLoader, loss_fn: nn.Module, results: Dict, epochs: int) -> Tuple[
    Annotated[float, "test_loss"],
    Annotated[float, "test_acc"],
    Annotated[Dict, "results"]
]:
    """
    Perform testing step for a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        loss_fn (nn.Module): Loss function for the evaluation.
        results (Dict): Dictionary to store results.
        epochs (int): Number of epochs.

    Returns:
        Tuple[float, float, Dict]: Tuple containing test loss, test accuracy, and results.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    test_loss, test_acc = 0, 0

    for epoch in tqdm(range(epochs)):
        with torch.inference_mode():
            for batch, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                test_pred_logits = model(X)
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.cpu().item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

            test_loss = test_loss / len(test_dataloader)
            test_acc = test_acc / len(test_dataloader)
            
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_acc", test_acc)

    # Log the PyTorch model as an artifact
    mlflow.pytorch.log_model(model, "model")
    torch.save(model.state_dict(), 'save/TCM1.pth')

    return test_loss, test_acc, results
