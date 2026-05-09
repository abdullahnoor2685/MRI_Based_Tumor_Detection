import torch
from sklearn.metrics import f1_score, balanced_accuracy_score


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def compute_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> dict:
    _, preds = torch.max(outputs, 1)
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return {
        "accuracy": accuracy(outputs, labels),
        "f1_macro": f1_score(labels_np, preds_np, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels_np, preds_np),
    }