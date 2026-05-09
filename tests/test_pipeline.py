import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline_mlp import BaselineMLP
from src.utils.metrics import compute_metrics, accuracy
from src.utils.config import Config


def make_dummy_loader(n: int = 16, num_classes: int = 4) -> DataLoader:
    images = torch.randn(n, 3, 224, 224)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=8)


def test_mlp_output_shape() -> None:
    cfg = Config()
    model = BaselineMLP(
        input_dim=3 * 224 * 224,
        hidden_dims=cfg.MLP_HIDDEN_DIMS,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.MLP_DROPOUT,
    )
    model.eval()
    dummy = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (8, cfg.NUM_CLASSES)


def test_accuracy_range() -> None:
    outputs = torch.randn(32, 4)
    labels = torch.randint(0, 4, (32,))
    acc = accuracy(outputs, labels)
    assert 0.0 <= acc <= 1.0


def test_compute_metrics_keys() -> None:
    outputs = torch.randn(32, 4)
    labels = torch.randint(0, 4, (32,))
    metrics = compute_metrics(outputs, labels)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "balanced_accuracy" in metrics


def test_pipeline_runs() -> None:
    loader = make_dummy_loader()
    model = BaselineMLP()
    model.eval()
    for images, labels in loader:
        with torch.no_grad():
            out = model(images)
        assert out.shape[1] == 4
        break