import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from src.utils.logger import get_logger
from src.training.train import evaluate

logger = get_logger()


def final_test_evaluation(model, test_loader, class_names, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    logger.info("=" * 60)
    logger.info("FINAL TEST SET RESULTS")
    logger.info(f"Test Loss     : {test_loss:.4f}")
    logger.info(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Macro : {test_metrics['f1_macro']:.4f}")
    logger.info(f"Balanced Acc  : {test_metrics['balanced_accuracy']:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info("=" * 60)

    return {
        "loss": test_loss,
        "metrics": test_metrics,
        "confusion_matrix": cm,
        "report": report,
    }