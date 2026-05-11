import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from src.utils.logger import get_logger
from src.utils.metrics import compute_metrics
from src.utils.seed import seed_everything
from src.utils.config import Config

logger = get_logger()


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    all_outputs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_outputs.append(outputs.detach())
        all_labels.append(labels.detach())

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_outputs, all_labels)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_outputs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        all_outputs.append(outputs)
        all_labels.append(labels)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_outputs, all_labels)
    return avg_loss, metrics


def train_model(model, train_loader, val_loader, cfg=None):
    if cfg is None:
        cfg = Config()

    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    best_checkpoint = os.path.join(cfg.CHECKPOINT_DIR, "best_mlp.pth")

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg.GRAD_CLIP)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint)
            logger.info(f"  Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement ({patience_counter}/{cfg.EARLY_STOPPING_PATIENCE})")

        if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    logger.info("Training complete. Best model loaded.")
    return model
if __name__ == "__main__":
    # Load Config without arguments if it doesn't support them
    cfg = Config() 
    
    # Manually ensure the right config file is used if your class has a load method
    cfg.load('configs/unreg.yaml') 

    from src.data.dataset import get_dataloaders
    from src.models.cnn_backbone import tumorCNN

    # Get Data
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # Initialize Model
    model = tumorCNN(num_classes=4)

    # Start Training
    logger.info("Starting Unregularized Baseline Run...")
    train_model(model, train_loader, val_loader, cfg=cfg)