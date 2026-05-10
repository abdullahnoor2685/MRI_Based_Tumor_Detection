import torch
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.dataset import MRIDataset
from src.preprocessing.transforms import get_train_transforms, get_val_transforms
from src.models.baseline_mlp import BaselineMLP
from src.training.train import train_model
from src.training.evaluate_final import final_test_evaluation

logger = get_logger()
cfg = Config()
seed_everything(cfg.SEED)


def main():
    logger.info("=== Phase 2 Baseline Training ===")

    train_dataset = MRIDataset(
        root_dir=f"{cfg.DATA_DIR}train",
        transform=get_train_transforms(cfg.IMAGE_SIZE),
    )
    val_dataset = MRIDataset(
        root_dir=f"{cfg.DATA_DIR}val",
        transform=get_val_transforms(cfg.IMAGE_SIZE),
        class_to_idx=train_dataset.class_to_idx,
    )
    test_dataset = MRIDataset(
        root_dir=f"{cfg.DATA_DIR}test",
        transform=get_val_transforms(cfg.IMAGE_SIZE),
        class_to_idx=train_dataset.class_to_idx,
    )

    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    logger.info(f"Class distribution (train): {train_dataset.get_class_counts()}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = BaselineMLP(
        input_dim=3 * cfg.IMAGE_SIZE * cfg.IMAGE_SIZE,
        hidden_dims=cfg.MLP_HIDDEN_DIMS,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.MLP_DROPOUT,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_model(model, train_loader, val_loader, cfg)

    class_names = list(train_dataset.class_to_idx.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_test_evaluation(model, test_loader, class_names, device)


if __name__ == "__main__":
    main()