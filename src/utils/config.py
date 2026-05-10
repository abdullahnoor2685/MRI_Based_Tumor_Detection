class Config:
    DATA_DIR: str = "data/"
    IMAGE_SIZE: int = 224
    NUM_CLASSES: int = 4
    NUM_WORKERS: int = 4
    BATCH_SIZE: int = 64
    EPOCHS: int = 10
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EARLY_STOPPING_PATIENCE: int = 5
    GRAD_CLIP: float = 1.0
    SEED: int = 42
    CHECKPOINT_DIR: str = "models/"
    LOG_DIR: str = "experiments/"
    REPORT_DIR: str = "reports/"
    MLP_HIDDEN_DIMS: list = [512, 256]
    MLP_DROPOUT: float = 0.4