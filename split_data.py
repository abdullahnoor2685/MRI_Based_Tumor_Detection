from src.data.splitter import split_dataset

split_dataset(
    source_dir="data/raw",
    output_dir="data",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
)