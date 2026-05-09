# MRI-Based Brain Tumor Detection

AI335L Deep Learning Lab

## Team
Abdullah Noor
Massam Abbas
Muhammad Hanzla

## Setup

```bash
git clone <your-repo-url>
cd MRI_Based_Tumor_Detection
pip install -r requirements.txt
```

## Reproduce Baseline

```bash
python run_baseline.py
```

Seed: **42** (set in Config and seed_everything())

## Dataset

Download via Kaggle (see `DATA_CARD.md` for source). Place under `data/raw/`.
Then run `notebooks/download_data.ipynb` to verify and split.

## Run Tests

```bash
pytest tests/
```

## Project Structure

```
MRI_Based_Tumor_Detection/
├── data/               # gitignored
├── notebooks/
├── src/
│   ├── data/           # dataset.py, splitter.py
│   ├── preprocessing/  # transforms.py, preprocessor.py
│   ├── models/         # baseline_mlp.py
│   ├── training/       # train.py, evaluate_final.py
│   └── utils/          # seed.py, logger.py, metrics.py, config.py
├── models/             # saved checkpoints (gitignored)
├── experiments/        # train.log lives here
├── tests/              # smoke tests
├── reports/
├── DATA_CARD.md
├── requirements.txt
└── README.md
```
