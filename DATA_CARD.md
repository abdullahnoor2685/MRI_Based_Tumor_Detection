# Data Card — MRI Brain Tumor Dataset

| Field            | Details                                                                 |
|------------------|-------------------------------------------------------------------------|
| Source URL       | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| License          | CC0: Public Domain                                                      |
| Modality         | MRI (T1-weighted, contrast-enhanced)                                    |
| Total Samples    | ~7,023 images                                                           |
| Classes          | glioma, meningioma, pituitary, notumor (4 classes)                      |
| Image Format     | JPEG / PNG, variable resolution                                         |
| Known Issues     | Class imbalance present; some duplicates reported in community          |
| Split Used       | 70% train / 15% val / 15% test (stratified by class)                   |
| Checksum Method  | File count per class verified after download (see download_data.ipynb)  |
