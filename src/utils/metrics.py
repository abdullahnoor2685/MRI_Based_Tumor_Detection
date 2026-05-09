{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a585c487-fa2d-4e12-9fc1-d5ff06635cf8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from sklearn.metrics import f1_score, balanced_accuracy_score\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:\n",
    "    \"\"\"Compute simple accuracy from raw model outputs.\n",
    "\n",
    "    Args:\n",
    "        outputs: Raw logits of shape (N, num_classes).\n",
    "        labels: Ground-truth class indices of shape (N,).\n",
    "\n",
    "    Returns:\n",
    "        Accuracy as a float between 0 and 1.\n",
    "    \"\"\"\n",
    "    _, preds = torch.max(outputs, 1)\n",
    "    correct = (preds == labels).sum().item()\n",
    "    return correct / len(labels)\n",
    "\n",
    "\n",
    "def compute_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> dict:\n",
    "    \"\"\"Compute accuracy, F1 (macro), and balanced accuracy.\n",
    "\n",
    "    Args:\n",
    "        outputs: Raw logits of shape (N, num_classes).\n",
    "        labels: Ground-truth class indices of shape (N,).\n",
    "\n",
    "    Returns:\n",
    "        Dictionary with keys: accuracy, f1_macro, balanced_accuracy.\n",
    "    \"\"\"\n",
    "    _, preds = torch.max(outputs, 1)\n",
    "    preds_np = preds.cpu().numpy()\n",
    "    labels_np = labels.cpu().numpy()\n",
    "\n",
    "    return {\n",
    "        \"accuracy\": accuracy(outputs, labels),\n",
    "        \"f1_macro\": f1_score(labels_np, preds_np, average=\"macro\", zero_division=0),\n",
    "        \"balanced_accuracy\": balanced_accuracy_score(labels_np, preds_np),\n",
    "    }"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
