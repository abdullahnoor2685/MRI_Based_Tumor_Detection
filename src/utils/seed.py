{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d7ee18dc-d87b-4d15-aa98-0e79019bbbd6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random\n",
    "import numpy as np\n",
    "import torch\n",
    "\n",
    "\n",
    "def seed_everything(seed: int = 42) -> None:\n",
    "    \"\"\"Set seeds for Python, NumPy, and PyTorch for full reproducibility.\n",
    "\n",
    "    Args:\n",
    "        seed: Integer seed value. Document this in every report.\n",
    "    \"\"\"\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    torch.cuda.manual_seed_all(seed)\n",
    "    torch.backends.cudnn.deterministic = True\n",
    "    torch.backends.cudnn.benchmark = False"
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
