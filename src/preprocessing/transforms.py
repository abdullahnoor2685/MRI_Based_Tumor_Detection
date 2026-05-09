{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "076fbc94-c24f-44bb-95fc-c939431a6dc7",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'torchvision'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mModuleNotFoundError\u001b[39m                       Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 1\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m1\u001b[39m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mtorchvision\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m transforms\n\u001b[32m      4\u001b[39m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34mget_train_transforms\u001b[39m(image_size: \u001b[38;5;28mint\u001b[39m = \u001b[32m224\u001b[39m) -> transforms.Compose:\n\u001b[32m      5\u001b[39m \u001b[38;5;250m    \u001b[39m\u001b[33;03m\"\"\"Training transforms with two augmentation strategies:\u001b[39;00m\n\u001b[32m      6\u001b[39m \u001b[33;03m    1. Random horizontal flip (spatial augmentation)\u001b[39;00m\n\u001b[32m      7\u001b[39m \u001b[33;03m    2. Random rotation up to 15 degrees (orientation invariance)\u001b[39;00m\n\u001b[32m   (...)\u001b[39m\u001b[32m     13\u001b[39m \u001b[33;03m        Composed transform pipeline for training.\u001b[39;00m\n\u001b[32m     14\u001b[39m \u001b[33;03m    \"\"\"\u001b[39;00m\n",
      "\u001b[31mModuleNotFoundError\u001b[39m: No module named 'torchvision'"
     ]
    }
   ],
   "source": [
    "from torchvision import transforms\n",
    "\n",
    "\n",
    "def get_train_transforms(image_size: int = 224) -> transforms.Compose:\n",
    "    \"\"\"Training transforms with two augmentation strategies:\n",
    "    1. Random horizontal flip (spatial augmentation)\n",
    "    2. Random rotation up to 15 degrees (orientation invariance)\n",
    "\n",
    "    Args:\n",
    "        image_size: Target H x W for resizing.\n",
    "\n",
    "    Returns:\n",
    "        Composed transform pipeline for training.\n",
    "    \"\"\"\n",
    "    return transforms.Compose([\n",
    "        transforms.Resize((image_size, image_size)),\n",
    "        transforms.RandomHorizontalFlip(p=0.5),          # Augmentation 1\n",
    "        transforms.RandomRotation(degrees=15),            # Augmentation 2\n",
    "        transforms.ColorJitter(brightness=0.2, contrast=0.2),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize(mean=[0.485, 0.456, 0.406],\n",
    "                             std=[0.229, 0.224, 0.225]),  # ImageNet stats\n",
    "    ])\n",
    "\n",
    "\n",
    "def get_val_transforms(image_size: int = 224) -> transforms.Compose:\n",
    "    \"\"\"Validation/test transforms — NO augmentation, only resize + normalize.\n",
    "\n",
    "    Stats are fixed (ImageNet). If you compute custom stats from your train\n",
    "    set, replace mean/std here and in get_train_transforms consistently.\n",
    "\n",
    "    Args:\n",
    "        image_size: Target H x W for resizing.\n",
    "\n",
    "    Returns:\n",
    "        Composed transform pipeline for val/test.\n",
    "    \"\"\"\n",
    "    return transforms.Compose([\n",
    "        transforms.Resize((image_size, image_size)),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize(mean=[0.485, 0.456, 0.406],\n",
    "                             std=[0.229, 0.224, 0.225]),\n",
    "    ])"
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
