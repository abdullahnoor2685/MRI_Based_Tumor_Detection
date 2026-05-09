{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "09c6bf23-61d4-4b93-bf11-97de4865ee43",
   "metadata": {},
   "outputs": [],
   "source": [
    "import logging\n",
    "import os\n",
    "\n",
    "\n",
    "def get_logger(log_dir: str = \"experiments\", log_file: str = \"train.log\") -> logging.Logger:\n",
    "    \"\"\"Create and return a logger that writes to both console and file.\n",
    "\n",
    "    Args:\n",
    "        log_dir: Directory where the log file will be saved.\n",
    "        log_file: Name of the log file.\n",
    "\n",
    "    Returns:\n",
    "        Configured Logger instance.\n",
    "    \"\"\"\n",
    "    os.makedirs(log_dir, exist_ok=True)\n",
    "\n",
    "    logger = logging.getLogger(\"MRI_Project\")\n",
    "    logger.setLevel(logging.INFO)\n",
    "\n",
    "    if not logger.handlers:\n",
    "        # File handler\n",
    "        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))\n",
    "        file_handler.setLevel(logging.INFO)\n",
    "\n",
    "        # Console handler\n",
    "        console_handler = logging.StreamHandler()\n",
    "        console_handler.setLevel(logging.INFO)\n",
    "\n",
    "        formatter = logging.Formatter(\"%(asctime)s - %(levelname)s - %(message)s\")\n",
    "        file_handler.setFormatter(formatter)\n",
    "        console_handler.setFormatter(formatter)\n",
    "\n",
    "        logger.addHandler(file_handler)\n",
    "        logger.addHandler(console_handler)\n",
    "\n",
    "    return logger"
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
