"""
Kaggle Submission Notebook Code for CSIRO Pasture Biomass Estimation

This code will be copied into a Kaggle notebook cell.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import os

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model architecture (copy from scripts/predict_multitask.py)
# ... [Model code will be inserted here]

print("✅ Code template created")
