# CSIRO Pasture Biomass Estimation - Kaggle Submission Notebook

This notebook generates predictions using the advanced multi-task ResNet50 model.

**Model:** ResNet50 with multi-task learning + NDVI/Height fusion  
**Best Validation Loss:** 148.45  
**Training:** 67 epochs on competition dataset

## Setup Instructions

1. **Upload Model Checkpoint as Dataset:**
   - Create a new Kaggle dataset named `csiro-biomass-model`
   - Upload `models/checkpoints/best_multitask_model.pth`
   - Make dataset public

2. **Add Dataset to Notebook:**
   - Go to Notebook settings
   - Add dataset: `csiro-biomass-model`
   - Add dataset: `csiro-biomass` (competition data)

3. **Configure Notebook:**
   - Internet: OFF
   - GPU: ON (recommended) or CPU
   - Accelerator: T4 GPU (free) or P100 (free)

4. **Run and Submit:**
   - Click "Run All"
   - Once completed, click "Save Version" → "Save & Run All"
   - Click "Submit" button

---

## Code Cell

```python
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

# [Full code continues - see scripts/predict_multitask.py for complete implementation]
```

Note: For the complete notebook, copy the code from `scripts/predict_multitask.py` and adapt it for Kaggle's input paths.

