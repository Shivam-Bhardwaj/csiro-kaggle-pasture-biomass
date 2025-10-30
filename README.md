# CSIRO Kaggle Competition: Pasture Biomass Estimation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-brightgreen.svg)](https://www.kaggle.com/)

AI-powered pasture biomass estimation challenge hosted by CSIRO, Meat & Livestock Australia (MLA), and Google Australia.

## 🎯 Competition Overview

This Kaggle competition aims to improve the accuracy and efficiency of estimating pasture biomass - the amount of grass and other edible plants available for livestock to graze. This is a critical factor in grazing management that impacts productivity, environmental sustainability, and biodiversity.

**Competition Details:**
- **Prize Pool:** US$75,000
- **Host:** Kaggle
- **Partners:** CSIRO, Meat & Livestock Australia (MLA), Google Australia
- **Deadline:** January 28, 2026
- **Official Link:** [Explore the challenge](https://www.kaggle.com/competitions/csiro-pasture-biomass-estimation)

## 🌱 Problem Statement

Grazing systems cover around half of Australia's landmass and roughly a quarter of the Earth's land surface. Accurately measuring pasture biomass enables farmers to:

- Balance livestock needs with pasture regrowth
- Make better grazing decisions
- Maintain long-term land health
- Reduce the need for manual sampling
- Get faster and more reliable information

## 📊 Dataset

The competition dataset includes:
- **Pasture images** captured across diverse seasons, geographic locations, and pasture species compositions
- **Field measurements** paired with each image:
  - Plant height measurements
  - Vegetation indices (how green and healthy plants look)
  - Light reflectance data
  - Biomass measurements

### Task
Participants will develop AI models to:
1. Predict the amount of pasture available from images
2. Estimate the quantity of other plant species (like clover) with greater accuracy
3. Combine image data with plant health and vigour information for enhanced predictions

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Kaggle account and API credentials
- GPU recommended for deep learning models

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/csiro-kaggle-pasture-biomass.git
cd csiro-kaggle-pasture-biomass
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Kaggle API credentials:
```bash
# Place your kaggle.json credentials file in ~/.kaggle/
# Or use environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### Downloading the Dataset

```bash
# Download competition data using Kaggle API
kaggle competitions download -c csiro-pasture-biomass-estimation -p data/
unzip data/csiro-pasture-biomass-estimation.zip -d data/
```

## 📁 Project Structure

```
csiro-kaggle-pasture-biomass/
├── data/                   # Dataset files (not in git)
│   ├── raw/               # Raw competition data
│   ├── processed/         # Processed/preprocessed data
│   └── external/          # External datasets
├── models/                # Saved model files
│   ├── checkpoints/       # Training checkpoints
│   └── final/             # Final model weights
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/                   # Source code
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model architectures
│   ├── training/         # Training scripts
│   └── utils/            # Utility functions
├── scripts/              # Training and evaluation scripts
│   ├── train.py
│   ├── evaluate.py
│   └── submit.py
├── config/               # Configuration files
│   └── config.yaml
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🧪 Usage

### Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Training a Model
```bash
python scripts/train.py --config config/config.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --model models/final/best_model.pth
```

### Submission
```bash
python scripts/submit.py --predictions predictions/submission.csv
```

## 🔬 Approach

This project explores multiple approaches:

1. **Computer Vision Models**
   - CNN architectures (ResNet, EfficientNet, Vision Transformers)
   - Transfer learning from pretrained models
   - Multi-task learning for biomass and species prediction

2. **Feature Engineering**
   - Vegetation indices (NDVI, EVI, etc.)
   - Image augmentation techniques
   - Multi-scale feature extraction

3. **Ensemble Methods**
   - Combining multiple model predictions
   - Stacking and blending approaches

## 📈 Results

### 🏆 Advanced Multi-Task Model (Final)

**Model:** ResNet50 with Multi-Task Learning + Feature Fusion  
**Hardware:** NVIDIA H100 PCIe (85GB VRAM)  
**Training Setup:**
- Dataset: 1,785 real competition samples (357 unique images)
- Train/Val Split: 285 train images / 72 val images (image-based split)
- Batch Size: 16
- Learning Rate: 1e-4
- Optimizer: AdamW
- Scheduler: Cosine Annealing
- Image Size: 224x224
- Features: NDVI + Height fusion

**Best Results:**
- **Best Validation Loss:** 148.45 (Epoch 65)
- **Final Validation Loss:** 153.25
- **Training Completed:** 67 epochs (Early stopping)

**Per-Target Validation RMSE:**
- **Dry_Clover_g:** 13.64
- **Dry_Dead_g:** 15.92
- **Dry_Green_g:** 26.47
- **Dry_Total_g:** 45.41 (main target)
- **GDM_g:** 32.63

**Improvements:**
- ✅ **13% improvement** over baseline single-target model
- ✅ Multi-task learning captures relationships between targets
- ✅ Constraint enforcement: Dry_Total = sum of components
- ✅ Feature fusion with NDVI and Height

### 📊 Training Visualizations

Comprehensive training visualizations are available in the `visualizations/` directory:

- **[Loss Curves](visualizations/loss_curves.png)** - Training and validation loss over epochs
- **[RMSE by Target](visualizations/rmse_by_target.png)** - Per-target RMSE progression
- **[Final RMSE Comparison](visualizations/final_rmse_comparison.png)** - Final validation RMSE for each target type
- **[Training Summary](visualizations/training_summary.png)** - Overall training statistics and improvements

Generate visualizations anytime with:
```bash
python scripts/visualize_training.py
```

### 📤 Submission Files

Three submission files are ready:

1. **`submissions/submission.csv`** - Baseline single-target model
2. **`submissions/submission_multitarget.csv`** - Distribution-adjusted predictions
3. **`submissions/submission_multitask.csv`** - Advanced multi-task model ⭐ **BEST**

### 🔬 Key Findings

1. **Data Structure:** Each image has 5 targets (Dry_Clover_g, Dry_Dead_g, Dry_Green_g, Dry_Total_g, GDM_g)
2. **Constraint:** Dry_Total = Dry_Clover + Dry_Dead + Dry_Green (exact relationship)
3. **Feature Importance:** Height (0.33 correlation) > NDVI (0.25 correlation)
4. **Geographic Variation:** NSW pastures have highest biomass (39.70), Tas lowest (19.03)
5. **Species Impact:** Phalaris (59.23) > Fescue (46.68) > Clover (18.85)

### 🚀 Next Steps for Further Improvement

- [ ] Test-Time Augmentation (TTA)
- [ ] Higher resolution training (384x384 or 512x512)
- [ ] Ensemble of multiple models
- [ ] Different architectures (EfficientNet, Vision Transformers)
- [ ] Pseudo-labeling with confident predictions

## 🤝 Contributing

This is a competition project, but contributions and suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is created for the CSIRO Kaggle Competition. Please refer to the competition rules and terms for usage guidelines.

## 🙏 Acknowledgments

- **CSIRO** - Australia's National Science Agency
- **Meat & Livestock Australia (MLA)** - Industry partner
- **Google Australia** - Partnership support
- **FrontierSI** - Supporting organization
- **Kaggle** - Competition platform

## 📚 References

- [CSIRO Competition Announcement](https://www.csiro.au/en/news/All/News/2025/October/Kaggle-competition)
- [Kaggle Competition Page](https://www.kaggle.com/competitions/csiro-pasture-biomass-estimation)

## 📧 Contact

For questions about this repository, please open an issue.

---

**Good luck with the competition! 🚀**

