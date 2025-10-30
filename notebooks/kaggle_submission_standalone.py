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

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

"""
Advanced multi-task model for CSIRO competition.
Predicts all 5 target types simultaneously with separate heads.
"""

import torch
import torch.nn as nn
import torchvision.models as models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class AdvancedMultiTaskBiomassModel(nn.Module):
    """
    Advanced multi-task model that predicts all 5 target types simultaneously.
    Uses shared backbone with separate regression heads for each target type.
    """
    
    def __init__(self, model_name='resnet50', pretrained=True, dropout=0.5):
        """
        Initialize advanced multi-task model.
        
        Args:
            model_name: Base model name
            pretrained: Use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        self.target_types = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        
        # Backbone
        if TIMM_AVAILABLE:
            try:
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool=''
                )
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    features = self.backbone(dummy_input)
                    if isinstance(features, tuple):
                        features = features[0]
                    if len(features.shape) == 4:
                        gap = nn.AdaptiveAvgPool2d(1)
                        features = gap(features)
                        self.feature_dim = features.shape[1]
                    else:
                        self.feature_dim = features.shape[-1]
            except Exception:
                model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
                self.backbone = nn.Sequential(*list(model.children())[:-2])
                self.feature_dim = 2048
        else:
            model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            self.backbone = nn.Sequential(*list(model.children())[:-2])
            self.feature_dim = 2048
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        
        # Feature fusion: combine image features with NDVI and Height
        self.feature_fusion = nn.Sequential(
            nn.Linear(512 + 2, 512),  # +2 for NDVI and Height
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        
        # Separate heads for each target type
        self.target_heads = nn.ModuleDict()
        for target_type in self.target_types:
            self.target_heads[target_type] = nn.Sequential(
                nn.Dropout(dropout / 2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(dropout / 4),
                nn.Linear(256, 1)
            )
    
    def forward(self, x, ndvi=None, height=None, target_type=None):
        """
        Forward pass.
        
        Args:
            x: Input images
            ndvi: NDVI values (optional)
            height: Height values (optional)
            target_type: If specified, only return prediction for this target type
            
        Returns:
            Dictionary of predictions for each target type, or single prediction
        """
        # Extract features
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.view(features.size(0), -1)
        elif len(features.shape) == 3:
            features = features.mean(dim=1)
        else:
            features = features.flatten(1) if len(features.shape) > 2 else features
        
        if features.shape[1] != self.feature_dim:
            features = features[:, :self.feature_dim]
        
        # Shared features
        shared = self.shared_features(features)
        
        # Fuse with NDVI and Height if provided
        if ndvi is not None and height is not None:
            # Normalize features
            ndvi_norm = (ndvi - 0.5) / 0.3  # Approximate normalization
            height_norm = (height - 15.0) / 10.0  # Approximate normalization
            aux_features = torch.stack([ndvi_norm, height_norm], dim=1)
            fused = torch.cat([shared, aux_features], dim=1)
            fused = self.feature_fusion(fused)
        else:
            fused = shared
        
        # Predictions for all target types
        predictions = {}
        for target_type_name in self.target_types:
            pred = self.target_heads[target_type_name](fused)
            predictions[target_type_name] = pred.squeeze(-1)
        
        if target_type:
            return predictions.get(target_type, predictions[self.target_types[0]])
        
        return predictions



# ============================================================================
# DATASET CLASS
# ============================================================================

class TestDataset(Dataset):
    def __init__(self, test_df, image_dir, transform):
        self.test_df = test_df
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.unique_images = test_df['image_path'].unique()
        
    def __len__(self):
        return len(self.unique_images)
    
    def __getitem__(self, idx):
        img_path = self.unique_images[idx]
        full_path = self.image_dir / img_path
        
        try:
            image = Image.open(full_path).convert('RGB')
            image = np.array(image)
        except:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {'image': image, 'image_path': img_path}

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📁 Loading data...")
test_df = pd.read_csv('/kaggle/input/csiro-biomass/test.csv')
print(f"Test samples: {len(test_df)}")

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n🤖 Loading model...")
model = AdvancedMultiTaskBiomassModel(pretrained=False).to(device)

# Try to load checkpoint
checkpoint_paths = [
    '/kaggle/input/csiro-biomass-model/best_multitask_model.pth',
    '/kaggle/input/csiro-biomass-model/models/checkpoints/best_multitask_model.pth',
    '/kaggle/input/csiro-biomass-model/checkpoints/best_multitask_model.pth',
]

checkpoint_loaded = False
for checkpoint_path in checkpoint_paths:
    if os.path.exists(checkpoint_path):
        print(f"✅ Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint_loaded = True
        print(f"✅ Model loaded! Best val loss: {checkpoint.get('val_loss', 'N/A')}")
        break

if not checkpoint_loaded:
    print("⚠️  No checkpoint found. Using pretrained ImageNet weights only.")
    print("   Note: Performance will be lower without fine-tuning.")
    model = AdvancedMultiTaskBiomassModel(pretrained=True).to(device)

model.eval()

# ============================================================================
# PREPARE TEST DATA
# ============================================================================

print("\n🔧 Preparing test data...")
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Find image directory
image_dir_candidates = [
    '/kaggle/input/csiro-biomass/test',
    '/kaggle/input/csiro-biomass/train',
    '/kaggle/input/csiro-biomass',
]

image_dir = '/kaggle/input/csiro-biomass'
for candidate in image_dir_candidates:
    if os.path.exists(candidate):
        test_img_path = test_df['image_path'].iloc[0]
        if os.path.exists(os.path.join(candidate, test_img_path)) or \
           os.path.exists(os.path.join(candidate, os.path.basename(test_img_path))):
            image_dir = candidate
            break

print(f"Using image directory: {image_dir}")

test_dataset = TestDataset(test_df, image_dir, transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n🔮 Generating predictions...")
image_predictions = {}

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(device)
        predictions = model(images)
        
        batch_size = len(batch['image_path'])
        for i in range(batch_size):
            img_path = batch['image_path'][i]
            pred_dict = {
                target_type: float(predictions[target_type][i].cpu().item())
                for target_type in model.target_types
            }
            image_predictions[img_path] = pred_dict

# ============================================================================
# CREATE SUBMISSION
# ============================================================================

print("\n📝 Creating submission file...")
submission_rows = []

for _, row in test_df.iterrows():
    img_path = row['image_path']
    target_name = row['target_name']
    
    if img_path in image_predictions:
        pred = image_predictions[img_path].get(target_name, 0.0)
    else:
        pred = 0.0
    
    pred = max(0.0, pred)
    
    submission_rows.append({
        'sample_id': row['sample_id'],
        'target': pred
    })

submission_df = pd.DataFrame(submission_rows)

# Apply constraint: Dry_Total = sum of components
for img_path in test_df['image_path'].unique():
    img_rows = test_df[test_df['image_path'] == img_path]
    sub_rows = submission_df[submission_df['sample_id'].isin(img_rows['sample_id'])]
    
    if len(sub_rows) >= 4:
        clover_mask = sub_rows['sample_id'].str.contains('Dry_Clover_g', na=False)
        dead_mask = sub_rows['sample_id'].str.contains('Dry_Dead_g', na=False)
        green_mask = sub_rows['sample_id'].str.contains('Dry_Green_g', na=False)
        total_mask = sub_rows['sample_id'].str.contains('Dry_Total_g', na=False)
        
        if clover_mask.any() and dead_mask.any() and green_mask.any() and total_mask.any():
            clover_val = sub_rows[clover_mask]['target'].values[0]
            dead_val = sub_rows[dead_mask]['target'].values[0]
            green_val = sub_rows[green_mask]['target'].values[0]
            calculated_total = clover_val + dead_val + green_val
            
            submission_df.loc[sub_rows[total_mask].index[0], 'target'] = calculated_total

# Round to 6 decimal places
submission_df['target'] = submission_df['target'].round(6)

# Save submission
submission_df.to_csv('submission.csv', index=False)

print("\n✅ Submission file created: submission.csv")
print(f"\n📊 Submission Statistics:")
print(f"   Total predictions: {len(submission_df)}")
print(f"   Target range: {submission_df['target'].min():.2f} - {submission_df['target'].max():.2f}")
print(f"   Mean: {submission_df['target'].mean():.2f}")
print(f"   Std: {submission_df['target'].std():.2f}")

print("\n🎯 Predictions by target type:")
for target_name in sorted(test_df['target_name'].unique()):
    mask = test_df['target_name'] == target_name
    preds = submission_df[mask]['target']
    if len(preds) > 0:
        print(f"   {target_name}: {preds.values[0]:.6f}")

print("\n✨ Ready to submit!")
