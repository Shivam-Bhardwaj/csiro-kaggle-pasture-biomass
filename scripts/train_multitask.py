"""
Advanced training script with multi-task learning.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.dataset import PastureBiomassDataset, get_train_transforms, get_val_transforms
from src.models.advanced_multitask import AdvancedMultiTaskBiomassModel
from src.training import RMSELoss, calculate_metrics, EarlyStopping


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train advanced multi-task model')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()


def create_multitask_dataset(df, image_dir, transform, image_list=None):
    """Create dataset that returns targets for all target types."""
    # Filter by image_list if provided
    if image_list is not None:
        df = df[df['image_path'].isin(image_list)]
    
    # Group by image_path to get all targets for each image
    grouped = df.groupby('image_path')
    
    class MultiTaskDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = []
            for img_path, group in grouped:
                row = group.iloc[0]  # Use first row for metadata
                targets = {}
                for _, target_row in group.iterrows():
                    targets[target_row['target_name']] = target_row['target']
                
                # Ensure we have all 5 targets
                expected_targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
                for target_name in expected_targets:
                    if target_name not in targets:
                        targets[target_name] = 0.0  # Default if missing
                
                self.samples.append({
                    'image_path': img_path,
                    'targets': targets,
                    'ndvi': row.get('Pre_GSHH_NDVI', 0.0),
                    'height': row.get('Height_Ave_cm', 0.0),
                    'state': row.get('State', ''),
                    'species': row.get('Species', '')
                })
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Load image
            from PIL import Image
            import numpy as np
            import albumentations as A
            
            img_path = Path(image_dir) / sample['image_path']
            if not img_path.exists():
                img_path = Path(image_dir).parent / sample['image_path']
            
            try:
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
            except Exception as e:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Apply transform
            if transform:
                if isinstance(transform, A.Compose):
                    transformed = transform(image=image)
                    image = transformed['image']
                else:
                    image = transform(image)
            
            result = {'image': image}
            
            # Add all targets
            for target_type in ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']:
                result[target_type] = torch.tensor(float(sample['targets'][target_type]), dtype=torch.float32)
            
            # Add features
            result['ndvi'] = torch.tensor(float(sample['ndvi']), dtype=torch.float32)
            result['height'] = torch.tensor(float(sample['height']), dtype=torch.float32)
            
            return result
    
    return MultiTaskDataset()


def train_epoch_multitask(model, dataloader, criterion, optimizer, device, logger):
    """Train epoch for multi-task model."""
    model.train()
    running_loss = 0.0
    all_predictions = {target: [] for target in model.target_types}
    all_targets = {target: [] for target in model.target_types}
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        
        # Get all targets and features
        targets_dict = {}
        for target_type in model.target_types:
            if target_type in batch:
                targets_dict[target_type] = batch[target_type].to(device)
        
        ndvi = batch.get('ndvi', None)
        height = batch.get('height', None)
        if ndvi is not None:
            ndvi = ndvi.to(device)
        if height is not None:
            height = height.to(device)
        
        optimizer.zero_grad()
        predictions = model(images, ndvi=ndvi, height=height)
        
        # Calculate loss for each target type with weights
        # Higher weight for Dry_Total as it's the sum constraint
        target_weights = {
            'Dry_Clover_g': 1.0,
            'Dry_Dead_g': 1.0,
            'Dry_Green_g': 1.0,
            'Dry_Total_g': 1.5,  # Higher weight for total
            'GDM_g': 1.0
        }
        
        total_loss = 0.0
        for target_type in model.target_types:
            if target_type in targets_dict:
                pred = predictions[target_type]
                target = targets_dict[target_type]
                loss = criterion(pred, target)
                total_loss += target_weights.get(target_type, 1.0) * loss
                
                all_predictions[target_type].extend(pred.detach().cpu().numpy())
                all_targets[target_type].extend(target.cpu().numpy())
        
        # Add constraint loss: Dry_Total should equal sum of components
        if 'Dry_Total_g' in predictions and all(t in predictions for t in ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']):
            pred_total = predictions['Dry_Total_g']
            pred_sum = predictions['Dry_Clover_g'] + predictions['Dry_Dead_g'] + predictions['Dry_Green_g']
            constraint_loss = nn.functional.mse_loss(pred_total, pred_sum)
            total_loss += 0.5 * constraint_loss  # Constraint weight
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        pbar.set_postfix({'loss': running_loss / len(all_predictions[model.target_types[0]])})
    
    # Calculate metrics for each target
    metrics = {}
    for target_type in model.target_types:
        if len(all_predictions[target_type]) > 0:
            metrics[target_type] = calculate_metrics(
                np.array(all_predictions[target_type]),
                np.array(all_targets[target_type])
            )
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, metrics


def validate_epoch_multitask(model, dataloader, criterion, device, logger):
    """Validate epoch for multi-task model."""
    model.eval()
    running_loss = 0.0
    all_predictions = {target: [] for target in model.target_types}
    all_targets = {target: [] for target in model.target_types}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            
            targets_dict = {}
            for target_type in model.target_types:
                if target_type in batch:
                    targets_dict[target_type] = batch[target_type].to(device)
            
            ndvi = batch.get('ndvi', None)
            height = batch.get('height', None)
            if ndvi is not None:
                ndvi = ndvi.to(device)
            if height is not None:
                height = height.to(device)
            
            predictions = model(images, ndvi=ndvi, height=height)
            
            target_weights = {
                'Dry_Clover_g': 1.0,
                'Dry_Dead_g': 1.0,
                'Dry_Green_g': 1.0,
                'Dry_Total_g': 1.5,
                'GDM_g': 1.0
            }
            
            total_loss = 0.0
            for target_type in model.target_types:
                if target_type in targets_dict:
                    pred = predictions[target_type]
                    target = targets_dict[target_type]
                    loss = criterion(pred, target)
                    total_loss += target_weights.get(target_type, 1.0) * loss
                    
                    all_predictions[target_type].extend(pred.cpu().numpy())
                    all_targets[target_type].extend(target.cpu().numpy())
            
            # Constraint loss
            if 'Dry_Total_g' in predictions and all(t in predictions for t in ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g']):
                pred_total = predictions['Dry_Total_g']
                pred_sum = predictions['Dry_Clover_g'] + predictions['Dry_Dead_g'] + predictions['Dry_Green_g']
                constraint_loss = nn.functional.mse_loss(pred_total, pred_sum)
                total_loss += 0.5 * constraint_loss
            
            running_loss += total_loss.item()
    
    metrics = {}
    for target_type in model.target_types:
        if len(all_predictions[target_type]) > 0:
            metrics[target_type] = calculate_metrics(
                np.array(all_predictions[target_type]),
                np.array(all_targets[target_type])
            )
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, metrics


def main():
    """Main training function."""
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    
    set_seed(config.get('seed', 42))
    logger = setup_logger(log_dir=config['paths']['log_dir'])
    logger.info("Starting ADVANCED multi-task training...")
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_csv = config['data']['train_csv']
    image_dir = config['data']['image_dir']
    
    df = pd.read_csv(train_csv)
    logger.info(f"Loaded {len(df)} training samples")
    
    # Split by unique images (not samples)
    unique_images = df['image_path'].unique()
    train_images, val_images = train_test_split(
        unique_images, test_size=config['data']['val_split'], random_state=42
    )
    
    train_df = df[df['image_path'].isin(train_images)]
    val_df = df[df['image_path'].isin(val_images)]
    
    logger.info(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Create datasets
    train_transform = get_train_transforms(tuple(config['data']['image_size']))
    val_transform = get_val_transforms(tuple(config['data']['image_size']))
    
    train_dataset = create_multitask_dataset(
        df, image_dir, train_transform, image_list=train_images
    )
    val_dataset = create_multitask_dataset(
        df, image_dir, val_transform, image_list=val_images
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=config['data']['num_workers'], pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config['data']['num_workers'], pin_memory=True
    )
    
    # Create model
    model = AdvancedMultiTaskBiomassModel(
        model_name='resnet50',
        pretrained=True,
        dropout=0.5
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = RMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    early_stopping = EarlyStopping(patience=15, mode='min')
    
    best_val_loss = float('inf')
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_metrics = train_epoch_multitask(
            model, train_loader, criterion, optimizer, device, logger
        )
        
        val_loss, val_metrics = validate_epoch_multitask(
            model, val_loader, criterion, device, logger
        )
        
        scheduler.step()
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        for target_type in model.target_types:
            if target_type in train_metrics and target_type in val_metrics:
                logger.info(f"  {target_type}:")
                logger.info(f"    Train RMSE: {train_metrics[target_type]['rmse']:.4f}, "
                          f"Val RMSE: {val_metrics[target_type]['rmse']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, checkpoint_dir / 'best_multitask_model.pth')
            logger.info(f"Saved best model (Val Loss: {val_loss:.4f})")
        
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

