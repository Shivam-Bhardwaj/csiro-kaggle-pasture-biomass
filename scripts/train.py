#!/usr/bin/env python
"""
Training script for the CSIRO Pasture Biomass Estimation competition.
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
from src.models import BiomassRegressionModel
from src.training import RMSELoss, calculate_metrics, EarlyStopping


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train model for pasture biomass estimation')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--data-csv', type=str, default=None,
                        help='Path to training CSV file')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Path to image directory')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        biomass = batch['biomass'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, biomass)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(biomass.cpu().numpy())
        
        pbar.set_postfix({'loss': running_loss / len(predictions)})
    
    avg_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(np.array(predictions), np.array(targets))
    
    return avg_loss, metrics


def validate_epoch(model, dataloader, criterion, device, logger):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            biomass = batch['biomass'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, biomass)
            
            running_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(biomass.cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss / len(predictions)})
    
    avg_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(np.array(predictions), np.array(targets))
    
    return avg_loss, metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup logger
    logger = setup_logger(log_dir=config['paths']['log_dir'])
    logger.info("Starting training process...")
    logger.info(f"Configuration: {config}")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.2f} GB")
    
    # Prepare data paths
    data_csv = args.data_csv or config['data']['train_csv']
    image_dir = args.image_dir or config['data']['image_dir']
    
    # Check if data exists, if not create synthetic data for demo
    if not Path(data_csv).exists():
        logger.warning(f"Training CSV not found at {data_csv}. Creating synthetic dataset for demonstration...")
        Path(data_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(image_dir).mkdir(parents=True, exist_ok=True)
        
        # Create synthetic dataset
        from src.data.create_synthetic_data import create_synthetic_dataset
        create_synthetic_dataset(data_csv, image_dir, num_samples=1000)
        logger.info(f"Created synthetic dataset with 1000 samples")
    
    # Load data
    df = pd.read_csv(data_csv)
    logger.info(f"Loaded {len(df)} training samples")
    
    # Split data
    train_df, val_df = train_test_split(
        df, test_size=config['data']['val_split'], 
        random_state=config.get('seed', 42)
    )
    
    # Save splits
    train_split_path = Path(data_csv).parent / 'train_split.csv'
    val_split_path = Path(data_csv).parent / 'val_split.csv'
    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)
    
    # Create datasets
    train_transform = get_train_transforms(tuple(config['data']['image_size']))
    val_transform = get_val_transforms(tuple(config['data']['image_size']))
    
    train_dataset = PastureBiomassDataset(
        train_split_path, image_dir, transform=train_transform, is_train=True
    )
    val_dataset = PastureBiomassDataset(
        val_split_path, image_dir, transform=val_transform, is_train=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = BiomassRegressionModel(
        model_name=config['model']['name'],
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)
    
    logger.info(f"Created model: {config['model']['name']}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = RMSELoss()
    
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            momentum=0.9
        )
    
    # Scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        mode='min'
    )
    
    # Training loop
    best_val_loss = float('inf')
    results = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'train_r2': [],
        'val_r2': []
    }
    
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    for epoch in range(config['training']['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, logger)
        
        # Update scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Log results
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['train_rmse'].append(train_metrics['rmse'])
        results['val_rmse'].append(val_metrics['rmse'])
        results['train_r2'].append(train_metrics['r2'])
        results['val_r2'].append(val_metrics['r2'])
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train RMSE: {train_metrics['rmse']:.4f}, Train R²: {train_metrics['r2']:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val RMSE: {val_metrics['rmse']:.4f}, Val R²: {val_metrics['r2']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config['training']['save_best_only']:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                }, checkpoint_dir / 'best_model.pth')
                logger.info(f"Saved best model (Val Loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    logger.info(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    
    # Save final results
    import json
    results_path = Path(config['paths']['log_dir']) / 'training_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()

