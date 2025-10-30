#!/usr/bin/env python
"""
Multi-target prediction script for CSIRO competition.
Predicts different values for each target_name type.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.dataset import PastureBiomassDataset, get_val_transforms
from src.models import BiomassRegressionModel
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate multi-target predictions')
    parser.add_argument('--model', type=str, default='models/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--test-csv', type=str, default='data/raw/test.csv',
                        help='Path to test CSV')
    parser.add_argument('--output', type=str, default='submissions/submission_multitarget.csv',
                        help='Path to save submission file')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main prediction function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting multi-target prediction generation...")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)
    logger.info(f"Loaded {len(test_df)} test samples")
    logger.info(f"Unique target names: {test_df['target_name'].unique()}")
    
    # Get unique images
    unique_images = test_df['image_path'].unique()
    logger.info(f"Unique test images: {len(unique_images)}")
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create model
    model = BiomassRegressionModel(
        model_name=config['model']['name'],
        pretrained=False,
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded and set to evaluation mode")
    
    # Create test dataset - one entry per unique image
    image_dir = config['data']['image_dir']
    test_transform = get_val_transforms(tuple(config['data']['image_size']))
    
    # Create a temporary CSV with unique images
    unique_images_df = pd.DataFrame({
        'sample_id': [Path(img).stem for img in unique_images],
        'image_path': unique_images
    })
    temp_csv = Path('data/raw/temp_test_unique.csv')
    unique_images_df.to_csv(temp_csv, index=False)
    
    test_dataset = PastureBiomassDataset(
        str(temp_csv), image_dir, transform=test_transform, is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Generate predictions for unique images
    logger.info("Generating predictions for unique images...")
    image_predictions = {}
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            images = batch['image'].to(device)
            outputs = model(images)
            
            # Map predictions to image paths
            batch_image_paths = unique_images[idx * config['data']['batch_size']:(idx + 1) * config['data']['batch_size']]
            for img_path, pred in zip(batch_image_paths, outputs.cpu().numpy()):
                image_predictions[img_path] = float(pred)
    
    # Clean up temp file
    temp_csv.unlink()
    
    # Create predictions for all test samples
    # Use the base prediction and adjust based on target_name
    # This is a simple approach - could be improved with actual multi-task model
    predictions = []
    
    # Get statistics from training data for target_name adjustments
    train_df = pd.read_csv(config['data']['train_csv'])
    target_stats = train_df.groupby('target_name')['target'].agg(['mean', 'std']).to_dict('index')
    
    logger.info("Creating predictions for all test samples...")
    for _, row in test_df.iterrows():
        img_path = row['image_path']
        target_name = row['target_name']
        
        # Get base prediction for this image
        base_pred = image_predictions.get(img_path, 0.0)
        
        # Adjust based on target_name statistics from training data
        if target_name in target_stats:
            # Use the ratio of mean target for this type vs overall mean
            overall_mean = train_df['target'].mean()
            target_mean = target_stats[target_name]['mean']
            if overall_mean > 0:
                adjustment_factor = target_mean / overall_mean
                pred = base_pred * adjustment_factor
            else:
                pred = base_pred
        else:
            # Fallback: use base prediction
            pred = base_pred
        
        predictions.append(max(0.0, pred))  # Ensure non-negative
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'].values,
        'target': predictions
    })
    
    # Save submission
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"Submission file saved to {output_path}")
    logger.info(f"Submission stats:")
    logger.info(f"  Total predictions: {len(submission_df)}")
    logger.info(f"  Target range: {submission_df['target'].min():.2f} - {submission_df['target'].max():.2f}")
    logger.info(f"  Mean target: {submission_df['target'].mean():.2f}")
    
    # Show predictions by target type
    logger.info(f"\nPredictions by target type:")
    for target_name in test_df['target_name'].unique():
        mask = test_df['target_name'] == target_name
        preds = submission_df[mask]['target']
        logger.info(f"  {target_name}: mean={preds.mean():.2f}, range=[{preds.min():.2f}, {preds.max():.2f}]")
    
    logger.info("\nMulti-target submission file ready!")


if __name__ == '__main__':
    main()

