#!/usr/bin/env python
"""
Generate predictions for CSIRO competition test set.
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
    parser = argparse.ArgumentParser(description='Generate predictions for test set')
    parser.add_argument('--model', type=str, default='models/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--test-csv', type=str, default='data/raw/test.csv',
                        help='Path to test CSV')
    parser.add_argument('--output', type=str, default='submissions/submission.csv',
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
    logger.info("Starting prediction generation...")
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Best validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    # Create model
    model = BiomassRegressionModel(
        model_name=config['model']['name'],
        pretrained=False,  # We're loading weights
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded and set to evaluation mode")
    
    # Create test dataset - need to handle test data properly
    # For test set, we need to predict for each sample_id
    image_dir = config['data']['image_dir']
    test_transform = get_val_transforms(tuple(config['data']['image_size']))
    
    # Create a temporary CSV with all test samples for dataset loading
    test_dataset = PastureBiomassDataset(
        args.test_csv, image_dir, transform=test_transform, is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Generate predictions
    predictions = []
    image_ids = []
    
    logger.info("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch['image'].to(device)
            outputs = model(images)
            
            predictions.extend(outputs.cpu().numpy())
            
            # Get image IDs
            if 'image_id' in batch:
                image_ids.extend(batch['image_id'])
            else:
                # Fallback: use indices
                batch_start = len(image_ids)
                image_ids.extend([f"test_{batch_start + i}" for i in range(len(outputs))])
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'].values,
        'target': predictions[:len(test_df)]
    })
    
    # Ensure we have predictions for all test samples
    if len(submission_df) < len(test_df):
        logger.warning(f"Only {len(submission_df)} predictions generated for {len(test_df)} test samples")
        # Pad with zeros if needed
        missing = len(test_df) - len(submission_df)
        submission_df = pd.concat([
            submission_df,
            pd.DataFrame({
                'sample_id': test_df['sample_id'].values[-missing:],
                'target': [0.0] * missing
            })
        ])
    
    # Save submission
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"Submission file saved to {output_path}")
    logger.info(f"Submission stats:")
    logger.info(f"  Total predictions: {len(submission_df)}")
    logger.info(f"  Target range: {submission_df['target'].min():.2f} - {submission_df['target'].max():.2f}")
    logger.info(f"  Mean target: {submission_df['target'].mean():.2f}")
    
    # Show first few predictions
    logger.info(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    
    logger.info("\nSubmission file ready for Kaggle!")


if __name__ == '__main__':
    main()

