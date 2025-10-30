#!/usr/bin/env python
"""
Submission script for the CSIRO Pasture Biomass Estimation competition.
"""

import argparse
import pandas as pd
from pathlib import Path

from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create submission file for Kaggle')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions CSV file')
    parser.add_argument('--output', type=str, default='submissions/submission.csv',
                        help='Path to save submission file')
    parser.add_argument('--format', type=str, default='csv',
                        help='Submission format (csv or zip)')
    return parser.parse_args()


def main():
    """Main submission function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info("Creating submission file...")
    
    # Load predictions
    predictions = pd.read_csv(args.predictions)
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Validate submission format
    # TODO: Add validation logic based on competition requirements
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save submission file
    predictions.to_csv(output_path, index=False)
    logger.info(f"Submission file saved to {output_path}")
    
    # Create zip file if requested
    if args.format == 'zip':
        import zipfile
        zip_path = output_path.with_suffix('.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(output_path, output_path.name)
        logger.info(f"Zipped submission saved to {zip_path}")
    
    logger.info("Submission file created successfully!")
    logger.info("You can now submit to Kaggle using:")
    logger.info(f"  kaggle competitions submit -c csiro-pasture-biomass-estimation -f {output_path}")


if __name__ == '__main__':
    main()

