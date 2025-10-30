"""
Generate predictions using advanced multi-task model.
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.models.advanced_multitask import AdvancedMultiTaskBiomassModel
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions with multi-task model')
    parser.add_argument('--model', type=str, default='models/checkpoints/best_multitask_model.pth')
    parser.add_argument('--test-csv', type=str, default='data/raw/test.csv')
    parser.add_argument('--output', type=str, default='submissions/submission_multitask.csv')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


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
        full_path = self.image_dir.parent / img_path
        
        try:
            image = Image.open(full_path).convert('RGB')
            image = np.array(image)
        except:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {'image': image, 'image_path': img_path}


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    
    set_seed(42)
    logger = setup_logger()
    logger.info("Generating predictions with advanced multi-task model...")
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = AdvancedMultiTaskBiomassModel(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)
    
    # Create dataset
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    test_dataset = TestDataset(test_df, config['data']['image_dir'], transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Generate predictions
    image_predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            predictions = model(images)
            
            # Handle batch predictions
            batch_size = len(batch['image_path'])
            for i in range(batch_size):
                img_path = batch['image_path'][i]
                pred_dict = {
                    target_type: float(predictions[target_type][i].cpu().item())
                    for target_type in model.target_types
                }
                image_predictions[img_path] = pred_dict
    
    # Create submission
    submission_rows = []
    for _, row in test_df.iterrows():
        img_path = row['image_path']
        target_name = row['target_name']
        
        if img_path in image_predictions:
            pred = image_predictions[img_path].get(target_name, 0.0)
        else:
            pred = 0.0
        
        # Ensure non-negative
        pred = max(0.0, pred)
        
        submission_rows.append({
            'sample_id': row['sample_id'],
            'target': pred
        })
    
    submission_df = pd.DataFrame(submission_rows)
    
    # Apply constraint: Dry_Total = sum of components
    for img_path in test_df['image_path'].unique():
        img_rows = test_df[test_df['image_path'] == img_path]
        if len(img_rows) >= 4:
            indices = submission_df[submission_df['sample_id'].isin(img_rows['sample_id'])].index
            
            if len(indices) >= 4:
                # Get component predictions
                clover_idx = submission_df[submission_df['sample_id'].str.contains('Dry_Clover_g')].index
                dead_idx = submission_df[submission_df['sample_id'].str.contains('Dry_Dead_g')].index
                green_idx = submission_df[submission_df['sample_id'].str.contains('Dry_Green_g')].index
                total_idx = submission_df[submission_df['sample_id'].str.contains('Dry_Total_g')].index
                
                if len(clover_idx) > 0 and len(dead_idx) > 0 and len(green_idx) > 0 and len(total_idx) > 0:
                    # Ensure consistency
                    clover_val = submission_df.loc[clover_idx[0], 'target']
                    dead_val = submission_df.loc[dead_idx[0], 'target']
                    green_val = submission_df.loc[green_idx[0], 'target']
                    calculated_total = clover_val + dead_val + green_val
                    
                    # Update Dry_Total to match sum
                    submission_df.loc[total_idx[0], 'target'] = calculated_total
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Predictions by target type:")
    for target_type in model.target_types:
        mask = test_df['target_name'] == target_type
        if mask.any():
            preds = submission_df[mask]['target']
            logger.info(f"  {target_type}: mean={preds.mean():.2f}, range=[{preds.min():.2f}, {preds.max():.2f}]")


if __name__ == '__main__':
    main()

