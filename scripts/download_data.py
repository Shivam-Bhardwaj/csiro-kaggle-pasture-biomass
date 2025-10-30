#!/usr/bin/env python
"""
Background script to download CSIRO Kaggle competition dataset.
This script runs continuously and attempts to download the dataset once Kaggle credentials are available.
"""

import subprocess
import sys
import time
import os
from pathlib import Path
import json

def check_kaggle_credentials():
    """Check if Kaggle credentials are available."""
    kaggle_paths = [
        Path.home() / '.kaggle' / 'kaggle.json',
        Path.home() / '.config' / 'kaggle' / 'kaggle.json',
    ]
    
    for path in kaggle_paths:
        if path.exists():
            try:
                with open(path) as f:
                    creds = json.load(f)
                    if 'username' in creds and 'key' in creds:
                        print(f"✓ Found Kaggle credentials at {path}")
                        return True
            except:
                pass
    
    # Check environment variables
    if os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'):
        print("✓ Found Kaggle credentials in environment variables")
        return True
    
    return False

def download_dataset():
    """Download the competition dataset."""
    
    # Competition name (confirmed: csiro-biomass)
    competition_names = [
        'csiro-biomass',  # Official competition name
        'csiro-pasture-biomass-estimation',
        'csiro-pasture-biomass',
        'pasture-biomass-estimation',
    ]
    
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Attempting to download dataset...")
    print(f"Output directory: {data_dir.absolute()}")
    
    # Try to find the competition
    for comp_name in competition_names:
        print(f"\nTrying competition: {comp_name}")
        try:
            cmd = [
                'kaggle', 'competitions', 'download',
                '-c', comp_name,
                '-p', str(data_dir)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print(f"\n✓✓✓ SUCCESS! Downloaded dataset from {comp_name} ✓✓✓")
                print(result.stdout)
                
                # Extract zip files
                import zipfile
                for zip_file in data_dir.glob('*.zip'):
                    print(f"\nExtracting {zip_file.name}...")
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    zip_file.unlink()  # Remove zip file
                    print(f"✓ Extracted and removed {zip_file.name}")
                
                # List downloaded files
                print("\nDownloaded files:")
                for file in sorted(data_dir.rglob('*')):
                    if file.is_file():
                        size_mb = file.stat().st_size / (1024 * 1024)
                        print(f"  {file.relative_to(data_dir)} ({size_mb:.2f} MB)")
                
                return True
            else:
                print(f"✗ Competition '{comp_name}' not found")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"✗ Download timed out for {comp_name}")
        except FileNotFoundError:
            print("✗ Kaggle CLI not found. Install with: pip install kaggle")
            return False
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return False

def main():
    """Main function - runs continuously until dataset is downloaded."""
    print("=" * 60)
    print("CSIRO Kaggle Competition - Dataset Downloader")
    print("=" * 60)
    print("\nThis script will continuously check for Kaggle credentials")
    print("and attempt to download the competition dataset.")
    print("\nTo set up Kaggle API:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Create API token (download kaggle.json)")
    print("3. Place kaggle.json in ~/.kaggle/ or ~/.config/kaggle/")
    print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
    print("\nOr set environment variables:")
    print("  export KAGGLE_USERNAME=your_username")
    print("  export KAGGLE_KEY=your_api_key")
    print("=" * 60)
    
    max_attempts = None  # None = infinite attempts
    attempt = 0
    check_interval = 300  # Check every 5 minutes
    
    while max_attempts is None or attempt < max_attempts:
        attempt += 1
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Attempt {attempt}")
        
        # Check credentials
        if not check_kaggle_credentials():
            print("⚠ Kaggle credentials not found. Waiting...")
            print(f"   Will check again in {check_interval // 60} minutes")
            time.sleep(check_interval)
            continue
        
        # Try to download
        if download_dataset():
            print("\n✓✓✓ Dataset download completed successfully! ✓✓✓")
            break
        
        print(f"\nWill retry in {check_interval // 60} minutes...")
        time.sleep(check_interval)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(0)
