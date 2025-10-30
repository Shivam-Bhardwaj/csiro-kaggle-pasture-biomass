"""
Visualize training progress from logs.
"""

import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)


def parse_training_log(log_path):
    """Parse training log to extract metrics."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract epochs
    epochs = re.findall(r'Epoch (\d+)/100', content)
    
    # Extract losses
    train_losses = re.findall(r'Train Loss: ([\d.]+)', content)
    val_losses = re.findall(r'Val Loss: ([\d.]+)', content)
    
    # Extract RMSE for each target type
    targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    train_rmse = {target: [] for target in targets}
    val_rmse = {target: [] for target in targets}
    
    for target in targets:
        train_pattern = rf'{target}:.*?Train RMSE: ([\d.]+)'
        val_pattern = rf'{target}:.*?Val RMSE: ([\d.]+)'
        train_rmse[target] = re.findall(train_pattern, content)
        val_rmse[target] = re.findall(val_pattern, content)
    
    # Create DataFrame
    data = []
    max_len = max(len(epochs), len(train_losses), len(val_losses))
    
    for i in range(max_len):
        if i < len(epochs):
            epoch = int(epochs[i])
            train_loss = float(train_losses[i]) if i < len(train_losses) else None
            val_loss = float(val_losses[i]) if i < len(val_losses) else None
            
            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            
            for target in targets:
                row[f'train_rmse_{target}'] = float(train_rmse[target][i]) if i < len(train_rmse[target]) else None
                row[f'val_rmse_{target}'] = float(val_rmse[target][i]) if i < len(val_rmse[target]) else None
            
            data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_visualizations(df, output_dir='visualizations'):
    """Create training visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RMSE by target type
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    targets = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        train_col = f'train_rmse_{target}'
        val_col = f'val_rmse_{target}'
        
        if train_col in df.columns:
            ax.plot(df['epoch'], df[train_col], label='Train RMSE', 
                   linewidth=2, color=colors[idx], alpha=0.7)
        if val_col in df.columns:
            ax.plot(df['epoch'], df[val_col], label='Val RMSE', 
                   linewidth=2, color=colors[idx], linestyle='--')
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.set_title(target.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_by_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Final validation RMSE comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    final_epoch = df.iloc[-1]
    val_rmse_values = []
    target_labels = []
    
    for target in targets:
        col = f'val_rmse_{target}'
        if col in df.columns and not pd.isna(final_epoch[col]):
            val_rmse_values.append(final_epoch[col])
            target_labels.append(target.replace('_', ' '))
    
    bars = ax.bar(target_labels, val_rmse_values, color=colors[:len(val_rmse_values)])
    ax.set_xlabel('Target Type', fontsize=12)
    ax.set_ylabel('Validation RMSE', fontsize=12)
    ax.set_title('Final Validation RMSE by Target Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'final_rmse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Training progress summary
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Loss improvement
    ax = axes[0, 0]
    initial_loss = df['val_loss'].iloc[0]
    final_loss = df['val_loss'].iloc[-1]
    improvement = ((initial_loss - final_loss) / initial_loss) * 100
    ax.bar(['Initial', 'Final'], [initial_loss, final_loss], color=['#d62728', '#2ca02c'])
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.set_title(f'Loss Improvement: {improvement:.1f}%', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Best epoch
    ax = axes[0, 1]
    best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    best_loss = df['val_loss'].min()
    ax.bar(['Best Epoch'], [best_epoch], color='#9467bd')
    ax.set_ylabel('Epoch', fontsize=11)
    ax.set_title(f'Best Model: Epoch {best_epoch} (Loss: {best_loss:.2f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # RMSE improvement per target
    ax = axes[1, 0]
    improvements = []
    target_short = []
    for target in targets:
        col = f'val_rmse_{target}'
        if col in df.columns:
            initial = df[col].iloc[0]
            final = df[col].iloc[-1]
            if not pd.isna(initial) and not pd.isna(final):
                imp = ((initial - final) / initial) * 100
                improvements.append(imp)
                target_short.append(target.replace('_g', '').replace('_', ' '))
    
    bars = ax.bar(range(len(improvements)), improvements, color=colors[:len(improvements)])
    ax.set_xticks(range(len(improvements)))
    ax.set_xticklabels(target_short, rotation=45, ha='right')
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('RMSE Improvement by Target', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Training epochs
    ax = axes[1, 1]
    total_epochs = len(df)
    ax.bar(['Total Epochs'], [total_epochs], color='#ff7f0e')
    ax.set_ylabel('Epochs', fontsize=11)
    ax.set_title(f'Training Duration: {total_epochs} Epochs', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {output_dir}/")
    print(f"   - loss_curves.png")
    print(f"   - rmse_by_target.png")
    print(f"   - final_rmse_comparison.png")
    print(f"   - training_summary.png")


def main():
    """Main function."""
    log_path = 'logs/training_multitask.log'
    
    if not Path(log_path).exists():
        print(f"❌ Log file not found: {log_path}")
        return
    
    print("📊 Parsing training log...")
    df = parse_training_log(log_path)
    
    print(f"✅ Parsed {len(df)} epochs")
    print(f"\n📈 Training Statistics:")
    print(f"   Total epochs: {len(df)}")
    print(f"   Initial val loss: {df['val_loss'].iloc[0]:.4f}")
    print(f"   Final val loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"   Best val loss: {df['val_loss'].min():.4f} (Epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})")
    
    print("\n🎨 Creating visualizations...")
    create_visualizations(df)
    
    # Save CSV for reference
    csv_path = Path('visualizations') / 'training_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Training metrics saved to {csv_path}")


if __name__ == '__main__':
    main()

