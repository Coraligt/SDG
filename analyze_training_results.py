# analyze_training_results.py

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, List


class TrainingAnalyzer:
    """Analyze and visualize training results."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_dir = self.checkpoint_dir / 'metrics'
        
        # Load metrics
        self.metrics = self._load_metrics()
        
    def _load_metrics(self) -> Dict:
        """Load training metrics from file."""
        pkl_path = self.metrics_dir / 'training_metrics.pkl'
        json_path = self.metrics_dir / 'training_metrics.json'
        
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        elif json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError("No metrics file found!")
            
    def print_summary(self):
        """Print training summary statistics."""
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        # Training info
        num_epochs = len(self.metrics['train']['loss'])
        print(f"\nTotal epochs trained: {num_epochs}")
        
        # Best performance
        val_losses = self.metrics['val']['loss']
        best_epoch = np.argmin(val_losses)
        best_loss = val_losses[best_epoch]
        
        print(f"\nBest validation loss: {best_loss:.4f} (epoch {best_epoch + 1})")
        
        # Final performance
        final_train_loss = self.metrics['train']['loss'][-1]
        final_val_loss = self.metrics['val']['loss'][-1]
        print(f"\nFinal train loss: {final_train_loss:.4f}")
        print(f"Final val loss: {final_val_loss:.4f}")
        
        # Breakdown prediction accuracy
        if 'breakdown_mae' in self.metrics['val']:
            final_breakdown_mae = self.metrics['val']['breakdown_mae'][-1]
            best_breakdown_mae = min(self.metrics['val']['breakdown_mae'])
            best_breakdown_epoch = np.argmin(self.metrics['val']['breakdown_mae'])
            
            print(f"\nBreakdown prediction MAE:")
            print(f"  Final: {final_breakdown_mae:.2f} cycles")
            print(f"  Best: {best_breakdown_mae:.2f} cycles (epoch {best_breakdown_epoch + 1})")
            
        # Final defect prediction accuracy
        if 'final_defect_mae' in self.metrics['val']:
            final_defect_mae = self.metrics['val']['final_defect_mae'][-1]
            best_defect_mae = min(self.metrics['val']['final_defect_mae'])
            
            print(f"\nFinal defect count MAE:")
            print(f"  Final: {final_defect_mae:.2f}")
            print(f"  Best: {best_defect_mae:.2f}")
            
        # Training time
        if 'epoch_time' in self.metrics['train']:
            total_time = sum(self.metrics['train']['epoch_time'])
            avg_time = np.mean(self.metrics['train']['epoch_time'])
            print(f"\nTraining time:")
            print(f"  Total: {total_time/60:.1f} minutes")
            print(f"  Average per epoch: {avg_time:.1f} seconds")
            
        # Loss components analysis
        print("\n" + "-"*50)
        print("LOSS COMPONENTS (final epoch)")
        print("-"*50)
        
        components = ['prediction', 'breakdown', 'monotonic', 'smoothness', 
                     'physics', 'generation', 'cycle']
        
        for component in components:
            if component in self.metrics['train'] and self.metrics['train'][component]:
                train_val = self.metrics['train'][component][-1]
                val_val = self.metrics['val'][component][-1]
                print(f"{component.capitalize():15s}: Train={train_val:.4f}, Val={val_val:.4f}")
                
    def plot_learning_curves(self, save_path: str = None):
        """Plot detailed learning curves."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        # Components to plot
        components = ['loss', 'prediction', 'breakdown', 'monotonic', 
                     'smoothness', 'physics', 'generation', 'cycle', 'breakdown_mae']
        
        for idx, component in enumerate(components):
            ax = axes[idx]
            
            # Plot train
            if component in self.metrics['train'] and self.metrics['train'][component]:
                epochs = range(1, len(self.metrics['train'][component]) + 1)
                ax.plot(epochs, self.metrics['train'][component], 
                       'b-', label='Train', linewidth=2, alpha=0.8)
                
            # Plot val
            if component in self.metrics['val'] and self.metrics['val'][component]:
                epochs = range(1, len(self.metrics['val'][component]) + 1)
                ax.plot(epochs, self.metrics['val'][component], 
                       'r-', label='Val', linewidth=2, alpha=0.8)
                
            ax.set_xlabel('Epoch')
            ax.set_ylabel(component.replace('_', ' ').capitalize())
            ax.set_title(f'{component.replace("_", " ").capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add best value annotation
            if component in self.metrics['val'] and self.metrics['val'][component]:
                val_data = self.metrics['val'][component]
                best_val = min(val_data)
                best_epoch = np.argmin(val_data) + 1
                ax.axhline(best_val, color='green', linestyle='--', alpha=0.5)
                ax.text(0.02, 0.98, f'Best: {best_val:.3f} @ ep{best_epoch}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Training Progress - All Metrics', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
    def plot_overfitting_analysis(self, save_path: str = None):
        """Analyze overfitting by comparing train and val losses."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Train vs Val loss
        epochs = range(1, len(self.metrics['train']['loss']) + 1)
        train_loss = self.metrics['train']['loss']
        val_loss = self.metrics['val']['loss']
        
        ax1.plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, val_loss, 'r-', label='Val', linewidth=2)
        ax1.fill_between(epochs, train_loss, val_loss, alpha=0.3, color='gray')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Train vs Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss gap over time
        loss_gap = np.array(val_loss) - np.array(train_loss)
        ax2.plot(epochs, loss_gap, 'g-', linewidth=2)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(epochs, 0, loss_gap, alpha=0.3, color='green')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation - Train Loss')
        ax2.set_title('Generalization Gap')
        ax2.grid(True, alpha=0.3)
        
        # Add overfitting indicator
        if loss_gap[-1] > loss_gap[len(loss_gap)//2]:
            ax2.text(0.5, 0.95, 'Overfitting Detected!', 
                    transform=ax2.transAxes, ha='center',
                    color='red', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
    def analyze_convergence(self):
        """Analyze training convergence."""
        print("\n" + "="*50)
        print("CONVERGENCE ANALYSIS")
        print("="*50)
        
        # Check if training converged
        val_loss = self.metrics['val']['loss']
        last_10_percent = int(0.1 * len(val_loss))
        
        if last_10_percent > 1:
            recent_losses = val_loss[-last_10_percent:]
            loss_std = np.std(recent_losses)
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            print(f"\nLast {last_10_percent} epochs:")
            print(f"  Loss std deviation: {loss_std:.4f}")
            print(f"  Loss trend: {loss_trend:.6f} per epoch")
            
            if abs(loss_trend) < 0.0001 and loss_std < 0.01:
                print("  Status: CONVERGED ✓")
            elif loss_trend > 0:
                print("  Status: DIVERGING ✗")
            else:
                print("  Status: STILL IMPROVING")
                
        # Learning rate analysis
        print("\nLearning rate impact:")
        early_improvement = val_loss[0] - val_loss[min(10, len(val_loss)-1)]
        late_improvement = val_loss[-min(10, len(val_loss))] - val_loss[-1]
        
        print(f"  First 10 epochs improvement: {early_improvement:.4f}")
        print(f"  Last 10 epochs improvement: {late_improvement:.4f}")
        
        if late_improvement < 0.1 * early_improvement:
            print("  Suggestion: Learning rate might be too low now")
            
    def export_metrics_table(self, save_path: str = None):
        """Export key metrics to a table."""
        data = []
        
        num_epochs = len(self.metrics['train']['loss'])
        
        for epoch in range(num_epochs):
            row = {
                'Epoch': epoch + 1,
                'Train_Loss': self.metrics['train']['loss'][epoch],
                'Val_Loss': self.metrics['val']['loss'][epoch],
            }
            
            # Add other metrics if available
            for metric in ['breakdown_mae', 'final_defect_mae']:
                if metric in self.metrics['val'] and epoch < len(self.metrics['val'][metric]):
                    row[f'Val_{metric}'] = self.metrics['val'][metric][epoch]
                    
            data.append(row)
            
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\nMetrics exported to {save_path}")
        
        # Display summary statistics
        print("\nMetrics Summary:")
        print(df.describe())
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='checkpoints',
                       help='Directory containing checkpoints and metrics')
    parser.add_argument('--output_dir', type=str,
                       default='analysis_results',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = TrainingAnalyzer(args.checkpoint_dir)
    
    # Print summary
    analyzer.print_summary()
    
    # Generate plots
    print("\nGenerating analysis plots...")
    analyzer.plot_learning_curves(save_path=output_dir / 'learning_curves.png')
    analyzer.plot_overfitting_analysis(save_path=output_dir / 'overfitting_analysis.png')
    
    # Convergence analysis
    analyzer.analyze_convergence()
    
    # Export metrics
    analyzer.export_metrics_table(save_path=output_dir / 'training_metrics.csv')
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()