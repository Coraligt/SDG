# training/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, List, Tuple
import os
import wandb
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

# PhysicsNeMo imports 
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger, RankZeroLoggingWrapper

import sys
sys.path.append('/storage/home/hcoda1/6/cli872/scratch/work/SDG')

from ferroelectric_dataset import create_dataloaders
from models.fe_surrogate import PhysicsInformedFerroelectricSurrogate
from models.losses import PhysicsInformedLoss, GenerationRateLoss, CycleLoss


class MetricsTracker:
    """Track and visualize training metrics."""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric storage
        self.metrics = {
            'train': {
                'loss': [],
                'prediction': [],
                'breakdown': [],
                'monotonic': [],
                'smoothness': [],
                'physics': [],
                'generation': [],
                'cycle': [],
                'epoch_time': []
            },
            'val': {
                'loss': [],
                'prediction': [],
                'breakdown': [],
                'monotonic': [],
                'smoothness': [],
                'physics': [],
                'generation': [],
                'cycle': [],
                'breakdown_mae': [],
                'final_defect_mae': []
            },
            'predictions': {
                'epochs': [],
                'samples': []  # Will store sample predictions at key epochs
            }
        }
        
    def update_train_metrics(self, epoch: int, metrics: Dict, epoch_time: float):
        """Update training metrics."""
        self.metrics['train']['epoch_time'].append(epoch_time)
        for key, value in metrics.items():
            if key in self.metrics['train'] and key != 'epoch_time':
                self.metrics['train'][key].append(value)
                
    def update_val_metrics(self, epoch: int, metrics: Dict):
        """Update validation metrics."""
        for key, value in metrics.items():
            if key in self.metrics['val']:
                self.metrics['val'][key].append(value)
                
    def save_predictions(self, epoch: int, predictions: Dict):
        """Save sample predictions for visualization."""
        self.metrics['predictions']['epochs'].append(epoch)
        self.metrics['predictions']['samples'].append(predictions)
        
    def plot_losses(self):
        """Plot training and validation losses."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Loss components to plot
        components = ['loss', 'prediction', 'breakdown', 'monotonic', 
                     'smoothness', 'physics', 'generation', 'cycle']
        
        for idx, component in enumerate(components):
            ax = axes[idx]
            
            # Plot train and val
            if component in self.metrics['train'] and self.metrics['train'][component]:
                epochs = range(1, len(self.metrics['train'][component]) + 1)
                ax.plot(epochs, self.metrics['train'][component], 
                       'b-', label='Train', alpha=0.8)
                
            if component in self.metrics['val'] and self.metrics['val'][component]:
                epochs = range(1, len(self.metrics['val'][component]) + 1)
                ax.plot(epochs, self.metrics['val'][component], 
                       'r-', label='Val', alpha=0.8)
                
            ax.set_xlabel('Epoch')
            ax.set_ylabel(component.capitalize())
            ax.set_title(f'{component.capitalize()} Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_validation_metrics(self):
        """Plot additional validation metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if self.metrics['val']['breakdown_mae']:
            epochs = range(1, len(self.metrics['val']['breakdown_mae']) + 1)
            
            # Breakdown MAE
            ax1.plot(epochs, self.metrics['val']['breakdown_mae'], 'g-', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Breakdown Cycle MAE')
            ax1.set_title('Breakdown Prediction Error')
            ax1.grid(True, alpha=0.3)
            
            # Final defect MAE
            ax2.plot(epochs, self.metrics['val']['final_defect_mae'], 'm-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Final Defect Count MAE')
            ax2.set_title('Final Defect Prediction Error')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'validation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_metrics(self):
        """Save metrics to file."""
        with open(self.save_dir / 'training_metrics.pkl', 'wb') as f:
            pickle.dump(self.metrics, f)
            
        # Also save as JSON for easy viewing
        import json
        
        # Convert numpy values to Python native types
        json_metrics = {}
        for split in ['train', 'val']:
            json_metrics[split] = {}
            for key, values in self.metrics[split].items():
                if isinstance(values, list) and values:
                    json_metrics[split][key] = [float(v) if isinstance(v, (np.ndarray, np.generic)) else v for v in values]
                    
        with open(self.save_dir / 'training_metrics.json', 'w') as f:
            json.dump(json_metrics, f, indent=2)


class PredictionVisualizer:
    """Visualize model predictions during training."""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_predictions(
        self, 
        epoch: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        breakdown_prob: torch.Tensor,
        valid_mask: torch.Tensor,
        voltages: torch.Tensor,
        sample_indices: List[int] = None
    ):
        """Visualize predictions vs targets for selected samples."""
        
        # Handle the case where predictions might be empty or a list
        if isinstance(predictions, list):
            if not predictions:
                self.logger.warning("Empty predictions list, skipping visualization")
                return None
            # Convert list to tensor if needed
            predictions = torch.stack(predictions) if all(isinstance(p, torch.Tensor) for p in predictions) else predictions[0]
        
        if sample_indices is None:
            # Select 6 random samples
            batch_size = predictions.size(0)
            sample_indices = np.random.choice(batch_size, min(6, batch_size), replace=False)
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, sample_idx in enumerate(sample_indices):
            ax = axes[idx]
            
            # Get data for this sample
            pred = predictions[sample_idx].cpu().numpy()
            target = targets[sample_idx].cpu().numpy()
            mask = valid_mask[sample_idx].cpu().numpy()
            breakdown = breakdown_prob[sample_idx].cpu().numpy()
            voltage = voltages[sample_idx].cpu().item()
            
            # Plot only valid timesteps
            timesteps = np.arange(len(pred))
            valid_timesteps = timesteps[mask]
            
            if len(valid_timesteps) > 0:
                # Plot predictions and targets
                ax.plot(valid_timesteps, target[mask], 'b-', label='Target', linewidth=2)
                ax.plot(valid_timesteps, pred[mask], 'r--', label='Prediction', linewidth=2, alpha=0.8)
                
                # Add breakdown probability on secondary axis
                ax2 = ax.twinx()
                ax2.plot(timesteps, breakdown, 'g:', label='Breakdown Prob', alpha=0.6)
                ax2.set_ylabel('Breakdown Probability', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                ax2.set_ylim([0, 1.1])
                
                # Find breakdown point
                breakdown_idx = np.where(~mask)[0]
                if len(breakdown_idx) > 0:
                    breakdown_cycle = breakdown_idx[0]
                    ax.axvline(breakdown_cycle, color='red', linestyle=':', alpha=0.5, label='Breakdown')
                
            ax.set_xlabel('Cycles')
            ax.set_ylabel('Defect Count')
            ax.set_title(f'Sample {sample_idx}, V={voltage:.2f}V')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
        plt.suptitle(f'Predictions vs Targets - Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'predictions_epoch_{epoch:03d}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig


class Trainer:
    """Enhanced trainer with comprehensive metrics tracking and visualization."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Setup logging first
        self.setup_logging()
        
        # Setup metrics tracking
        self.metrics_dir = Path(config['training']['checkpoint_dir']) / 'metrics'
        self.metrics_tracker = MetricsTracker(self.metrics_dir)
        self.prediction_visualizer = PredictionVisualizer(self.metrics_dir / 'predictions')
        
        # Check if we're in a distributed environment
        self.is_distributed = self._check_distributed_env()
        
        if self.is_distributed:
            # Initialize distributed training
            if not DistributedManager.is_initialized():
                try:
                    DistributedManager.initialize()
                    self.dist_manager = DistributedManager()
                    self.device = self.dist_manager.device
                    self.logger.info(f"Initialized distributed training on rank {self.dist_manager.rank}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize distributed manager: {e}")
                    self.logger.info("Falling back to single GPU training")
                    self.is_distributed = False
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.dist_manager = None
        else:
            # Single GPU or CPU training
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dist_manager = None
            self.logger.info(f"Using single GPU/CPU training on device: {self.device}")
        
        # Create model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_path=config['data']['csv_path'],
            batch_size=config['training']['batch_size'],
            initial_cycles=config['data']['initial_cycles'],
            prediction_horizon=config['data']['prediction_horizon'],
            num_workers=config['data']['num_workers']
        )
        
        # Log dataset statistics
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Batch size: {config['training']['batch_size']}")
        self.logger.info(f"Training batches per epoch: {len(self.train_loader)}")
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create loss functions with adjusted weights to prevent NaN
        self.criterion = PhysicsInformedLoss(
            prediction_weight=1.0,
            breakdown_weight=0.3,  # Reduced from 0.5
            monotonic_weight=0.1,  # Reduced from 0.2
            smoothness_weight=0.05, # Reduced from 0.1
            physics_weight=0.1,    # Reduced from 0.3
            breakdown_threshold=config['model']['breakdown_threshold']
        )
        
        # Get temperature from physics config, with fallback to default
        temperature = config.get('physics', {}).get('temperature', 300.0)
        self.generation_loss = GenerationRateLoss(temperature=temperature)
        self.cycle_loss = CycleLoss()
        
        # Adjust loss weights
        self.generation_weight = 0.1  # Reduced from config value
        self.cycle_weight = 0.2       # Reduced from config value
        
        # Setup tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_breakdown_mae = float('inf')
        
        # Initialize LaunchLogger (handles both distributed and single GPU)
        LaunchLogger.initialize(
            use_wandb=config['training'].get('use_wandb', False),
            use_mlflow=False
        )
        
        # Setup tensorboard
        self.writer = SummaryWriter(config['training']['log_dir'])
        
        # Create rank zero logger wrapper
        if self.dist_manager:
            self.rank_zero_logger = RankZeroLoggingWrapper(self.logger, self.dist_manager)
        else:
            # For single GPU, just use regular logger
            self.rank_zero_logger = self.logger
            
    def _check_distributed_env(self):
        """Check if we're in a distributed environment."""
        # Check for SLURM environment variables
        if os.environ.get('SLURM_JOB_ID'):
            return os.environ.get('SLURM_NPROCS', '1') != '1'
        
        # Check for torchrun environment variables
        if os.environ.get('RANK') is not None:
            return True
            
        # Check for OpenMPI
        if os.environ.get('OMPI_COMM_WORLD_SIZE'):
            return int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1')) > 1
            
        return False
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = PythonLogger(__name__)
        
    def _create_model(self) -> PhysicsInformedFerroelectricSurrogate:
        """Create the surrogate model."""
        model_config = self.config['model']
        physics_config = self.config.get('physics', {
            'temperature': 300.0,
            'constants': {
                'k_B': 8.617e-5,
                'q': 1.602e-19
            },
            'device': {
                'temperature_default': 300.0
            }
        })
        
        model = PhysicsInformedFerroelectricSurrogate(
            trap_dim=model_config['trap_dim'],
            state_dim=model_config['state_dim'],
            latent_dim=model_config['latent_dim'],
            hidden_dim=model_config['hidden_dim'],
            encoder_layers=model_config['encoder_layers'],
            evolution_layers=model_config['evolution_layers'],
            breakdown_threshold=model_config['breakdown_threshold'],
            physics_config=physics_config
        )
        
        # Initialize model weights to prevent NaN
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
                        
        model.apply(init_weights)
        
        # Wrap with DDP only if truly distributed
        if self.is_distributed and self.dist_manager and self.dist_manager.distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.dist_manager.local_rank]
            )
            
        return model
        
    def _create_optimizer(self):
        """Create optimizer."""
        opt_config = self.config['optimizer']
        
        if opt_config['type'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.0),
                eps=1e-8  # Add epsilon for numerical stability
            )
        elif opt_config['type'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01),
                eps=1e-8  # Add epsilon for numerical stability
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
            
        return optimizer
        
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        sched_config = self.config['scheduler']
        
        if sched_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_config['type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        else:
            scheduler = None
            
        return scheduler
        
    def check_and_clip_gradients(self):
        """Check for NaN gradients and clip if necessary."""
        total_norm = 0.0
        has_nan = False
        
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    has_nan = True
                    p.grad.zero_()  # Zero out NaN gradients
                else:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    
        total_norm = total_norm ** 0.5
        
        if has_nan:
            self.logger.warning("NaN gradients detected and zeroed")
            
        # Always clip gradients
        if self.config['training'].get('gradient_clip_val'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip_val']
            )
            
        return total_norm, has_nan
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        import time
        epoch_start_time = time.time()
        
        with LaunchLogger(
            "train", 
            epoch=self.current_epoch, 
            num_mini_batch=len(self.train_loader),
            epoch_alert_freq=1
        ) as logger:
            
            # Initialize all possible loss keys
            epoch_losses = {
                'total': 0.0,
                'prediction': 0.0,
                'breakdown': 0.0,
                'monotonic': 0.0,
                'smoothness': 0.0,
                'physics': 0.0,
                'boundary': 0.0,
                'generation': 0.0,
                'cycle': 0.0
            }
            
            # Progress bar only on rank 0 (or always for single GPU)
            disable_progress = self.is_distributed and self.dist_manager and self.dist_manager.rank != 0
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.current_epoch}",
                disable=disable_progress
            )
            
            num_batches = 0
            num_valid_batches = 0
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Skip batch if all samples have early breakdown
                if batch['valid_mask'].sum() == 0:
                    self.logger.warning(f"Skipping batch {batch_idx} - no valid samples")
                    continue
                
                # Forward pass with gradient scaling for stability
                try:
                    outputs = self.model(
                        trap_params=batch['trap_parameters'],
                        voltage=batch['voltage'],
                        thickness=batch['thickness'],
                        pulsewidth=batch['pulsewidth'],
                        initial_states=batch['initial_states'],
                        target_length=batch['target_states'].size(1)
                    )
                    
                    # Check for NaN in outputs
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                            self.logger.warning(f"NaN detected in output {key}")
                            raise ValueError(f"NaN in {key}")
                            
                except Exception as e:
                    self.logger.error(f"Forward pass error in batch {batch_idx}: {e}")
                    continue
                
                # Calculate losses with try-except for each component
                try:
                    losses = self.criterion(
                        predictions=outputs['predictions'],
                        targets=batch['target_states'],
                        breakdown_prob=outputs['breakdown_prob'],
                        valid_mask=batch['valid_mask'],
                        voltage=batch['voltage'],
                        thickness=batch['thickness']
                    )
                    
                    # Check each loss component
                    for key, value in losses.items():
                        if torch.isnan(value):
                            self.logger.warning(f"NaN in {key} loss")
                            losses[key] = torch.tensor(0.0, device=self.device)
                    
                    # Additional physics losses with safety checks
                    gen_loss = torch.tensor(0.0, device=self.device)
                    try:
                        gen_loss = self.generation_loss(
                            predictions=outputs['predictions'],
                            voltage=batch['voltage'],
                            thickness=batch['thickness'],
                            trap_params=batch['trap_parameters'],
                            valid_mask=batch['valid_mask']
                        )
                        if torch.isnan(gen_loss):
                            gen_loss = torch.tensor(0.0, device=self.device)
                    except Exception as e:
                        self.logger.warning(f"Generation loss error: {e}")
                    
                    cycle_loss = torch.tensor(0.0, device=self.device)
                    try:
                        cycle_loss = self.cycle_loss(
                            breakdown_prob=outputs['breakdown_prob'],
                            breakdown_cycles=batch['breakdown_cycle'],
                            initial_cycles=self.config['data']['initial_cycles']
                        )
                        if torch.isnan(cycle_loss):
                            cycle_loss = torch.tensor(0.0, device=self.device)
                    except Exception as e:
                        self.logger.warning(f"Cycle loss error: {e}")
                    
                    # Combine all losses with reduced weights
                    total_loss = (
                        losses['total'] + 
                        self.generation_weight * gen_loss +
                        self.cycle_weight * cycle_loss
                    )
                    
                    # Final NaN check
                    if torch.isnan(total_loss):
                        self.logger.warning(f"NaN total loss in batch {batch_idx}, using prediction loss only")
                        total_loss = losses.get('prediction', torch.tensor(1.0, device=self.device))
                        if torch.isnan(total_loss):
                            self.logger.warning(f"Still NaN, skipping batch {batch_idx}")
                            continue
                            
                except Exception as e:
                    self.logger.error(f"Loss calculation error: {e}")
                    continue
                
                # Backward pass with gradient checking
                self.optimizer.zero_grad()
                
                try:
                    total_loss.backward()
                    
                    # Check and clip gradients
                    grad_norm, has_nan_grad = self.check_and_clip_gradients()
                    
                    if not has_nan_grad:
                        self.optimizer.step()
                        num_valid_batches += 1
                        
                        # Update metrics only for valid batches
                        epoch_losses['total'] += total_loss.item()
                        for key, value in losses.items():
                            if key != 'total' and not torch.isnan(value):
                                epoch_losses[key] += value.item()
                        epoch_losses['generation'] += gen_loss.item() if not torch.isnan(gen_loss) else 0.0
                        epoch_losses['cycle'] += cycle_loss.item() if not torch.isnan(cycle_loss) else 0.0
                        
                        # Log minibatch
                        logger.log_minibatch({
                            'loss': total_loss.item(),
                            'pred_loss': losses['prediction'].item() if not torch.isnan(losses['prediction']) else 0.0
                        })
                    else:
                        self.logger.warning(f"Skipping optimizer step due to NaN gradients in batch {batch_idx}")
                        
                except Exception as e:
                    self.logger.error(f"Backward pass error: {e}")
                    continue
                
                num_batches += 1
                
                # Update progress bar
                if num_valid_batches > 0 and batch_idx % 10 == 0:
                    avg_loss = epoch_losses['total'] / num_valid_batches
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'valid': f"{num_valid_batches}/{num_batches}"
                    })
                    
            # Average losses
            if num_valid_batches > 0:
                for key in epoch_losses:
                    epoch_losses[key] /= num_valid_batches
                self.logger.info(f"Epoch {self.current_epoch}: {num_valid_batches}/{num_batches} valid batches")
            else:
                self.logger.error("No valid batches in epoch!")
                # Set default values
                for key in epoch_losses:
                    epoch_losses[key] = 1.0
                    
        epoch_time = time.time() - epoch_start_time
        epoch_losses['epoch_time'] = epoch_time
        
        return epoch_losses
        
    @torch.no_grad()
    def validate(self):
        """Validate the model and save sample predictions."""
        self.model.eval()
        
        with LaunchLogger(
            "val",
            epoch=self.current_epoch,
            num_mini_batch=len(self.val_loader),
            epoch_alert_freq=1
        ) as logger:
            
            # Initialize all possible loss keys
            val_losses = {
                'total': 0.0,
                'prediction': 0.0,
                'breakdown': 0.0,
                'monotonic': 0.0,
                'smoothness': 0.0,
                'physics': 0.0,
                'boundary': 0.0,
                'generation': 0.0,
                'cycle': 0.0
            }
            
            # Additional metrics
            breakdown_errors = []
            final_defect_errors = []
            
            # Store samples for visualization
            sample_predictions = None
            
            disable_progress = self.is_distributed and self.dist_manager and self.dist_manager.rank != 0
            num_batches = 0
            num_valid_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", disable=disable_progress)):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Skip batch if all samples have early breakdown
                if batch['valid_mask'].sum() == 0:
                    continue
                
                # Forward pass
                try:
                    outputs = self.model(
                        trap_params=batch['trap_parameters'],
                        voltage=batch['voltage'],
                        thickness=batch['thickness'],
                        pulsewidth=batch['pulsewidth'],
                        initial_states=batch['initial_states'],
                        target_length=batch['target_states'].size(1)
                    )
                    
                    # Check for NaN
                    if torch.isnan(outputs['predictions']).any():
                        self.logger.warning(f"NaN predictions in validation batch {batch_idx}")
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Validation forward pass error: {e}")
                    continue
                
                # Calculate losses
                try:
                    losses = self.criterion(
                        predictions=outputs['predictions'],
                        targets=batch['target_states'],
                        breakdown_prob=outputs['breakdown_prob'],
                        valid_mask=batch['valid_mask'],
                        voltage=batch['voltage'],
                        thickness=batch['thickness']
                    )
                    
                    gen_loss = self.generation_loss(
                        predictions=outputs['predictions'],
                        voltage=batch['voltage'],
                        thickness=batch['thickness'],
                        trap_params=batch['trap_parameters'],
                        valid_mask=batch['valid_mask']
                    )
                    
                    cycle_loss = self.cycle_loss(
                        breakdown_prob=outputs['breakdown_prob'],
                        breakdown_cycles=batch['breakdown_cycle'],
                        initial_cycles=self.config['data']['initial_cycles']
                    )
                    
                    total_loss = (
                        losses['total'] + 
                        self.generation_weight * gen_loss +
                        self.cycle_weight * cycle_loss
                    )
                    
                    # Skip if NaN
                    if torch.isnan(total_loss):
                        continue
                        
                    # Log minibatch
                    logger.log_minibatch({
                        'loss': total_loss.item(),
                        'pred_loss': losses['prediction'].item()
                    })
                    
                    # Update losses
                    val_losses['total'] += total_loss.item()
                    for key, value in losses.items():
                        if key != 'total':
                            val_losses[key] += value.item()
                    val_losses['generation'] += gen_loss.item()
                    val_losses['cycle'] += cycle_loss.item()
                    
                    num_valid_batches += 1
                    
                    # Calculate additional metrics
                    predicted_breakdown = outputs['breakdown_prob'].argmax(dim=1) + self.config['data']['initial_cycles']
                    breakdown_error = torch.abs(predicted_breakdown - batch['breakdown_cycle']).float().mean()
                    breakdown_errors.append(breakdown_error.item())
                    
                    # Final defect count error
                    batch_size = batch['target_states'].size(0)
                    final_predictions = []
                    final_targets = []
                    
                    for i in range(batch_size):
                        valid_idx = batch['valid_mask'][i].nonzero()
                        if len(valid_idx) > 0:
                            last_idx = valid_idx[-1].item()
                            final_predictions.append(outputs['predictions'][i, last_idx])
                            final_targets.append(batch['target_states'][i, last_idx])
                            
                    if final_predictions:
                        final_predictions = torch.stack(final_predictions)
                        final_targets = torch.stack(final_targets)
                        final_error = torch.abs(final_predictions - final_targets).mean()
                        final_defect_errors.append(final_error.item())
                        
                    # Store first batch for visualization
                    if batch_idx == 0 and sample_predictions is None:
                        sample_predictions = {
                            'predictions': outputs['predictions'],
                            'targets': batch['target_states'],
                            'breakdown_prob': outputs['breakdown_prob'],
                            'valid_mask': batch['valid_mask'],
                            'voltages': batch['voltage']
                        }
                        
                except Exception as e:
                    self.logger.error(f"Validation loss calculation error: {e}")
                    continue
                    
                num_batches += 1
                    
            # Average losses
            if num_valid_batches > 0:
                for key in val_losses:
                    val_losses[key] /= num_valid_batches
                self.logger.info(f"Validation: {num_valid_batches}/{num_batches} valid batches")
            else:
                self.logger.error("No valid batches in validation!")
                for key in val_losses:
                    val_losses[key] = 1.0
                
            # Additional metrics
            val_losses['breakdown_mae'] = np.mean(breakdown_errors) if breakdown_errors else 0.0
            val_losses['final_defect_mae'] = np.mean(final_defect_errors) if final_defect_errors else 0.0
            
            # Log epoch-level metrics
            logger.log_epoch({
                'breakdown_mae': val_losses['breakdown_mae'],
                'final_defect_mae': val_losses['final_defect_mae']
            })
            
            # Visualize predictions if we have samples
            if sample_predictions is not None:
                try:
                    self.prediction_visualizer.visualize_predictions(
                        epoch=self.current_epoch,
                        predictions=sample_predictions['predictions'],
                        targets=sample_predictions['targets'],
                        breakdown_prob=sample_predictions['breakdown_prob'],
                        valid_mask=sample_predictions['valid_mask'],
                        voltages=sample_predictions['voltages']
                    )
                except Exception as e:
                    self.logger.error(f"Visualization error: {e}")
            
        return val_losses
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_breakdown_mae': self.best_breakdown_mae,
            'config': self.config,
            'metrics': self.metrics_tracker.metrics  # Save all tracked metrics
        }
        
        checkpoint_path = Path(self.config['training']['checkpoint_dir']) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        self.rank_zero_logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_breakdown_mae = checkpoint.get('best_breakdown_mae', float('inf'))
        
        # Load metrics history if available
        if 'metrics' in checkpoint:
            self.metrics_tracker.metrics = checkpoint['metrics']
        
        self.rank_zero_logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
    def train(self):
        """Main training loop."""
        self.rank_zero_logger.info("Starting training...")
        self.rank_zero_logger.info(f"Device: {self.device}")
        self.rank_zero_logger.info(f"Distributed: {self.is_distributed}")
        
        # Training summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.rank_zero_logger.info(f"Total parameters: {total_params:,}")
        self.rank_zero_logger.info(f"Trainable parameters: {trainable_params:,}")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                
            # Track metrics
            self.metrics_tracker.update_train_metrics(epoch, train_losses, train_losses.get('epoch_time', 0))
            self.metrics_tracker.update_val_metrics(epoch, val_losses)
            
            # Log metrics
            self.log_metrics(train_losses, val_losses)
            
            # Save checkpoint
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pt')
                
            if val_losses['breakdown_mae'] < self.best_breakdown_mae:
                self.best_breakdown_mae = val_losses['breakdown_mae']
                self.save_checkpoint('best_breakdown_model.pt')
                
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_checkpoint(f'model_epoch_{epoch + 1}.pt')
                
            # Save plots every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.metrics_tracker.plot_losses()
                self.metrics_tracker.plot_validation_metrics()
                self.metrics_tracker.save_metrics()
                
        # Final plots and metrics save
        self.metrics_tracker.plot_losses()
        self.metrics_tracker.plot_validation_metrics()
        self.metrics_tracker.save_metrics()
        
        self.rank_zero_logger.info("Training completed!")
        self.rank_zero_logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.rank_zero_logger.info(f"Best breakdown MAE: {self.best_breakdown_mae:.2f} cycles")
        
    def log_metrics(self, train_losses: Dict, val_losses: Dict):
        """Log metrics to tensorboard and console."""
        # Console logging
        self.rank_zero_logger.info(
            f"Epoch {self.current_epoch}: "
            f"Train Loss = {train_losses['total']:.4f}, "
            f"Val Loss = {val_losses['total']:.4f}, "
            f"Val Breakdown MAE = {val_losses['breakdown_mae']:.2f} cycles, "
            f"Time = {train_losses.get('epoch_time', 0):.2f}s"
        )
        
        # Tensorboard logging (always write in single GPU, only rank 0 in distributed)
        should_write = not self.is_distributed or (self.dist_manager and self.dist_manager.rank == 0)
        if should_write:
            for key, value in train_losses.items():
                if key != 'epoch_time':
                    self.writer.add_scalar(f'train/{key}', value, self.current_epoch)
                
            for key, value in val_losses.items():
                self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
                
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/lr', current_lr, self.current_epoch)


def main():
    """Main training function."""
    # Load config with full path
    with open('/storage/home/hcoda1/6/cli872/scratch/work/SDG/config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Create trainer
    trainer = Trainer(config)
    
    # Load checkpoint if resuming
    if config['training'].get('resume_from'):
        trainer.load_checkpoint(config['training']['resume_from'])
        
    # Train
    trainer.train()
    
    # Cleanup
    if hasattr(trainer, 'writer'):
        trainer.writer.close()
        
    # Clean up distributed if needed
    if trainer.is_distributed and trainer.dist_manager:
        DistributedManager.cleanup()


if __name__ == '__main__':
    main()