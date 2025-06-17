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
from typing import Dict, Optional
import os
import wandb

# PhysicsNeMo imports 
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import LaunchLogger, PythonLogger, RankZeroLoggingWrapper

import sys
sys.path.append('/storage/home/hcoda1/6/cli872/scratch/work/SDG')

from ferroelectric_dataset import create_dataloaders
from models.fe_surrogate import PhysicsInformedFerroelectricSurrogate
from models.losses import PhysicsInformedLoss, GenerationRateLoss, CycleLoss


class Trainer:
    """Trainer for ferroelectric surrogate model."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Setup logging first
        self.setup_logging()
        
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
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create loss functions
        self.criterion = PhysicsInformedLoss(
            **config['loss']['physics_informed']
        )
        
        # Get temperature from physics config, with fallback to default
        temperature = config.get('physics', {}).get('temperature', 300.0)
        self.generation_loss = GenerationRateLoss(
            temperature=temperature
        )
        self.cycle_loss = CycleLoss()
        
        # Setup tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
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
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        elif opt_config['type'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01)
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
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        with LaunchLogger(
            "train", 
            epoch=self.current_epoch, 
            num_mini_batch=len(self.train_loader),
            epoch_alert_freq=1
        ) as logger:
            
            epoch_losses = {
                'total': 0.0,
                'prediction': 0.0,
                'breakdown': 0.0,
                'monotonic': 0.0,
                'smoothness': 0.0,
                'physics': 0.0,
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
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    trap_params=batch['trap_parameters'],
                    voltage=batch['voltage'],
                    thickness=batch['thickness'],
                    pulsewidth=batch['pulsewidth'],
                    initial_states=batch['initial_states'],
                    target_length=batch['target_states'].size(1)
                )
                
                # Calculate losses
                losses = self.criterion(
                    predictions=outputs['predictions'],
                    targets=batch['target_states'],
                    breakdown_prob=outputs['breakdown_prob'],
                    valid_mask=batch['valid_mask'],
                    voltage=batch['voltage'],
                    thickness=batch['thickness']
                )
                
                # Additional physics losses
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
                
                # Combine all losses
                total_loss = (
                    losses['total'] + 
                    self.config['loss']['generation_weight'] * gen_loss +
                    self.config['loss']['cycle_weight'] * cycle_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip_val'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_val']
                    )
                    
                self.optimizer.step()
                
                # Update metrics
                epoch_losses['total'] += total_loss.item()
                for key, value in losses.items():
                    if key != 'total':
                        # epoch_losses[key] += value.item()
                        epoch_losses[key] = epoch_losses.get(key, 0.0) + value.item()
                epoch_losses['generation'] += gen_loss.item()
                epoch_losses['cycle'] += cycle_loss.item()
                
                # Log minibatch
                logger.log_minibatch({
                    'loss': total_loss.item(),
                    'pred_loss': losses['prediction'].item()
                })
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({
                        'loss': f"{total_loss.item():.4f}",
                        'pred': f"{losses['prediction'].item():.4f}"
                    })
                    
            # Average losses
            num_batches = len(self.train_loader)
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
                
        return epoch_losses
        
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        with LaunchLogger(
            "val",
            epoch=self.current_epoch,
            num_mini_batch=len(self.val_loader),
            epoch_alert_freq=1
        ) as logger:
            
            val_losses = {
                'total': 0.0,
                'prediction': 0.0,
                'breakdown': 0.0,
                'monotonic': 0.0,
                'smoothness': 0.0,
                'physics': 0.0,
                'generation': 0.0,
                'cycle': 0.0
            }
            
            # Additional metrics
            breakdown_errors = []
            final_defect_errors = []
            
            disable_progress = self.is_distributed and self.dist_manager and self.dist_manager.rank != 0
            for batch in tqdm(self.val_loader, desc="Validation", disable=disable_progress):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    trap_params=batch['trap_parameters'],
                    voltage=batch['voltage'],
                    thickness=batch['thickness'],
                    pulsewidth=batch['pulsewidth'],
                    initial_states=batch['initial_states'],
                    target_length=batch['target_states'].size(1)
                )
                
                # Calculate losses
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
                    self.config['loss']['generation_weight'] * gen_loss +
                    self.config['loss']['cycle_weight'] * cycle_loss
                )
                
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
                    
            # Average losses
            num_batches = len(self.val_loader)
            for key in val_losses:
                val_losses[key] /= num_batches
                
            # Additional metrics
            val_losses['breakdown_mae'] = np.mean(breakdown_errors) if breakdown_errors else 0.0
            val_losses['final_defect_mae'] = np.mean(final_defect_errors) if final_defect_errors else 0.0
            
            # Log epoch-level metrics
            logger.log_epoch({
                'breakdown_mae': val_losses['breakdown_mae'],
                'final_defect_mae': val_losses['final_defect_mae']
            })
            
        return val_losses
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
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
        
        self.rank_zero_logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
    def train(self):
        """Main training loop."""
        self.rank_zero_logger.info("Starting training...")
        self.rank_zero_logger.info(f"Device: {self.device}")
        self.rank_zero_logger.info(f"Distributed: {self.is_distributed}")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                
            # Log metrics
            self.log_metrics(train_losses, val_losses)
            
            # Save checkpoint
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pt')
                
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_checkpoint(f'model_epoch_{epoch + 1}.pt')
                
        self.rank_zero_logger.info("Training completed!")
        
    def log_metrics(self, train_losses: Dict, val_losses: Dict):
        """Log metrics to tensorboard and console."""
        # Console logging
        self.rank_zero_logger.info(
            f"Epoch {self.current_epoch}: "
            f"Train Loss = {train_losses['total']:.4f}, "
            f"Val Loss = {val_losses['total']:.4f}, "
            f"Val Breakdown MAE = {val_losses['breakdown_mae']:.2f} cycles"
        )
        
        # Tensorboard logging (always write in single GPU, only rank 0 in distributed)
        should_write = not self.is_distributed or (self.dist_manager and self.dist_manager.rank == 0)
        if should_write:
            for key, value in train_losses.items():
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