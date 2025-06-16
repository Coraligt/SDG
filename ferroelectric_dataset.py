# data/datapipes/ferroelectric_dataset.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class FerroelectricDataset(Dataset):
    """Dataset for ferroelectric device defect evolution.
    
    This dataset handles the kinetic Monte Carlo simulation data where each sample
    represents a device with initial trap parameters and voltage conditions,
    and the evolution of defect states over cycles.
    """
    
    def __init__(
        self,
        data_path: str,
        initial_cycles: int = 5,
        prediction_horizon: int = 50,
        max_cycles: int = 2000,
        breakdown_threshold: int = 200,
        normalize: bool = True,
        train: bool = True,
        train_split: float = 0.8,
        seed: int = 42,
    ):
        """
        Args:
            data_path: Path to CSV file
            initial_cycles: Number of initial cycles to use as input (K)
            prediction_horizon: Number of future cycles to predict (M)
            max_cycles: Maximum number of cycles in data
            breakdown_threshold: Defect count threshold for breakdown
            normalize: Whether to normalize the data
            train: Whether this is training or validation set
            train_split: Fraction of data for training
            seed: Random seed for train/val split
        """
        self.initial_cycles = initial_cycles
        self.prediction_horizon = prediction_horizon
        self.max_cycles = max_cycles
        self.breakdown_threshold = breakdown_threshold
        self.normalize = normalize
        
        # Load and process data
        self.data = self._load_data(data_path)
        
        # Split train/val
        np.random.seed(seed)
        n_samples = len(self.data)
        indices = np.random.permutation(n_samples)
        n_train = int(train_split * n_samples)
        
        if train:
            self.indices = indices[:n_train]
        else:
            self.indices = indices[n_train:]
            
        logger.info(f"Dataset initialized with {len(self.indices)} samples ({'train' if train else 'val'})")
        
        # Calculate normalization statistics from training data
        if train and normalize:
            self._calculate_normalization_stats()
            
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate CSV data."""
        df = pd.read_csv(data_path, header=None)
        
        # Define column names
        col_names = [
            "peak_density",
            "thermal_ionization_mean",
            "thermal_ionization_spread",
            "relaxation_energy",
            "voltage1",
            "voltage2", 
            "pulsewidth",
            "thickness"
        ]
        
        # Add cycle columns
        for i in range(1, self.max_cycles + 1):
            col_names.append(f"cycle_{i}")
            
        df.columns = col_names
        
        # Add fixed trap parameters not in CSV
        df['electron_affinity'] = 2.4  # eV
        df['work_function_te'] = 4.6   # eV
        df['work_function_be'] = 4.6   # eV
        df['bandgap'] = 5.8            # eV
        
        return df
    
    def _calculate_normalization_stats(self):
        """Calculate mean and std for normalization."""
        # Collect all trap parameters
        trap_params = []
        voltages = []
        cycle_values = []
        
        for idx in self.indices:
            sample = self._get_raw_sample(idx)
            trap_params.append(sample['trap_parameters'])
            voltages.append(sample['voltage'])
            
            # Collect non-NaN cycle values
            cycles = sample['defect_evolution']
            valid_cycles = cycles[~np.isnan(cycles)]
            cycle_values.extend(valid_cycles)
            
        trap_params = np.array(trap_params)
        voltages = np.array(voltages)
        cycle_values = np.array(cycle_values)
        
        # Calculate statistics
        self.trap_mean = np.mean(trap_params, axis=0)
        self.trap_std = np.std(trap_params, axis=0) + 1e-8
        
        self.voltage_mean = np.mean(voltages)
        self.voltage_std = np.std(voltages) + 1e-8
        
        self.cycle_mean = np.mean(cycle_values)
        self.cycle_std = np.std(cycle_values) + 1e-8
        
        # Handle thickness and pulsewidth
        thicknesses = self.data.iloc[self.indices]['thickness'].values
        pulsewidths = self.data.iloc[self.indices]['pulsewidth'].values
        
        self.thickness_mean = np.mean(thicknesses)
        self.thickness_std = np.std(thicknesses) + 1e-8
        
        self.pulsewidth_mean = np.mean(pulsewidths) 
        self.pulsewidth_std = np.std(pulsewidths) + 1e-8
        
    def _get_raw_sample(self, idx: int) -> Dict:
        """Get raw sample data."""
        row = self.data.iloc[self.indices[idx]]
        
        # Extract trap parameters (8 parameters)
        trap_parameters = np.array([
            row['peak_density'],
            row['thermal_ionization_mean'],
            row['thermal_ionization_spread'],
            row['relaxation_energy'],
            row['electron_affinity'],
            row['work_function_te'],
            row['work_function_be'],
            row['bandgap']
        ], dtype=np.float32)
        
        # Extract device parameters
        voltage = row['voltage1']  # Use voltage1 as the stress voltage
        thickness = row['thickness']
        pulsewidth = row['pulsewidth']
        
        # Extract defect evolution
        cycle_cols = [f'cycle_{i}' for i in range(1, self.max_cycles + 1)]
        defect_evolution = row[cycle_cols].values.astype(np.float32)
        
        return {
            'trap_parameters': trap_parameters,
            'voltage': voltage,
            'thickness': thickness,
            'pulsewidth': pulsewidth,
            'defect_evolution': defect_evolution
        }
    
    def _find_breakdown_cycle(self, defect_evolution: np.ndarray) -> int:
        """Find the cycle where breakdown occurs.
        
        Breakdown is indicated by NaN values in the data. When defect count
        exceeds 200 in cycle N, cycle N+1 and all subsequent cycles are NaN.
        
        Returns:
            The cycle number where breakdown occurred (last valid cycle + 1)
        """
        # Find first NaN - this indicates breakdown happened in previous cycle
        nan_mask = np.isnan(defect_evolution)
        if np.any(nan_mask):
            first_nan = np.argmax(nan_mask)
            # Breakdown occurred at the cycle before first NaN
            return first_nan
        
        # If no NaN, find if any value exceeds threshold
        exceeds_threshold = defect_evolution >= self.breakdown_threshold
        if np.any(exceeds_threshold):
            # Breakdown should occur after first exceedance
            return np.argmax(exceeds_threshold) + 1
            
        # No breakdown detected, return last cycle
        return len(defect_evolution)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.
        
        Returns:
            Dictionary containing:
                - trap_parameters: [8] trap parameter values
                - voltage: scalar voltage value
                - thickness: scalar thickness value
                - pulsewidth: scalar pulsewidth value
                - initial_states: [initial_cycles] initial defect counts
                - target_states: [prediction_horizon] target defect counts
                - valid_mask: [prediction_horizon] mask for valid timesteps
                - breakdown_cycle: scalar, cycle where breakdown occurs
        """
        sample = self._get_raw_sample(idx)
        
        # Find breakdown cycle
        breakdown_cycle = self._find_breakdown_cycle(sample['defect_evolution'])
        
        # Extract sequences with proper breakdown handling
        initial_states = sample['defect_evolution'][:self.initial_cycles]
        
        # Validate initial states don't contain NaN
        if np.isnan(initial_states).any():
            raise ValueError(f"Sample {idx}: Initial states contain NaN, which indicates data issue")
        
        # For target states, extract the next prediction_horizon cycles
        start_idx = self.initial_cycles
        end_idx = min(start_idx + self.prediction_horizon, len(sample['defect_evolution']))
        target_states = sample['defect_evolution'][start_idx:end_idx]
        
        # Pad if necessary (when sequence is shorter than prediction horizon)
        if len(target_states) < self.prediction_horizon:
            pad_length = self.prediction_horizon - len(target_states)
            # Pad with NaN to indicate no data available
            target_states = np.pad(target_states, (0, pad_length), 
                                 mode='constant', constant_values=np.nan)
            
        # Create valid mask - True where we have actual data (not NaN)
        valid_mask = ~np.isnan(target_states)
        
        # For normalized targets, we need to handle NaN carefully
        # Only normalize the valid values
        if self.normalize and hasattr(self, 'cycle_mean'):
            # Normalize only non-NaN values
            valid_indices = ~np.isnan(target_states)
            if valid_indices.any():
                target_states_norm = target_states.copy()
                target_states_norm[valid_indices] = (
                    (target_states[valid_indices] - self.cycle_mean) / self.cycle_std
                )
            else:
                target_states_norm = target_states
                
            initial_states_norm = (initial_states - self.cycle_mean) / self.cycle_std
        else:
            target_states_norm = target_states
            initial_states_norm = initial_states
        # Normalize if required
        if self.normalize and hasattr(self, 'trap_mean'):
            trap_params_norm = (sample['trap_parameters'] - self.trap_mean) / self.trap_std
            voltage_norm = (sample['voltage'] - self.voltage_mean) / self.voltage_std
            thickness_norm = (sample['thickness'] - self.thickness_mean) / self.thickness_std
            pulsewidth_norm = (sample['pulsewidth'] - self.pulsewidth_mean) / self.pulsewidth_std
        else:
            trap_params_norm = sample['trap_parameters']
            voltage_norm = sample['voltage']
            thickness_norm = sample['thickness']
            pulsewidth_norm = sample['pulsewidth']
            
        return {
            'trap_parameters': torch.tensor(trap_params_norm, dtype=torch.float32),
            'voltage': torch.tensor(voltage_norm, dtype=torch.float32),
            'thickness': torch.tensor(thickness_norm, dtype=torch.float32),
            'pulsewidth': torch.tensor(pulsewidth_norm, dtype=torch.float32),
            'initial_states': torch.tensor(initial_states_norm, dtype=torch.float32),
            'target_states': torch.tensor(target_states_norm, dtype=torch.float32),  # May contain NaN
            'valid_mask': torch.tensor(valid_mask, dtype=torch.float32),
            'breakdown_cycle': torch.tensor(breakdown_cycle, dtype=torch.long),
        }


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    initial_cycles: int = 5,
    prediction_horizon: int = 50,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_dataset = FerroelectricDataset(
        data_path=data_path,
        initial_cycles=initial_cycles,
        prediction_horizon=prediction_horizon,
        train=True,
        **dataset_kwargs
    )
    
    val_dataset = FerroelectricDataset(
        data_path=data_path,
        initial_cycles=initial_cycles,
        prediction_horizon=prediction_horizon,
        train=False,
        normalize=True,
        **dataset_kwargs
    )
    
    # Copy normalization stats from train to val
    if hasattr(train_dataset, 'trap_mean'):
        val_dataset.trap_mean = train_dataset.trap_mean
        val_dataset.trap_std = train_dataset.trap_std
        val_dataset.voltage_mean = train_dataset.voltage_mean
        val_dataset.voltage_std = train_dataset.voltage_std
        val_dataset.cycle_mean = train_dataset.cycle_mean
        val_dataset.cycle_std = train_dataset.cycle_std
        val_dataset.thickness_mean = train_dataset.thickness_mean
        val_dataset.thickness_std = train_dataset.thickness_std
        val_dataset.pulsewidth_mean = train_dataset.pulsewidth_mean
        val_dataset.pulsewidth_std = train_dataset.pulsewidth_std
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader