# inference/generate.py

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import json
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the project root to path
import sys
sys.path.append('/storage/home/hcoda1/6/cli872/scratch/work/SDG')

from models.fe_surrogate import PhysicsInformedFerroelectricSurrogate
from ferroelectric_dataset import FerroelectricDataset


class SyntheticDataGenerator:
    """Generate synthetic ferroelectric degradation data using trained surrogate."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        normalize: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.normalize = normalize
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load checkpoint
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Create model
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load normalization stats if needed
        if normalize:
            self._load_normalization_stats()
            
        self.logger.info("Model loaded successfully")
        
    def _create_model(self) -> PhysicsInformedFerroelectricSurrogate:
        """Create model from config."""
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
        
        return PhysicsInformedFerroelectricSurrogate(
            trap_dim=model_config['trap_dim'],
            state_dim=model_config['state_dim'],
            latent_dim=model_config['latent_dim'],
            hidden_dim=model_config['hidden_dim'],
            encoder_layers=model_config['encoder_layers'],
            evolution_layers=model_config['evolution_layers'],
            breakdown_threshold=model_config['breakdown_threshold'],
            physics_config=physics_config
        )
        
    def _load_normalization_stats(self):
        """Load normalization statistics from training data."""
        # Create a dummy dataset to get normalization stats
        dataset = FerroelectricDataset(
            data_path=self.config['data']['csv_path'],
            train=True,
            normalize=True
        )
        
        self.trap_mean = torch.tensor(dataset.trap_mean, device=self.device, dtype=torch.float32)
        self.trap_std = torch.tensor(dataset.trap_std, device=self.device, dtype=torch.float32)
        self.voltage_mean = dataset.voltage_mean
        self.voltage_std = dataset.voltage_std
        self.cycle_mean = dataset.cycle_mean
        self.cycle_std = dataset.cycle_std
        self.thickness_mean = dataset.thickness_mean
        self.thickness_std = dataset.thickness_std
        self.pulsewidth_mean = dataset.pulsewidth_mean
        self.pulsewidth_std = dataset.pulsewidth_std
        
    def generate_samples(
        self,
        num_samples: int,
        voltage_range: Tuple[float, float] = (-3.6, 3.6),
        thickness: float = 6e-9,
        pulsewidth: float = 2e-7,
        initial_cycles: int = 5,
        max_cycles: int = 2000,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic samples with random trap parameters.
        
        Args:
            num_samples: Number of samples to generate
            voltage_range: Range of voltages to sample from
            thickness: Film thickness (m)
            pulsewidth: Pulse width (s)
            initial_cycles: Number of initial cycles to simulate
            max_cycles: Maximum cycles to generate
            seed: Random seed
            
        Returns:
            Dictionary containing generated data
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.logger.info(f"Generating {num_samples} synthetic samples...")
        
        # Generate random trap parameters
        trap_params = self._generate_trap_parameters(num_samples)
        
        # Generate random voltages
        voltages = np.random.uniform(voltage_range[0], voltage_range[1], num_samples)
        
        # Generate initial states based on physics
        initial_states = self._generate_initial_states(
            num_samples, initial_cycles, trap_params, voltages, thickness
        )
        
        # Convert to tensors
        trap_params_tensor = torch.tensor(trap_params, dtype=torch.float32, device=self.device)
        voltages_tensor = torch.tensor(voltages, dtype=torch.float32, device=self.device)
        thickness_tensor = torch.full((num_samples,), thickness, device=self.device, dtype=torch.float32)
        pulsewidth_tensor = torch.full((num_samples,), pulsewidth, device=self.device, dtype=torch.float32)
        initial_states_tensor = torch.tensor(initial_states, dtype=torch.float32, device=self.device)
        
        # Normalize if needed
        if self.normalize:
            # Ensure std is not too small
            trap_std_safe = torch.clamp(self.trap_std, min=1e-6)
            voltage_std_safe = max(self.voltage_std, 0.1)
            thickness_std_safe = max(self.thickness_std, 1e-10)
            pulsewidth_std_safe = max(self.pulsewidth_std, 1e-9)
            cycle_std_safe = max(self.cycle_std, 1.0)
            
            trap_params_tensor = (trap_params_tensor - self.trap_mean) / trap_std_safe
            voltages_tensor = (voltages_tensor - self.voltage_mean) / voltage_std_safe
            thickness_tensor = (thickness_tensor - self.thickness_mean) / thickness_std_safe
            pulsewidth_tensor = (pulsewidth_tensor - self.pulsewidth_mean) / pulsewidth_std_safe
            initial_states_tensor = (initial_states_tensor - self.cycle_mean) / cycle_std_safe
            
        # Generate trajectories in batches to handle memory
        batch_size = 16
        all_trajectories = []
        all_breakdown_cycles = []
        all_final_defects = []
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_slice = slice(i, end_idx)
            
            self.logger.info(f"Generating batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
            
            with torch.no_grad():
                try:
                    results = self.model.generate(
                        trap_params=trap_params_tensor[batch_slice],
                        voltage=voltages_tensor[batch_slice],
                        thickness=thickness_tensor[batch_slice],
                        pulsewidth=pulsewidth_tensor[batch_slice],
                        initial_states=initial_states_tensor[batch_slice],
                        max_cycles=max_cycles - initial_cycles
                    )
                    
                    # Denormalize results if needed
                    if self.normalize:
                        results['full_trajectory'] = (
                            results['full_trajectory'] * self.cycle_std + self.cycle_mean
                        )
                        results['final_defect_counts'] = (
                            results['final_defect_counts'] * self.cycle_std + self.cycle_mean
                        )
                    
                    all_trajectories.append(results['full_trajectory'].cpu())
                    all_breakdown_cycles.append(results['breakdown_cycles'].cpu())
                    all_final_defects.append(results['final_defect_counts'].cpu())
                    
                except Exception as e:
                    self.logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                    # Create dummy data for failed batch
                    batch_len = end_idx - i
                    dummy_traj = torch.zeros(batch_len, max_cycles)
                    dummy_breakdown = torch.full((batch_len,), max_cycles)
                    dummy_final = torch.full((batch_len,), 200.0)
                    
                    all_trajectories.append(dummy_traj)
                    all_breakdown_cycles.append(dummy_breakdown)
                    all_final_defects.append(dummy_final)
        
        # Combine all batches
        full_trajectories = torch.cat(all_trajectories, dim=0).numpy()
        breakdown_cycles = torch.cat(all_breakdown_cycles, dim=0).numpy()
        final_defect_counts = torch.cat(all_final_defects, dim=0).numpy()
        
        # Ensure valid data
        # Clip values to reasonable ranges
        full_trajectories = np.clip(full_trajectories, 0, 300)
        final_defect_counts = np.clip(final_defect_counts, 0, 300)
        
        # Handle any remaining NaN values
        nan_mask = np.isnan(final_defect_counts)
        if nan_mask.any():
            self.logger.warning(f"Found {nan_mask.sum()} NaN values in final defect counts, replacing with mean")
            mean_value = np.nanmean(final_defect_counts)
            if np.isnan(mean_value):
                mean_value = 100.0  # Default fallback
            final_defect_counts[nan_mask] = mean_value
        
        return {
            'trap_parameters': trap_params,
            'voltages': voltages,
            'thickness': np.full(num_samples, thickness),
            'pulsewidth': np.full(num_samples, pulsewidth),
            'trajectories': full_trajectories,
            'breakdown_cycles': breakdown_cycles,
            'final_defect_counts': final_defect_counts
        }
        
    def _generate_trap_parameters(self, num_samples: int) -> np.ndarray:
        """Generate realistic trap parameters based on physical ranges."""
        # Define parameter ranges based on the physics config and papers
        param_ranges = {
            'peak_density': (1e19, 5e19),  # cm^-3
            'thermal_ionization_mean': (2.0, 3.0),  # eV
            'thermal_ionization_spread': (0.5, 1.5),  # eV
            'relaxation_energy': (1.0, 1.5),  # eV
            'electron_affinity': (2.4, 2.4),  # Fixed
            'work_function_te': (4.6, 4.6),  # Fixed
            'work_function_be': (4.6, 4.6),  # Fixed
            'bandgap': (5.8, 5.8)  # Fixed
        }
        
        trap_params = np.zeros((num_samples, 8))
        
        for i, (param, (low, high)) in enumerate(param_ranges.items()):
            if low == high:
                trap_params[:, i] = low
            else:
                trap_params[:, i] = np.random.uniform(low, high, num_samples)
                
        return trap_params
        
    def _generate_initial_states(
        self,
        num_samples: int,
        initial_cycles: int,
        trap_params: np.ndarray,
        voltages: np.ndarray,
        thickness: float
    ) -> np.ndarray:
        """Generate realistic initial states based on physics."""
        initial_states = np.zeros((num_samples, initial_cycles))
        
        for i in range(num_samples):
            # Calculate electric field
            e_field = np.abs(voltages[i]) / thickness  # V/m
            
            # Initial defect growth proportional to field and trap density
            # Using simplified thermochemical model
            field_factor = np.exp(0.1 * e_field / 1e9)  # Normalize and scale
            density_factor = trap_params[i, 0] / 1e19  # Normalize peak density
            
            base_rate = 0.5 * field_factor * density_factor
            
            for j in range(initial_cycles):
                if j == 0:
                    # Start with some initial defects
                    initial_states[i, j] = np.random.poisson(10)
                else:
                    # Physics-based growth model with some randomness
                    growth = np.random.poisson(base_rate * (j + 1))
                    initial_states[i, j] = initial_states[i, j-1] + growth
                    
                    # Cap at breakdown threshold
                    if initial_states[i, j] > 200:
                        initial_states[i, j] = 200
                        # Fill remaining cycles with breakdown value
                        initial_states[i, j:] = 200
                        break
                    
        return initial_states
        
    def save_dataset(
        self,
        data: Dict[str, np.ndarray],
        output_dir: str,
        format: str = 'hdf5'
    ):
        """Save generated dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format == 'hdf5':
            # Save as HDF5
            with h5py.File(output_path / 'synthetic_data.h5', 'w') as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value, compression='gzip')
                    
            # Save metadata
            metadata = {
                'num_samples': len(data['voltages']),
                'generation_config': {
                    'model_checkpoint': str(self.config['training']['checkpoint_dir']),
                    'breakdown_threshold': self.config['model']['breakdown_threshold'],
                    'physics_model': 'kinetic_monte_carlo',
                    'generation_model': 'thermochemical'
                },
                'statistics': {
                    'avg_breakdown_cycle': float(np.mean(data['breakdown_cycles'])),
                    'std_breakdown_cycle': float(np.std(data['breakdown_cycles'])),
                    'avg_final_defects': float(np.mean(data['final_defect_counts'])),
                    'std_final_defects': float(np.std(data['final_defect_counts']))
                }
            }
            
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
        elif format == 'csv':
            # Convert to CSV format similar to original
            num_samples = len(data['voltages'])
            max_cycles = data['trajectories'].shape[1]
            
            # Create dataframe
            df_data = []
            
            for i in range(num_samples):
                row = []
                # Add trap parameters (only first 4, others are fixed)
                row.extend(data['trap_parameters'][i, :4])
                # Add voltage
                row.append(data['voltages'][i])
                row.append(-data['voltages'][i])  # Voltage 2 (symmetric)
                # Add pulsewidth and thickness
                row.append(data['pulsewidth'][i])
                row.append(data['thickness'][i])
                # Add trajectory
                trajectory = data['trajectories'][i]
                # Pad with NaN after breakdown
                breakdown_idx = int(data['breakdown_cycles'][i])
                if breakdown_idx < max_cycles:
                    trajectory[breakdown_idx:] = np.nan
                row.extend(trajectory)
                
                df_data.append(row)
                
            df = pd.DataFrame(df_data)
            df.to_csv(output_path / 'synthetic_data.csv', index=False, header=False)
            
        self.logger.info(f"Saved {len(data['voltages'])} samples to {output_path}")
        
    def visualize_samples(
        self,
        data: Dict[str, np.ndarray],
        num_samples: int = 5,
        save_path: Optional[str] = None
    ):
        """Visualize generated samples."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Sample trajectories
        ax = axes[0, 0]
        num_to_plot = min(num_samples, len(data['trajectories']))
        
        for i in range(num_to_plot):
            trajectory = data['trajectories'][i]
            breakdown_cycle = int(data['breakdown_cycles'][i])
            
            # Plot until breakdown
            cycles = np.arange(len(trajectory))
            valid_mask = cycles < breakdown_cycle
            
            if valid_mask.any():
                ax.plot(cycles[valid_mask], trajectory[valid_mask], 
                       label=f"V={data['voltages'][i]:.1f}V", alpha=0.7)
                if breakdown_cycle > 0 and breakdown_cycle < len(trajectory):
                    ax.scatter(breakdown_cycle-1, trajectory[min(breakdown_cycle-1, len(trajectory)-1)], 
                              marker='x', s=100, c='red')
            
        ax.set_xlabel('Cycles')
        ax.set_ylabel('Defect Count')
        ax.set_title('Sample Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Breakdown cycles vs voltage
        ax = axes[0, 1]
        ax.scatter(data['voltages'], data['breakdown_cycles'], alpha=0.6)
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Breakdown Cycle')
        ax.set_title('Voltage Dependence of Breakdown')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Final defect count distribution
        ax = axes[1, 0]
        # Filter out any extreme values for visualization
        valid_final = data['final_defect_counts'][data['final_defect_counts'] < 500]
        if len(valid_final) > 0:
            ax.hist(valid_final, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(200, color='red', linestyle='--', label='Threshold')
            ax.set_xlabel('Final Defect Count')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Final Defect Counts')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Trap parameter correlation
        ax = axes[1, 1]
        ax.scatter(data['trap_parameters'][:, 0] / 1e19, 
                  data['breakdown_cycles'], alpha=0.6)
        ax.set_xlabel('Peak Density (×10¹⁹ cm⁻³)')
        ax.set_ylabel('Breakdown Cycle')
        ax.set_title('Effect of Initial Trap Density')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
            
        plt.close()


def main():
    """Main generation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic ferroelectric data')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='synthetic_data',
                       help='Output directory')
    parser.add_argument('--voltage_min', type=float, default=-3.6,
                       help='Minimum voltage')
    parser.add_argument('--voltage_max', type=float, default=3.6,
                       help='Maximum voltage')
    parser.add_argument('--format', type=str, default='hdf5',
                       choices=['hdf5', 'csv'],
                       help='Output format')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticDataGenerator(args.checkpoint)
    
    # Generate samples
    data = generator.generate_samples(
        num_samples=args.num_samples,
        voltage_range=(args.voltage_min, args.voltage_max),
        seed=args.seed
    )
    
    # Save dataset
    generator.save_dataset(data, args.output_dir, format=args.format)
    
    # Visualize if requested
    if args.visualize:
        viz_path = Path(args.output_dir) / 'sample_visualization.png'
        generator.visualize_samples(data, save_path=str(viz_path))
        
    print(f"\nGeneration complete!")
    print(f"Generated {args.num_samples} samples")
    print(f"Average breakdown cycle: {data['breakdown_cycles'].mean():.1f} ± {data['breakdown_cycles'].std():.1f}")
    print(f"Average final defect count: {data['final_defect_counts'].mean():.1f} ± {data['final_defect_counts'].std():.1f}")
    print(f"Saved to {args.output_dir}")


if __name__ == '__main__':
    main()