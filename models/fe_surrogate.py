# models/fe_surrogate.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List  # Make sure Dict is imported!
import numpy as np
import math

# PhysicsNeMo imports - only use what actually exists
from physicsnemo.models.mlp import FullyConnected

# Import our physics module
import sys
sys.path.append('/storage/home/hcoda1/6/cli872/scratch/work/SDG')
from physics.kmc_physics import KineticMonteCarloPhysics, PhysicsConstraints, DefectEvolutionModel


class TrapParameterEncoder(nn.Module):
    """Encodes trap parameters into a latent representation using PhysicsNeMo's FullyConnected."""
    
    def __init__(
        self,
        trap_dim: int = 8,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 3,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        # Using PhysicsNeMo's FullyConnected module
        self.encoder = FullyConnected(
            in_features=trap_dim + 3,  # trap params + voltage + thickness + pulsewidth
            out_features=latent_dim,
            num_layers=num_layers,
            layer_size=hidden_dim,
            activation_fn=activation,
            skip_connections=True
        )
        
    def forward(
        self, 
        trap_params: torch.Tensor,
        voltage: torch.Tensor,
        thickness: torch.Tensor,
        pulsewidth: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            trap_params: [batch_size, 8]
            voltage: [batch_size, 1] 
            thickness: [batch_size, 1]
            pulsewidth: [batch_size, 1]
            
        Returns:
            latent: [batch_size, latent_dim]
        """
        # Concatenate all device parameters
        device_params = torch.cat([
            trap_params,
            voltage.unsqueeze(-1) if voltage.ndim == 1 else voltage,
            thickness.unsqueeze(-1) if thickness.ndim == 1 else thickness,
            pulsewidth.unsqueeze(-1) if pulsewidth.ndim == 1 else pulsewidth
        ], dim=-1)
        
        return self.encoder(device_params)


class PhysicsInformedEvolution(nn.Module):
    """Physics-informed temporal evolution module that integrates KMC physics."""
    
    def __init__(
        self,
        state_dim: int = 1,  # defect count
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        physics_config: Dict = None  # Can also provide default value
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Default physics config if not provided
        if physics_config is None:
            physics_config = {
                'constants': {
                    'k_B': 8.617e-5,  # eV/K
                    'q': 1.602e-19    # C
                },
                'device': {
                    'temperature_default': 300.0  # K
                }
            }
        
        # Initialize physics modules
        self.physics = KineticMonteCarloPhysics(physics_config)
        self.constraints = PhysicsConstraints()
        
        # Physics-based evolution model with neural correction
        self.evolution_model = DefectEvolutionModel(
            physics_config=physics_config,
            use_neural_correction=True
        )
        
        # RNN for temporal context (captures history dependence)
        self.temporal_rnn = nn.LSTM(
            input_size=state_dim + latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Fusion network to combine physics and RNN predictions
        self.fusion_net = FullyConnected(
            in_features=state_dim + hidden_dim,  # physics pred + RNN output
            out_features=state_dim,
            num_layers=2,
            layer_size=hidden_dim // 2,
            activation_fn='gelu'
        )
        
    def forward(
        self,
        initial_states: torch.Tensor,
        device_latent: torch.Tensor,
        voltage: torch.Tensor,
        thickness: torch.Tensor,
        trap_params: torch.Tensor,
        target_length: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            initial_states: [batch_size, initial_cycles]
            device_latent: [batch_size, latent_dim]
            voltage: [batch_size]
            thickness: [batch_size]
            trap_params: [batch_size, 8]
            target_length: number of cycles to predict
            
        Returns:
            predictions: [batch_size, target_length]
            physics_info: Dictionary with physics quantities
        """
        batch_size = initial_states.size(0)
        device = initial_states.device
        
        # Initialize hidden states for RNN
        h0 = torch.zeros(self.temporal_rnn.num_layers, batch_size, 
                        self.temporal_rnn.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)
        
        predictions = []
        physics_info_list = []
        
        # Start from last known state
        current_state = initial_states[:, -1:]  # [batch_size, 1]
        
        # Auto-regressive generation
        for t in range(target_length):
            # 1. Physics-based prediction
            physics_pred, physics_info = self.evolution_model(
                current_state=current_state,
                voltage=voltage,
                thickness=thickness,
                trap_params=trap_params,
                time_step=1.0  # One cycle
            )
            
            # 2. RNN prediction for temporal context
            rnn_input = torch.cat([
                current_state,
                device_latent
            ], dim=-1).unsqueeze(1)  # [batch_size, 1, input_dim]
            
            rnn_out, hidden = self.temporal_rnn(rnn_input, hidden)
            rnn_out = rnn_out.squeeze(1)  # [batch_size, hidden_dim]
            
            # 3. Fuse physics and RNN predictions
            fusion_input = torch.cat([
                physics_pred,
                rnn_out
            ], dim=-1)
            
            next_state = self.fusion_net(fusion_input)
            
            # 4. Apply physics constraints
            next_state = self.constraints.enforce_monotonicity(
                torch.cat([current_state, next_state], dim=1)
            )[:, -1:]
            
            next_state = self.constraints.enforce_breakdown_limit(next_state)
            
            predictions.append(next_state)
            physics_info_list.append(physics_info)
            current_state = next_state
            
        # Stack predictions
        predictions = torch.cat(predictions, dim=1)  # [batch_size, target_length]
        
        # Aggregate physics info
        aggregated_physics_info = {
            'electric_field': physics_info_list[0]['electric_field'],
            'generation_rate': torch.stack([p['generation_rate'] for p in physics_info_list], dim=1),
            'breakdown_prob': torch.stack([p['breakdown_prob'] for p in physics_info_list], dim=1)
        }
        
        return predictions, aggregated_physics_info


class BreakdownPredictor(nn.Module):
    """Predicts breakdown probability using physics-informed approach."""
    
    def __init__(
        self,
        state_dim: int = 1,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        breakdown_threshold: float = 200.0,
        critical_density: float = 1e22
    ):
        super().__init__()
        self.breakdown_threshold = breakdown_threshold
        self.critical_density = critical_density
        
        # Neural network for learned breakdown patterns
        self.predictor = FullyConnected(
            in_features=state_dim + latent_dim + 2,  # state + device params + field + cycle
            out_features=1,
            num_layers=3,
            layer_size=hidden_dim,
            activation_fn='relu'
        )
        
        # Physics module for percolation-based breakdown
        self.physics = KineticMonteCarloPhysics({
            'constants': {
                'k_B': 8.617e-5,  # eV/K
                'q': 1.602e-19    # C
            },
            'device': {
                'temperature_default': 300.0  # K
            }
        })
        
    def forward(
        self,
        states: torch.Tensor,
        device_latent: torch.Tensor,
        electric_field: torch.Tensor,
        cycles: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            states: [batch_size, seq_len] defect counts
            device_latent: [batch_size, latent_dim]
            electric_field: [batch_size] 
            cycles: [batch_size, seq_len]
            
        Returns:
            breakdown_prob: [batch_size, seq_len]
        """
        if states.ndim == 2:
            states = states.unsqueeze(-1)
            
        batch_size, seq_len = states.shape[:2]
        
        # Convert defect count to density (assuming unit volume)
        defect_density = states.squeeze(-1) * 1e20 / 1e-12  # Convert to cm^-3
        
        # Physics-based breakdown probability
        physics_breakdown_prob = self.physics.breakdown_probability(
            defect_density, self.critical_density
        )
        
        # Neural network prediction
        device_latent_expanded = device_latent.unsqueeze(1).expand(
            batch_size, seq_len, -1
        )
        electric_field_expanded = electric_field.unsqueeze(1).expand(
            batch_size, seq_len
        ).unsqueeze(-1)
        cycles_norm = cycles.unsqueeze(-1) / 2000.0
        
        nn_inputs = torch.cat([
            states,
            device_latent_expanded,
            electric_field_expanded,
            cycles_norm
        ], dim=-1)
        
        nn_breakdown_logits = self.predictor(nn_inputs).squeeze(-1)
        nn_breakdown_prob = torch.sigmoid(nn_breakdown_logits)
        
        # Combine physics and learned predictions
        combined_prob = 0.7 * physics_breakdown_prob + 0.3 * nn_breakdown_prob
        
        # Hard threshold override
        hard_breakdown = (states.squeeze(-1) >= self.breakdown_threshold).float()
        final_prob = torch.maximum(combined_prob, hard_breakdown)
        
        return final_prob


class PhysicsInformedFerroelectricSurrogate(nn.Module):
    """Main physics-informed surrogate model for ferroelectric device degradation."""
    
    def __init__(
        self,
        trap_dim: int = 8,
        state_dim: int = 1,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        encoder_layers: int = 3,
        evolution_layers: int = 2,
        breakdown_threshold: float = 200.0,
        physics_config: Optional[Dict] = None  # Using Optional[Dict] is also good practice
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.breakdown_threshold = breakdown_threshold
        
        # Default physics config if not provided
        if physics_config is None:
            physics_config = {
                'constants': {
                    'k_B': 8.617e-5,  # eV/K
                    'q': 1.602e-19    # C
                },
                'device': {
                    'temperature_default': 300.0  # K
                }
            }
        
        # Trap parameter encoder
        self.trap_encoder = TrapParameterEncoder(
            trap_dim=trap_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=encoder_layers
        )
        
        # Initial state encoder
        self.initial_state_encoder = FullyConnected(
            in_features=state_dim * 5,  # Assuming 5 initial cycles
            out_features=latent_dim // 2,
            num_layers=2,
            layer_size=hidden_dim // 2,
            activation_fn='gelu'
        )
        
        # Combine encodings
        self.combine_latents = nn.Linear(latent_dim + latent_dim // 2, latent_dim)
        
        # Physics-informed temporal evolution
        self.evolution_model = PhysicsInformedEvolution(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=evolution_layers,
            physics_config=physics_config
        )
        
        # Physics module for field calculation
        self.physics_constraints = PhysicsConstraints()
            
        # Breakdown predictor
        self.breakdown_predictor = BreakdownPredictor(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim // 2,
            breakdown_threshold=breakdown_threshold
        )
        
    def encode_device(
        self,
        trap_params: torch.Tensor,
        voltage: torch.Tensor,
        thickness: torch.Tensor,
        pulsewidth: torch.Tensor,
        initial_states: torch.Tensor
    ) -> torch.Tensor:
        """Encode device parameters and initial states."""
        # Encode trap parameters
        trap_latent = self.trap_encoder(trap_params, voltage, thickness, pulsewidth)
        
        # Encode initial states
        initial_flat = initial_states.flatten(start_dim=1)
        initial_latent = self.initial_state_encoder(initial_flat)
        
        # Combine latents
        combined = torch.cat([trap_latent, initial_latent], dim=-1)
        device_latent = self.combine_latents(combined)
        
        return device_latent
        
    def forward(
        self,
        trap_params: torch.Tensor,
        voltage: torch.Tensor,
        thickness: torch.Tensor,
        pulsewidth: torch.Tensor,
        initial_states: torch.Tensor,
        target_length: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the physics-informed surrogate model.
        
        Args:
            trap_params: [batch_size, 8]
            voltage: [batch_size]
            thickness: [batch_size]
            pulsewidth: [batch_size]
            initial_states: [batch_size, initial_cycles]
            target_length: number of cycles to predict
            
        Returns:
            Dictionary containing:
                - predictions: [batch_size, target_length] predicted defect counts
                - breakdown_prob: [batch_size, target_length] breakdown probabilities
                - device_latent: [batch_size, latent_dim] encoded device representation
                - physics_info: Dictionary with physics quantities
        """
        batch_size = trap_params.size(0)
        device = trap_params.device
        
        # Encode device
        device_latent = self.encode_device(
            trap_params, voltage, thickness, pulsewidth, initial_states
        )
        
        # Calculate electric field
        electric_field = self.physics_constraints.calculate_electric_field(
            voltage, thickness
        )
        
        # Generate predictions with physics
        predictions, physics_info = self.evolution_model(
            initial_states=initial_states,
            device_latent=device_latent,
            voltage=voltage,
            thickness=thickness,
            trap_params=trap_params,
            target_length=target_length
        )
            
        # Calculate cycles for each prediction
        initial_cycles = initial_states.size(1)
        cycles = torch.arange(
            initial_cycles, 
            initial_cycles + target_length,
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Predict breakdown probability
        breakdown_prob = self.breakdown_predictor(
            predictions, device_latent, electric_field, cycles
        )
        
        return {
            'predictions': predictions,
            'breakdown_prob': breakdown_prob,
            'device_latent': device_latent,
            'physics_info': physics_info
        }
        
    @torch.no_grad()
    def generate(
        self,
        trap_params: torch.Tensor,
        voltage: torch.Tensor,
        thickness: torch.Tensor,
        pulsewidth: torch.Tensor,
        initial_states: torch.Tensor,
        max_cycles: int = 2000,
        breakdown_threshold: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        Generate full trajectory until breakdown using physics.
        
        Args:
            trap_params: [batch_size, 8]
            voltage: [batch_size]
            thickness: [batch_size]
            pulsewidth: [batch_size]
            initial_states: [batch_size, initial_cycles]
            max_cycles: maximum cycles to generate
            breakdown_threshold: probability threshold for stopping
            
        Returns:
            Dictionary containing:
                - full_trajectory: [batch_size, num_cycles] complete defect evolution
                - breakdown_cycles: [batch_size] cycle where breakdown occurred
                - final_defect_counts: [batch_size] defect count at breakdown
        """
        self.eval()
        batch_size = trap_params.size(0)
        device = trap_params.device
        
        # Encode device once
        device_latent = self.encode_device(
            trap_params, voltage, thickness, pulsewidth, initial_states
        )
        
        # Initialize trajectory with initial states
        trajectories = [initial_states]
        current_states = initial_states
        
        # Track breakdown
        breakdown_cycles = torch.full((batch_size,), max_cycles, device=device)
        has_broken = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Calculate electric field once
        electric_field = self.physics_constraints.calculate_electric_field(
            voltage, thickness
        )
        
        # Generate iteratively
        window_size = 50  # Generate in chunks
        current_cycle = initial_states.size(1)
        
        while current_cycle < max_cycles and not has_broken.all():
            # Determine how many cycles to generate
            remaining = max_cycles - current_cycle
            chunk_size = min(window_size, remaining)
            
            # Generate next chunk with physics
            outputs = self.forward(
                trap_params, voltage, thickness, pulsewidth,
                current_states[:, -5:],  # Use last 5 states
                target_length=chunk_size
            )
            
            predictions = outputs['predictions']
            breakdown_prob = outputs['breakdown_prob']
            
            # Check for breakdown
            for i in range(batch_size):
                if not has_broken[i]:
                    # Check defect threshold
                    exceeds_threshold = predictions[i] >= self.breakdown_threshold
                    if exceeds_threshold.any():
                        first_exceed = exceeds_threshold.nonzero()[0].item()
                        breakdown_cycles[i] = current_cycle + first_exceed
                        has_broken[i] = True
                        predictions[i, first_exceed + 1:] = self.breakdown_threshold
                    
                    # Check breakdown probability
                    high_prob = breakdown_prob[i] >= breakdown_threshold
                    if high_prob.any() and not has_broken[i]:
                        first_high_prob = high_prob.nonzero()[0].item()
                        breakdown_cycles[i] = current_cycle + first_high_prob
                        has_broken[i] = True
                        predictions[i, first_high_prob + 1:] = self.breakdown_threshold
                    
            trajectories.append(predictions)
            current_states = torch.cat([current_states, predictions], dim=1)
            current_cycle += chunk_size
            
        # Combine full trajectory
        full_trajectory = torch.cat(trajectories, dim=1)
        
        # Get final defect counts
        final_defect_counts = torch.gather(
            full_trajectory, 1, 
            breakdown_cycles.unsqueeze(1).clamp(max=full_trajectory.size(1) - 1)
        ).squeeze(1)
        
        return {
            'full_trajectory': full_trajectory,
            'breakdown_cycles': breakdown_cycles,
            'final_defect_counts': final_defect_counts
        }