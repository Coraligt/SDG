# models/fe_surrogate.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

# PhysicsNeMo imports
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.models.rnn import RNN
from physicsnemo.models.fno import FNO1d
from physicsnemo.models.embeddings import SinusoidalEmbedding


class TrapParameterEncoder(nn.Module):
    """Encodes trap parameters into a latent representation."""
    
    def __init__(
        self,
        trap_dim: int = 8,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 3,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.encoder = FullyConnected(
            in_features=trap_dim + 3,  # trap params + voltage + thickness + pulsewidth
            out_features=latent_dim,
            num_layers=num_layers,
            layer_size=hidden_dim,
            activation=activation,
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
    """Physics-informed temporal evolution module."""
    
    def __init__(
        self,
        state_dim: int = 1,  # defect count
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        
        # Time embedding for cycle information
        self.time_embedding = SinusoidalEmbedding(
            dim=32,
            max_period=2000  # max cycles
        )
        
        # Main evolution model - LSTM for temporal dynamics
        self.temporal_model = nn.LSTM(
            input_size=state_dim + latent_dim + 32,  # state + device latent + time
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = FullyConnected(
            in_features=hidden_dim,
            out_features=state_dim,
            num_layers=2,
            layer_size=hidden_dim // 2,
            activation='relu'
        )
        
        # Physics-based generation rate (inspired by kMC)
        self.generation_rate = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # device params + current state
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive rate
        )
        
    def forward(
        self,
        initial_states: torch.Tensor,
        device_latent: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """
        Args:
            initial_states: [batch_size, initial_cycles]
            device_latent: [batch_size, latent_dim]
            target_length: number of cycles to predict
            
        Returns:
            predictions: [batch_size, target_length]
        """
        batch_size = initial_states.size(0)
        device = initial_states.device
        
        # Initialize hidden states
        h0 = torch.zeros(self.temporal_model.num_layers, batch_size, 
                        self.temporal_model.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)
        
        predictions = []
        
        # Use last known state as starting point
        current_state = initial_states[:, -1:]  # [batch_size, 1]
        
        # Get initial cycle number
        initial_cycle = initial_states.size(1)
        
        # Auto-regressive generation
        for t in range(target_length):
            # Current cycle number
            cycle_num = initial_cycle + t
            time_embed = self.time_embedding(
                torch.tensor([cycle_num], device=device).expand(batch_size)
            )  # [batch_size, 32]
            
            # Prepare input: current state + device params + time
            lstm_input = torch.cat([
                current_state,
                device_latent,
                time_embed
            ], dim=-1).unsqueeze(1)  # [batch_size, 1, input_dim]
            
            # LSTM forward
            lstm_out, hidden = self.temporal_model(lstm_input, hidden)
            lstm_out = lstm_out.squeeze(1)  # [batch_size, hidden_dim]
            
            # Calculate physics-based generation rate
            rate_input = torch.cat([device_latent, current_state], dim=-1)
            generation_rate = self.generation_rate(rate_input)  # [batch_size, 1]
            
            # Predict state change
            delta_state = self.output_projection(lstm_out)  # [batch_size, 1]
            
            # Apply physics constraint: defects can only increase (monotonic)
            delta_state = F.relu(delta_state)
            
            # Scale by generation rate
            delta_state = delta_state * generation_rate
            
            # Update state
            next_state = current_state + delta_state
            
            predictions.append(next_state)
            current_state = next_state
            
        # Stack predictions
        predictions = torch.cat(predictions, dim=1)  # [batch_size, target_length]
        
        return predictions


class BreakdownPredictor(nn.Module):
    """Predicts breakdown probability based on current state and device parameters."""
    
    def __init__(
        self,
        state_dim: int = 1,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        threshold: float = 200.0
    ):
        super().__init__()
        self.threshold = threshold
        
        self.predictor = FullyConnected(
            in_features=state_dim + latent_dim + 1,  # state + device params + cycle
            out_features=1,
            num_layers=3,
            layer_size=hidden_dim,
            activation='relu'
        )
        
    def forward(
        self,
        states: torch.Tensor,
        device_latent: torch.Tensor,
        cycles: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            states: [batch_size, seq_len] or [batch_size, seq_len, 1]
            device_latent: [batch_size, latent_dim]
            cycles: [batch_size, seq_len]
            
        Returns:
            breakdown_prob: [batch_size, seq_len]
        """
        if states.ndim == 2:
            states = states.unsqueeze(-1)
            
        batch_size, seq_len = states.shape[:2]
        
        # Expand device latent for all timesteps
        device_latent_expanded = device_latent.unsqueeze(1).expand(
            batch_size, seq_len, -1
        )  # [batch_size, seq_len, latent_dim]
        
        # Normalize cycles
        cycles_norm = cycles.unsqueeze(-1) / 2000.0  # [batch_size, seq_len, 1]
        
        # Concatenate inputs
        inputs = torch.cat([
            states,
            device_latent_expanded,
            cycles_norm
        ], dim=-1)  # [batch_size, seq_len, input_dim]
        
        # Predict breakdown probability
        breakdown_logits = self.predictor(inputs).squeeze(-1)  # [batch_size, seq_len]
        
        # Add hard threshold - if state >= 200, breakdown prob = 1
        hard_breakdown = (states.squeeze(-1) >= self.threshold).float()
        
        # Combine learned and hard threshold
        breakdown_prob = torch.maximum(
            torch.sigmoid(breakdown_logits),
            hard_breakdown
        )
        
        return breakdown_prob


class FerroelectricSurrogate(nn.Module):
    """Main surrogate model for ferroelectric device degradation prediction."""
    
    def __init__(
        self,
        trap_dim: int = 8,
        state_dim: int = 1,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        encoder_layers: int = 3,
        evolution_layers: int = 2,
        breakdown_threshold: float = 200.0,
        use_fno: bool = False
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.breakdown_threshold = breakdown_threshold
        self.use_fno = use_fno
        
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
            activation='gelu'
        )
        
        # Combine encodings
        self.combine_latents = nn.Linear(latent_dim + latent_dim // 2, latent_dim)
        
        # Temporal evolution model
        if use_fno:
            self.evolution_model = FNO1d(
                in_channels=state_dim + latent_dim,
                out_channels=state_dim,
                modes=16,
                width=hidden_dim,
                n_layers=4
            )
        else:
            self.evolution_model = PhysicsInformedEvolution(
                state_dim=state_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=evolution_layers
            )
            
        # Breakdown predictor
        self.breakdown_predictor = BreakdownPredictor(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim // 2,
            threshold=breakdown_threshold
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
        Forward pass of the surrogate model.
        
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
        """
        batch_size = trap_params.size(0)
        device = trap_params.device
        
        # Encode device
        device_latent = self.encode_device(
            trap_params, voltage, thickness, pulsewidth, initial_states
        )
        
        # Generate predictions
        if self.use_fno:
            # FNO expects [batch, channels, length]
            # We'll use a different approach for FNO
            raise NotImplementedError("FNO evolution not yet implemented")
        else:
            predictions = self.evolution_model(
                initial_states, device_latent, target_length
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
            predictions, device_latent, cycles
        )
        
        return {
            'predictions': predictions,
            'breakdown_prob': breakdown_prob,
            'device_latent': device_latent
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
        Generate full trajectory until breakdown.
        
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
        
        # Generate iteratively
        window_size = 50  # Generate in chunks
        current_cycle = initial_states.size(1)
        
        while current_cycle < max_cycles and not has_broken.all():
            # Determine how many cycles to generate
            remaining = max_cycles - current_cycle
            chunk_size = min(window_size, remaining)
            
            # Generate next chunk
            outputs = self.forward(
                trap_params, voltage, thickness, pulsewidth,
                current_states[:, -5:],  # Use last 5 states
                target_length=chunk_size
            )
            
            predictions = outputs['predictions']
            breakdown_prob = outputs['breakdown_prob']
            
            # Check for breakdown in predictions
            # If current prediction >= threshold, device has broken down
            # Mark all subsequent predictions as broken (could use a special value or mask)
            for i in range(batch_size):
                if not has_broken[i]:
                    # Check if any prediction in this chunk exceeds threshold
                    exceeds_threshold = predictions[i] >= self.breakdown_threshold
                    if exceeds_threshold.any():
                        first_exceed = exceeds_threshold.nonzero()[0].item()
                        breakdown_cycles[i] = current_cycle + first_exceed
                        has_broken[i] = True
                        # Important: Don't continue predicting after breakdown
                        # Set remaining predictions to the breakdown value
                        predictions[i, first_exceed + 1:] = self.breakdown_threshold
                    
                    # Also check breakdown probability
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