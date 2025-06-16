# models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PhysicsInformedLoss(nn.Module):
    """Combined loss function with physics constraints for ferroelectric degradation."""
    
    def __init__(
        self,
        prediction_weight: float = 1.0,
        breakdown_weight: float = 0.5,
        monotonic_weight: float = 0.2,
        smoothness_weight: float = 0.1,
        physics_weight: float = 0.3,
        breakdown_threshold: float = 200.0
    ):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.breakdown_weight = breakdown_weight
        self.monotonic_weight = monotonic_weight
        self.smoothness_weight = smoothness_weight
        self.physics_weight = physics_weight
        self.breakdown_threshold = breakdown_threshold
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        breakdown_prob: torch.Tensor,
        valid_mask: torch.Tensor,
        voltage: Optional[torch.Tensor] = None,
        thickness: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            predictions: [batch_size, seq_len] predicted defect counts
            targets: [batch_size, seq_len] target defect counts
            breakdown_prob: [batch_size, seq_len] predicted breakdown probabilities
            valid_mask: [batch_size, seq_len] mask for valid timesteps
            voltage: [batch_size] applied voltage (optional)
            thickness: [batch_size] film thickness (optional)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # 1. Prediction loss (MSE on valid timesteps only)
        # Only calculate loss where we have valid targets (not NaN)
        # valid_mask already indicates where we have real data vs. padding/breakdown
        if valid_mask.sum() > 0:
            # Only compute loss on valid timesteps
            valid_predictions = predictions[valid_mask.bool()]
            valid_targets = targets[valid_mask.bool()]
            # Filter out any remaining NaN values
            not_nan_mask = ~torch.isnan(valid_targets)
            if not_nan_mask.any():
                pred_loss = F.mse_loss(valid_predictions[not_nan_mask], valid_targets[not_nan_mask])
            else:
                pred_loss = torch.tensor(0.0, device=predictions.device)
        else:
            pred_loss = torch.tensor(0.0, device=predictions.device)
        losses['prediction'] = pred_loss
        
        # 2. Breakdown prediction loss
        # Target: 1 where defect count >= threshold or at the last valid step before NaN
        breakdown_target = torch.zeros_like(breakdown_prob)
        
        # Mark where targets exceed threshold
        breakdown_target[targets >= self.breakdown_threshold] = 1.0
        
        # Also mark the last valid timestep before breakdown (NaN) occurs
        batch_size, seq_len = valid_mask.shape
        for i in range(batch_size):
            valid_indices = valid_mask[i].nonzero()
            if len(valid_indices) > 0:
                last_valid = valid_indices[-1].item()
                # If the next timestep would be NaN (breakdown), mark current as breakdown
                if last_valid < seq_len - 1 and not valid_mask[i, last_valid + 1]:
                    breakdown_target[i, last_valid] = 1.0
                    
        # Only compute breakdown loss on valid timesteps
        if valid_mask.sum() > 0:
            breakdown_loss = F.binary_cross_entropy(
                breakdown_prob[valid_mask],
                breakdown_target[valid_mask]
            )
        else:
            breakdown_loss = torch.tensor(0.0, device=predictions.device)
        losses['breakdown'] = breakdown_loss
        
        # 3. Monotonicity constraint (defects should not decrease)
        if predictions.size(1) > 1:
            diff = predictions[:, 1:] - predictions[:, :-1]
            monotonic_violation = F.relu(-diff)  # Penalize negative differences
            
            # Only apply where both timesteps are valid
            valid_pairs = valid_mask[:, 1:] * valid_mask[:, :-1]
            monotonic_loss = (monotonic_violation * valid_pairs).sum() / (valid_pairs.sum() + 1e-8)
            losses['monotonic'] = monotonic_loss
        else:
            losses['monotonic'] = torch.tensor(0.0, device=predictions.device)
            
        # 4. Smoothness constraint (penalize large jumps)
        if predictions.size(1) > 1:
            second_diff = predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2]
            
            # Only apply where all three timesteps are valid
            if predictions.size(1) > 2:
                valid_triplets = valid_mask[:, 2:] * valid_mask[:, 1:-1] * valid_mask[:, :-2]
                smoothness_loss = (second_diff.abs() * valid_triplets).sum() / (valid_triplets.sum() + 1e-8)
            else:
                smoothness_loss = torch.tensor(0.0, device=predictions.device)
            losses['smoothness'] = smoothness_loss
        else:
            losses['smoothness'] = torch.tensor(0.0, device=predictions.device)
            
        # 5. Physics-based loss (field-dependent generation rate)
        if voltage is not None and thickness is not None:
            # Electric field in MV/cm
            electric_field = voltage / thickness * 1e-7  # Convert from V/m to MV/cm
            
            # Expected acceleration with field (simplified thermochemical model)
            # Higher field should lead to faster degradation
            field_factor = torch.exp(0.1 * electric_field)  # Simplified field acceleration
            
            # Calculate average generation rate from predictions
            valid_diffs = (predictions[:, 1:] - predictions[:, :-1]) * valid_mask[:, :-1]
            avg_rate = valid_diffs.sum(dim=1) / (valid_mask[:, :-1].sum(dim=1) + 1e-8)
            
            # Physics loss: generation rate should correlate with field
            # This is a soft constraint to guide the model
            physics_loss = F.mse_loss(
                torch.log(avg_rate + 1e-8),
                torch.log(field_factor + 1e-8)
            )
            losses['physics'] = physics_loss
        else:
            losses['physics'] = torch.tensor(0.0, device=predictions.device)
            
        # Combine losses
        total_loss = (
            self.prediction_weight * losses['prediction'] +
            self.breakdown_weight * losses['breakdown'] +
            self.monotonic_weight * losses['monotonic'] +
            self.smoothness_weight * losses['smoothness'] +
            self.physics_weight * losses['physics']
        )
        
        losses['total'] = total_loss
        
        return losses


class GenerationRateLoss(nn.Module):
    """Additional physics loss based on kinetic Monte Carlo generation rates."""
    
    def __init__(self, temperature: float = 300.0):
        super().__init__()
        self.temperature = temperature
        self.k_B = 8.617e-5  # Boltzmann constant in eV/K
        
    def forward(
        self,
        predictions: torch.Tensor,
        voltage: torch.Tensor,
        thickness: torch.Tensor,
        trap_params: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate physics-based generation rate loss.
        
        Based on thermochemical model: G = ν * exp(-(E_A - p*E)/(k_B*T))
        """
        # Extract activation energy from trap parameters
        # Assuming it's related to relaxation energy (index 3)
        E_A = trap_params[:, 3]  # Relaxation energy
        
        # Calculate electric field
        electric_field = voltage / thickness  # V/m
        
        # Dipole moment (simplified - could be learned)
        p = 1e-29  # C*m (typical value)
        
        # Calculate expected generation rate
        exponent = -(E_A - p * electric_field) / (self.k_B * self.temperature)
        expected_rate = torch.exp(exponent)
        
        # Calculate observed generation rate from predictions
        if predictions.size(1) > 1:
            observed_rate = (predictions[:, 1:] - predictions[:, :-1]) * valid_mask[:, :-1]
            avg_observed_rate = observed_rate.sum(dim=1) / (valid_mask[:, :-1].sum(dim=1) + 1e-8)
        else:
            avg_observed_rate = torch.zeros_like(expected_rate)
            
        # Loss: observed should match expected (in log space for stability)
        loss = F.mse_loss(
            torch.log(avg_observed_rate + 1e-10),
            torch.log(expected_rate + 1e-10)
        )
        
        return loss


class CycleLoss(nn.Module):
    """Loss based on predicting the correct breakdown cycle."""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        breakdown_prob: torch.Tensor,
        breakdown_cycles: torch.Tensor,
        initial_cycles: int = 5
    ) -> torch.Tensor:
        """
        Calculate loss based on predicting the correct breakdown cycle.
        
        Args:
            breakdown_prob: [batch_size, seq_len] breakdown probabilities
            breakdown_cycles: [batch_size] true breakdown cycles
            initial_cycles: number of initial cycles (offset)
        """
        batch_size, seq_len = breakdown_prob.shape
        device = breakdown_prob.device
        
        # Create target distribution (one-hot at breakdown cycle)
        target = torch.zeros_like(breakdown_prob)
        
        for i in range(batch_size):
            # Adjust for the offset and sequence length
            target_idx = breakdown_cycles[i] - initial_cycles
            if 0 <= target_idx < seq_len:
                target[i, target_idx] = 1.0
            elif target_idx >= seq_len:
                # Breakdown happens after our prediction window
                target[i, -1] = 1.0  # Mark last timestep
                
        # Cross-entropy loss
        loss = F.binary_cross_entropy(breakdown_prob, target)
        
        return loss