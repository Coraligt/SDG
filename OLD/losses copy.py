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
            targets: [batch_size, seq_len] target defect counts (breakdown_threshold for invalid)
            breakdown_prob: [batch_size, seq_len] predicted breakdown probabilities
            valid_mask: [batch_size, seq_len] mask for valid timesteps (False after breakdown)
            voltage: [batch_size] applied voltage (optional)
            thickness: [batch_size] film thickness (optional)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Convert valid_mask to float for calculations
        valid_mask_float = valid_mask.float()
        
        # 1. Prediction loss (MSE on valid timesteps only)
        # Only compute loss where we have valid (non-breakdown) data
        if valid_mask.any():
            # Compute MSE only on valid timesteps
            pred_error = (predictions - targets) * valid_mask_float
            pred_loss = (pred_error ** 2).sum() / valid_mask_float.sum()
        else:
            pred_loss = torch.tensor(0.0, device=predictions.device)
        losses['prediction'] = pred_loss
        
        # 2. Breakdown prediction loss
        # Create breakdown target: 1 at the last valid timestep before breakdown
        breakdown_target = torch.zeros_like(breakdown_prob)
        
        batch_size, seq_len = valid_mask.shape
        for i in range(batch_size):
            # Find where breakdown occurs
            valid_indices = valid_mask[i].nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                last_valid = valid_indices[-1].item()
                
                # Check if breakdown happens in our prediction window
                if last_valid < seq_len - 1:
                    # Next timestep is invalid, so breakdown happens here
                    if not valid_mask[i, last_valid + 1]:
                        breakdown_target[i, last_valid] = 1.0
                
                # Also check if defect count exceeds threshold
                if targets[i, last_valid] >= self.breakdown_threshold * 0.95:  # Allow some margin
                    breakdown_target[i, last_valid] = 1.0
                    
        # Binary cross-entropy for breakdown prediction
        if valid_mask.any():
            eps = 1e-7
            breakdown_prob_clamped = breakdown_prob.clamp(min=eps, max=1-eps)
            
            # Only compute loss on valid timesteps
            bce = -(breakdown_target * torch.log(breakdown_prob_clamped) + 
                   (1 - breakdown_target) * torch.log(1 - breakdown_prob_clamped))
            
            # Weight the loss more heavily near breakdown points
            weight = valid_mask_float.clone()
            weight[breakdown_target > 0.5] = 5.0  # Higher weight for breakdown timesteps
            
            breakdown_loss = (bce * weight).sum() / weight.sum()
        else:
            breakdown_loss = torch.tensor(0.0, device=predictions.device)
        losses['breakdown'] = breakdown_loss
        
        # 3. Monotonicity constraint (defects should not decrease)
        if predictions.size(1) > 1 and valid_mask[:, 1:].any():
            diff = predictions[:, 1:] - predictions[:, :-1]
            monotonic_violation = F.relu(-diff)  # Penalize negative differences
            
            # Only apply where both timesteps are valid
            valid_pairs = valid_mask_float[:, 1:] * valid_mask_float[:, :-1]
            if valid_pairs.sum() > 0:
                monotonic_loss = (monotonic_violation * valid_pairs).sum() / valid_pairs.sum()
            else:
                monotonic_loss = torch.tensor(0.0, device=predictions.device)
        else:
            monotonic_loss = torch.tensor(0.0, device=predictions.device)
        losses['monotonic'] = monotonic_loss
            
        # 4. Smoothness constraint (penalize large jumps)
        if predictions.size(1) > 2 and valid_mask[:, 2:].any():
            second_diff = predictions[:, 2:] - 2 * predictions[:, 1:-1] + predictions[:, :-2]
            
            # Only apply where all three timesteps are valid
            valid_triplets = valid_mask_float[:, 2:] * valid_mask_float[:, 1:-1] * valid_mask_float[:, :-2]
            if valid_triplets.sum() > 0:
                smoothness_loss = (second_diff.abs() * valid_triplets).sum() / valid_triplets.sum()
            else:
                smoothness_loss = torch.tensor(0.0, device=predictions.device)
        else:
            smoothness_loss = torch.tensor(0.0, device=predictions.device)
        losses['smoothness'] = smoothness_loss
            
        # 5. Physics-based loss (field-dependent generation rate)
        if voltage is not None and thickness is not None and predictions.size(1) > 1:
            # Electric field in MV/cm
            electric_field = torch.abs(voltage) / thickness * 1e-7  # Convert from V/m to MV/cm
            
            # Expected acceleration with field (simplified thermochemical model)
            field_factor = torch.exp(0.1 * electric_field)  # Simplified field acceleration
            
            # Calculate average generation rate from predictions (only on valid timesteps)
            if valid_mask[:, 1:].any():
                valid_diffs = (predictions[:, 1:] - predictions[:, :-1]) * valid_mask_float[:, :-1]
                valid_counts = valid_mask_float[:, :-1].sum(dim=1).clamp(min=1)
                avg_rate = valid_diffs.sum(dim=1) / valid_counts
                
                # Only compute physics loss for samples with valid rates
                mask = avg_rate > 0
                if mask.any():
                    physics_loss = F.mse_loss(
                        torch.log(avg_rate[mask].clamp(min=1e-8)),
                        torch.log(field_factor[mask].clamp(min=1e-8))
                    )
                else:
                    physics_loss = torch.tensor(0.0, device=predictions.device)
            else:
                physics_loss = torch.tensor(0.0, device=predictions.device)
        else:
            physics_loss = torch.tensor(0.0, device=predictions.device)
        losses['physics'] = physics_loss
            
        # 6. Breakdown boundary loss - ensure predictions don't exceed threshold
        boundary_violation = F.relu(predictions - self.breakdown_threshold)
        boundary_loss = boundary_violation.mean()
        losses['boundary'] = boundary_loss * 0.1  # Small weight
        
        # Combine losses
        total_loss = (
            self.prediction_weight * losses['prediction'] +
            self.breakdown_weight * losses['breakdown'] +
            self.monotonic_weight * losses['monotonic'] +
            self.smoothness_weight * losses['smoothness'] +
            self.physics_weight * losses['physics'] +
            losses['boundary']
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
        E_A = trap_params[:, 3]  # Relaxation energy
        
        # Calculate electric field
        electric_field = torch.abs(voltage) / thickness  # V/m
        
        # Dipole moment (simplified - could be learned)
        p = 1e-29  # C*m (typical value)
        
        # Calculate expected generation rate
        exponent = -(E_A - p * electric_field / 1.602e-19) / (self.k_B * self.temperature)
        expected_rate = torch.exp(exponent.clamp(min=-20, max=20))  # Clamp for stability
        
        # Calculate observed generation rate from predictions
        if predictions.size(1) > 1 and valid_mask[:, 1:].any():
            valid_mask_float = valid_mask.float()
            
            # Only calculate rate where both timesteps are valid
            valid_pairs = valid_mask_float[:, 1:] * valid_mask_float[:, :-1]
            observed_diffs = (predictions[:, 1:] - predictions[:, :-1]) * valid_pairs
            
            # Average over valid timesteps
            valid_counts = valid_pairs.sum(dim=1).clamp(min=1)
            avg_observed_rate = observed_diffs.sum(dim=1) / valid_counts
            
            # Only include samples with positive rates
            mask = (avg_observed_rate > 0) & (valid_counts > 1)
            if mask.any():
                loss = F.mse_loss(
                    torch.log(avg_observed_rate[mask].clamp(min=1e-10)),
                    torch.log(expected_rate[mask].clamp(min=1e-10))
                )
            else:
                loss = torch.tensor(0.0, device=predictions.device)
        else:
            loss = torch.tensor(0.0, device=predictions.device)
            
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
        
        # Create target distribution
        target = torch.zeros_like(breakdown_prob)
        
        for i in range(batch_size):
            # Adjust for the offset and sequence length
            target_idx = breakdown_cycles[i] - initial_cycles
            
            if 0 <= target_idx < seq_len:
                # Breakdown occurs within our prediction window
                target[i, target_idx] = 1.0
            elif target_idx >= seq_len:
                # Breakdown happens after our prediction window
                # Encourage low breakdown probability throughout
                pass  # target remains all zeros
            else:
                # This shouldn't happen as we filter such samples
                logger.warning(f"Negative breakdown index: {target_idx}")
                
        # Focal loss variant for better handling of class imbalance
        gamma = 2.0  # Focusing parameter
        eps = 1e-7
        
        breakdown_prob_clamped = breakdown_prob.clamp(min=eps, max=1-eps)
        
        # Compute focal loss
        ce_loss = -(target * torch.log(breakdown_prob_clamped) + 
                   (1 - target) * torch.log(1 - breakdown_prob_clamped))
        
        # Apply focusing factor
        pt = torch.where(target == 1, breakdown_prob_clamped, 1 - breakdown_prob_clamped)
        focal_weight = (1 - pt) ** gamma
        
        loss = (focal_weight * ce_loss).mean()
        
        return loss