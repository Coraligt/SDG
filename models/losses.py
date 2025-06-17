# models/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PhysicsInformedLoss(nn.Module):
    """Combined loss function with physics constraints for ferroelectric degradation."""
    
    def __init__(
        self,
        prediction_weight: float = 1.0,
        breakdown_weight: float = 0.5,
        monotonic_weight: float = 0.2,
        smoothness_weight: float = 0.1,
        physics_weight: float = 0.3,
        breakdown_threshold: float = 200.0,
        eps: float = 1e-8  # Small constant for numerical stability
    ):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.breakdown_weight = breakdown_weight
        self.monotonic_weight = monotonic_weight
        self.smoothness_weight = smoothness_weight
        self.physics_weight = physics_weight
        self.breakdown_threshold = breakdown_threshold
        self.eps = eps
        
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
        Calculate combined loss with numerical stability improvements.
        
        Args:
            predictions: [batch_size, seq_len] predicted defect counts
            targets: [batch_size, seq_len] target defect counts
            breakdown_prob: [batch_size, seq_len] predicted breakdown probabilities
            valid_mask: [batch_size, seq_len] mask for valid timesteps (False after breakdown)
            voltage: [batch_size] applied voltage (optional)
            thickness: [batch_size] film thickness (optional)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        device = predictions.device
        
        # Convert valid_mask to float for calculations
        valid_mask_float = valid_mask.float()
        
        # Ensure we have at least some valid samples
        total_valid = valid_mask_float.sum()
        if total_valid < 1:
            logger.warning("No valid timesteps in batch")
            # Return minimal losses
            return {
                'total': torch.tensor(0.0, device=device),
                'prediction': torch.tensor(0.0, device=device),
                'breakdown': torch.tensor(0.0, device=device),
                'monotonic': torch.tensor(0.0, device=device),
                'smoothness': torch.tensor(0.0, device=device),
                'physics': torch.tensor(0.0, device=device),
                'boundary': torch.tensor(0.0, device=device)
            }
        
        # 1. Prediction loss (MSE on valid timesteps only)
        # Clip predictions and targets to reasonable range to prevent overflow
        predictions_clipped = torch.clamp(predictions, min=0, max=300)
        targets_clipped = torch.clamp(targets, min=0, max=300)
        
        # Compute MSE only on valid timesteps
        pred_error = (predictions_clipped - targets_clipped) * valid_mask_float
        pred_loss = (pred_error ** 2).sum() / (total_valid + self.eps)
        
        # Additional check for NaN
        if torch.isnan(pred_loss) or torch.isinf(pred_loss):
            pred_loss = torch.tensor(1.0, device=device)
            
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
                if targets_clipped[i, last_valid] >= self.breakdown_threshold * 0.95:
                    breakdown_target[i, last_valid] = 1.0
                    
        # Binary cross-entropy for breakdown prediction with numerical stability
        breakdown_prob_clamped = torch.clamp(breakdown_prob, min=self.eps, max=1-self.eps)
        
        # Only compute loss on valid timesteps
        bce = -(breakdown_target * torch.log(breakdown_prob_clamped) + 
               (1 - breakdown_target) * torch.log(1 - breakdown_prob_clamped))
        
        # Weight the loss more heavily near breakdown points
        weight = valid_mask_float.clone()
        weight[breakdown_target > 0.5] = 5.0  # Higher weight for breakdown timesteps
        
        weighted_bce = bce * weight
        breakdown_loss = weighted_bce.sum() / (weight.sum() + self.eps)
        
        if torch.isnan(breakdown_loss) or torch.isinf(breakdown_loss):
            breakdown_loss = torch.tensor(0.1, device=device)
            
        losses['breakdown'] = breakdown_loss
        
        # 3. Monotonicity constraint (defects should not decrease)
        if predictions.size(1) > 1:
            # Calculate differences with clamping
            diff = predictions_clipped[:, 1:] - predictions_clipped[:, :-1]
            
            # Penalize negative differences
            monotonic_violation = F.relu(-diff)
            
            # Only apply where both timesteps are valid
            valid_pairs = valid_mask_float[:, 1:] * valid_mask_float[:, :-1]
            
            if valid_pairs.sum() > 0:
                monotonic_loss = (monotonic_violation * valid_pairs).sum() / (valid_pairs.sum() + self.eps)
            else:
                monotonic_loss = torch.tensor(0.0, device=device)
                
            if torch.isnan(monotonic_loss) or torch.isinf(monotonic_loss):
                monotonic_loss = torch.tensor(0.0, device=device)
        else:
            monotonic_loss = torch.tensor(0.0, device=device)
            
        losses['monotonic'] = monotonic_loss
            
        # 4. Smoothness constraint (penalize large jumps)
        if predictions.size(1) > 2:
            # Second order differences
            second_diff = predictions_clipped[:, 2:] - 2 * predictions_clipped[:, 1:-1] + predictions_clipped[:, :-2]
            
            # Only apply where all three timesteps are valid
            valid_triplets = valid_mask_float[:, 2:] * valid_mask_float[:, 1:-1] * valid_mask_float[:, :-2]
            
            if valid_triplets.sum() > 0:
                # Use L1 norm instead of L2 to be more robust to outliers
                smoothness_loss = (second_diff.abs() * valid_triplets).sum() / (valid_triplets.sum() + self.eps)
            else:
                smoothness_loss = torch.tensor(0.0, device=device)
                
            if torch.isnan(smoothness_loss) or torch.isinf(smoothness_loss):
                smoothness_loss = torch.tensor(0.0, device=device)
        else:
            smoothness_loss = torch.tensor(0.0, device=device)
            
        losses['smoothness'] = smoothness_loss
            
        # 5. Physics-based loss (field-dependent generation rate)
        if voltage is not None and thickness is not None and predictions.size(1) > 1:
            # Electric field in MV/cm with clamping
            thickness_safe = torch.clamp(thickness, min=1e-10)
            electric_field = torch.abs(voltage) / thickness_safe * 1e-7  # Convert to MV/cm
            electric_field = torch.clamp(electric_field, min=0, max=100)  # Reasonable range
            
            # Expected acceleration with field (simplified thermochemical model)
            field_factor = torch.exp(torch.clamp(0.1 * electric_field, min=-10, max=10))
            
            # Calculate average generation rate from predictions
            valid_diffs = (predictions_clipped[:, 1:] - predictions_clipped[:, :-1]) * valid_mask_float[:, :-1]
            valid_counts = valid_mask_float[:, :-1].sum(dim=1).clamp(min=1)
            avg_rate = valid_diffs.sum(dim=1) / valid_counts
            
            # Only compute physics loss for samples with positive rates
            mask = (avg_rate > 0) & (field_factor > 0)
            if mask.any():
                # Use log space with clamping for stability
                log_rate = torch.log(avg_rate[mask].clamp(min=self.eps))
                log_field = torch.log(field_factor[mask].clamp(min=self.eps))
                
                # MSE in log space
                physics_loss = F.mse_loss(log_rate, log_field)
                
                if torch.isnan(physics_loss) or torch.isinf(physics_loss):
                    physics_loss = torch.tensor(0.0, device=device)
            else:
                physics_loss = torch.tensor(0.0, device=device)
        else:
            physics_loss = torch.tensor(0.0, device=device)
            
        losses['physics'] = physics_loss
            
        # 6. Boundary loss - ensure predictions don't exceed threshold
        boundary_violation = F.relu(predictions_clipped - self.breakdown_threshold)
        boundary_loss = boundary_violation.mean()
        
        if torch.isnan(boundary_loss) or torch.isinf(boundary_loss):
            boundary_loss = torch.tensor(0.0, device=device)
            
        losses['boundary'] = boundary_loss * 0.1  # Small weight
        
        # Combine losses with safety checks
        total_loss = torch.tensor(0.0, device=device)
        
        for name, (weight, loss) in zip(
            ['prediction', 'breakdown', 'monotonic', 'smoothness', 'physics', 'boundary'],
            [
                (self.prediction_weight, losses['prediction']),
                (self.breakdown_weight, losses['breakdown']),
                (self.monotonic_weight, losses['monotonic']),
                (self.smoothness_weight, losses['smoothness']),
                (self.physics_weight, losses['physics']),
                (1.0, losses['boundary'])
            ]
        ):
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss = total_loss + weight * loss
            else:
                logger.warning(f"Skipping {name} loss due to NaN/Inf")
        
        losses['total'] = total_loss
        
        return losses


class GenerationRateLoss(nn.Module):
    """Additional physics loss based on kinetic Monte Carlo generation rates."""
    
    def __init__(self, temperature: float = 300.0, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.k_B = 8.617e-5  # Boltzmann constant in eV/K
        self.eps = eps
        
    def forward(
        self,
        predictions: torch.Tensor,
        voltage: torch.Tensor,
        thickness: torch.Tensor,
        trap_params: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate physics-based generation rate loss with stability improvements.
        """
        device = predictions.device
        
        # Ensure valid inputs
        thickness_safe = torch.clamp(thickness, min=1e-10)
        
        # Extract activation energy from trap parameters
        E_A = torch.clamp(trap_params[:, 3], min=0.1, max=5.0)  # Relaxation energy
        
        # Calculate electric field
        electric_field = torch.abs(voltage) / thickness_safe  # V/m
        electric_field = torch.clamp(electric_field, min=0, max=1e10)  # Reasonable range
        
        # Dipole moment (simplified - could be learned)
        p = 1e-29  # C*m (typical value)
        
        # Calculate expected generation rate with stability
        field_term = p * electric_field / 1.602e-19  # Convert to eV
        field_term = torch.clamp(field_term, min=-2.0, max=2.0)  # Limit field effect
        
        exponent = -(E_A - field_term) / (self.k_B * self.temperature)
        exponent = torch.clamp(exponent, min=-20, max=20)  # Prevent overflow
        
        expected_rate = torch.exp(exponent)
        
        # Calculate observed generation rate from predictions
        if predictions.size(1) > 1:
            valid_mask_float = valid_mask.float()
            
            # Clamp predictions
            predictions_clipped = torch.clamp(predictions, min=0, max=300)
            
            # Only calculate rate where both timesteps are valid
            valid_pairs = valid_mask_float[:, 1:] * valid_mask_float[:, :-1]
            observed_diffs = (predictions_clipped[:, 1:] - predictions_clipped[:, :-1]) * valid_pairs
            
            # Average over valid timesteps
            valid_counts = valid_pairs.sum(dim=1).clamp(min=1)
            avg_observed_rate = observed_diffs.sum(dim=1) / valid_counts
            
            # Only include samples with positive rates and valid data
            mask = (avg_observed_rate > 0) & (expected_rate > 0) & (valid_counts > 1)
            
            if mask.any():
                # Work in log space for stability
                log_observed = torch.log(avg_observed_rate[mask].clamp(min=self.eps))
                log_expected = torch.log(expected_rate[mask].clamp(min=self.eps))
                
                # MSE in log space
                loss = F.mse_loss(log_observed, log_expected)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    loss = torch.tensor(0.0, device=device)
            else:
                loss = torch.tensor(0.0, device=device)
        else:
            loss = torch.tensor(0.0, device=device)
            
        return loss


class CycleLoss(nn.Module):
    """Loss based on predicting the correct breakdown cycle."""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
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
        
        # Clamp probabilities for stability
        breakdown_prob_clamped = torch.clamp(breakdown_prob, min=self.eps, max=1-self.eps)
        
        # Compute focal loss
        ce_loss = -(target * torch.log(breakdown_prob_clamped) + 
                   (1 - target) * torch.log(1 - breakdown_prob_clamped))
        
        # Apply focusing factor
        pt = torch.where(target == 1, breakdown_prob_clamped, 1 - breakdown_prob_clamped)
        focal_weight = (1 - pt) ** gamma
        
        focal_loss = focal_weight * ce_loss
        
        # Mean over all positions
        loss = focal_loss.mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=device)
        
        return loss