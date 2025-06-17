# physics/kmc_physics.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

class KineticMonteCarloPhysics:
    """Physics equations and constraints from kinetic Monte Carlo simulations."""
    
    def __init__(self, config: Optional[Dict] = None):
        # Default configuration
        if config is None:
            config = {
                'constants': {
                    'k_B': 8.617e-5,  # eV/K
                    'q': 1.602e-19    # C
                },
                'device': {
                    'temperature_default': 300.0  # K
                }
            }
        
        # Extract constants with defaults
        constants = config.get('constants', {})
        device_config = config.get('device', {})
        
        self.k_B = constants.get('k_B', 8.617e-5)  # eV/K
        self.q = constants.get('q', 1.602e-19)  # C
        self.temperature = device_config.get('temperature_default', 300.0)  # K
    
    def generation_rate(
        self,
        electric_field: torch.Tensor,
        activation_energy: torch.Tensor,
        dipole_moment: float = 1e-29
    ) -> torch.Tensor:
        """
        Calculate defect generation rate based on thermochemical model.
        
        G = ν * exp(-(E_A - p*E)/(k_B*T))
        
        Args:
            electric_field: Electric field (V/m)
            activation_energy: Activation energy (eV)
            dipole_moment: Dipole moment (C*m)
            
        Returns:
            Generation rate (1/s)
        """
        nu = 1e13  # Attempt frequency (Hz)
        
        # Ensure positive electric field
        electric_field = torch.abs(electric_field)
        
        # Field-lowered barrier
        barrier = activation_energy - dipole_moment * electric_field / self.q
        
        # Ensure positive barrier with reasonable bounds
        barrier = torch.clamp(barrier, min=0.1, max=10.0)
        
        # Calculate rate with numerical stability
        exponent = -barrier / (self.k_B * self.temperature)
        exponent = torch.clamp(exponent, min=-50.0, max=50.0)  # Prevent overflow
        
        rate = nu * torch.exp(exponent)
        
        # Clamp rate to reasonable physical bounds
        rate = torch.clamp(rate, min=1e-10, max=1e20)
        
        return rate

        
    def trap_assisted_tunneling_rate(
        self,
        trap_energy: torch.Tensor,
        electric_field: torch.Tensor,
        distance: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate trap-assisted tunneling rate.
        
        Based on multi-phonon trap-assisted tunneling theory.
        """
        # Simplified TAT rate calculation
        # In reality, this involves WKB approximation and phonon coupling
        
        # Field-enhanced emission
        field_factor = torch.sqrt(electric_field / 1e6)  # Normalize to MV/cm
        
        # Energy-dependent rate
        energy_factor = torch.exp(-trap_energy / (self.k_B * self.temperature))
        
        # Distance-dependent tunneling
        tunneling_factor = torch.exp(-2 * distance / 1e-9)  # Normalize to nm
        
        rate = 1e10 * field_factor * energy_factor * tunneling_factor
        
        return rate
        
    def poole_frenkel_emission(
        self,
        trap_depth: torch.Tensor,
        electric_field: torch.Tensor,
        dielectric_constant: float = 3.9
    ) -> torch.Tensor:
        """
        Calculate Poole-Frenkel emission rate.
        
        Rate = ν * exp(-(φ - β*sqrt(E))/(k_B*T))
        where β = sqrt(q³/(π*ε*ε₀))
        """
        nu = 1e13  # Attempt frequency
        epsilon_0 = 8.854e-12  # F/m
        
        # Poole-Frenkel constant
        beta = np.sqrt(self.q**3 / (np.pi * dielectric_constant * epsilon_0))
        
        # Field-lowered barrier
        barrier_lowering = beta * torch.sqrt(electric_field)
        effective_barrier = trap_depth - barrier_lowering
        
        # Emission rate
        rate = nu * torch.exp(-effective_barrier / (self.k_B * self.temperature))
        
        return rate
        
    def diffusion_coefficient(
        self,
        activation_energy: torch.Tensor,
        prefactor: float = 1e-4
    ) -> torch.Tensor:
        """
        Calculate diffusion coefficient for oxygen vacancies/interstitials.
        
        D = D₀ * exp(-E_a/(k_B*T))
        """
        return prefactor * torch.exp(-activation_energy / (self.k_B * self.temperature))
        
    def breakdown_probability(
        self,
        defect_density: torch.Tensor,
        critical_density: float = 1e22
    ) -> torch.Tensor:
        """
        Calculate breakdown probability based on percolation theory.
        
        When defect density approaches critical density, 
        conductive paths form leading to breakdown.
        """
        # Sigmoid function for smooth transition
        x = (defect_density - critical_density) / (0.1 * critical_density)
        prob = torch.sigmoid(x)
        
        return prob


class PhysicsConstraints(nn.Module):
    """Enforces physical constraints in the model."""
    
    def __init__(self, breakdown_threshold: float = 200.0):
        super().__init__()
        self.breakdown_threshold = breakdown_threshold
        
    def enforce_monotonicity(self, states: torch.Tensor) -> torch.Tensor:
        """Ensure defect counts only increase (no healing)."""
        if states.size(1) <= 1:
            return states
            
        # Calculate differences
        diffs = states[:, 1:] - states[:, :-1]
        
        # Force non-negative differences
        diffs = torch.clamp(diffs, min=0)
        
        # Reconstruct states
        corrected_states = torch.zeros_like(states)
        corrected_states[:, 0] = states[:, 0]
        
        for i in range(1, states.size(1)):
            corrected_states[:, i] = corrected_states[:, i-1] + diffs[:, i-1]
            
        return corrected_states
        
    def enforce_breakdown_limit(self, states: torch.Tensor) -> torch.Tensor:
        """Ensure states don't exceed breakdown threshold."""
        return torch.clamp(states, max=self.breakdown_threshold)
        
    def calculate_electric_field(
        self,
        voltage: torch.Tensor,
        thickness: torch.Tensor
    ) -> torch.Tensor:
        """Calculate electric field from voltage and thickness."""
        return torch.abs(voltage) / thickness
        
    def temperature_scaling(
        self,
        rate: torch.Tensor,
        temperature: torch.Tensor,
        reference_temp: float = 300.0,
        activation_energy: float = 1.3
    ) -> torch.Tensor:
        """
        Scale rates based on temperature using Arrhenius law.
        
        rate(T) = rate(T_ref) * exp(E_a/k_B * (1/T_ref - 1/T))
        """
        k_B = 8.617e-5  # eV/K
        
        scaling = torch.exp(
            activation_energy / k_B * (1/reference_temp - 1/temperature)
        )
        
        return rate * scaling


class DefectEvolutionModel(nn.Module):
    """Physics-based defect evolution model."""
    
    def __init__(
        self,
        physics_config: Optional[Dict] = None,
        use_neural_correction: bool = True
    ):
        super().__init__()
        
        # Default configuration
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
            
        self.physics = KineticMonteCarloPhysics(physics_config)
        self.constraints = PhysicsConstraints()
        self.use_neural_correction = use_neural_correction
        
        if use_neural_correction:
            # Small neural network for correction terms
            self.correction_net = nn.Sequential(
                nn.Linear(5, 32),  # Input: E-field, trap params, current state
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus()  # Positive correction
            )
            
    def forward(
        self,
        current_state: torch.Tensor,
        voltage: torch.Tensor,
        thickness: torch.Tensor,
        trap_params: torch.Tensor,
        time_step: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate next state based on physics.
        
        Returns:
            next_state: Predicted next defect count
            physics_info: Dictionary with intermediate physics quantities
        """
        # Calculate electric field
        e_field = self.constraints.calculate_electric_field(voltage, thickness)
        
        # Extract relevant trap parameters
        activation_energy = trap_params[:, 3]  # Relaxation energy
        
        # Calculate generation rate
        gen_rate = self.physics.generation_rate(e_field, activation_energy)
        
        # Base physics prediction
        delta_defects = gen_rate * time_step
        
        # Apply neural correction if enabled
        if self.use_neural_correction:
            # Prepare inputs for correction network
            correction_input = torch.stack([
                e_field / 1e6,  # Normalize to MV/cm
                trap_params[:, 0] / 1e19,  # Normalize peak density
                trap_params[:, 1],  # Thermal ionization mean
                trap_params[:, 2],  # Thermal ionization spread
                current_state.squeeze() / 200.0  # Normalize by threshold
            ], dim=-1)
            
            correction = self.correction_net(correction_input).squeeze()
            delta_defects = delta_defects * correction
            
        # Calculate next state
        next_state = current_state + delta_defects.unsqueeze(-1)
        
        # Apply constraints
        next_state = self.constraints.enforce_monotonicity(
            torch.cat([current_state.unsqueeze(1), next_state.unsqueeze(1)], dim=1)
        )[:, -1:]
        
        next_state = self.constraints.enforce_breakdown_limit(next_state)
        
        # Collect physics info
        physics_info = {
            'electric_field': e_field,
            'generation_rate': gen_rate,
            'delta_defects': delta_defects,
            'breakdown_prob': self.physics.breakdown_probability(
                next_state.squeeze() * 1e20 / 1e-12  # Convert to density
            )
        }
        
        return next_state, physics_info