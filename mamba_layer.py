"""
Simplified Mamba Block Implementation

This is a simplified version of the Mamba (Selective State Space Model) architecture
from "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2024).

Key simplifications:
- Uses naive PyTorch implementation (no hardware-aware optimizations)
- Sequential scan instead of parallel scan
- Basic selective mechanism without complex discretization schemes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_sequential(x, dt, A, B, C, D):
    """
    Simplified selective scan operation (sequential, not parallel).
    
    Args:
        x: Input tensor (batch, seq_len, d_model)
        dt: Step size (batch, seq_len, d_model)
        A: State transition matrix (d_model, d_state)
        B: Input projection (batch, seq_len, d_state)
        C: Output projection (batch, seq_len, d_state)
        D: Skip connection parameter (d_model,)
    
    Returns:
        Output tensor (batch, seq_len, d_model)
    """
    batch, seq_len, d_model = x.shape
    d_state = B.shape[-1]
    
    # Initialize state
    h = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
    
    outputs = []
    
    for t in range(seq_len):
        # Discretize: h_t = exp(dt * A) * h_{t-1} + dt * B_t * x_t
        # Simplified: we use element-wise operations
        dt_t = dt[:, t, :]  # (batch, d_model)
        A_t = A.unsqueeze(0)  # (1, d_model, d_state)
        
        # Expand dt for broadcasting
        dt_expanded = dt_t.unsqueeze(-1)  # (batch, d_model, 1)
        
        # State transition: h = exp(dt * A) * h + B * (dt * x)
        # For numerical stability, we use the approximation: exp(dt * A) ≈ 1 + dt * A for small dt
        decay = torch.exp(dt_expanded * A_t)  # (batch, d_model, d_state)
        
        # Update state
        h_per_channel = h.unsqueeze(1) * decay  # (batch, d_model, d_state)
        
        # Add input contribution
        x_t = x[:, t, :]  # (batch, d_model)
        B_t = B[:, t, :]  # (batch, d_state)
        
        # Input contribution: sum over input dimensions
        input_contrib = (dt_t * x_t).unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, d_model, d_state)
        
        h_per_channel = h_per_channel + input_contrib
        
        # Pool back to shared state (average across channels)
        h = h_per_channel.mean(dim=1)  # (batch, d_state)
        
        # Output: y_t = C_t * h_t + D * x_t
        C_t = C[:, t, :]  # (batch, d_state)
        y_t = torch.einsum('bd,bd->b', C_t, h).unsqueeze(-1) + D * x_t  # (batch, d_model)
        
        outputs.append(y_t)
    
    return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)


class MambaBlock(nn.Module):
    """
    Simplified Mamba block with selective state space model.
    
    Architecture:
    1. Expand input dimension via linear projection
    2. Apply 1D convolution for local context
    3. Selective SSM with data-dependent parameters (B, C, dt)
    4. Gated output with SiLU activation
    5. Project back to model dimension
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config.n_embd
        self.d_state = getattr(config, 'd_state', 16)  # SSM state dimension
        self.d_conv = getattr(config, 'd_conv', 4)  # Conv kernel size
        self.expand = 2  # Expansion factor
        
        self.d_inner = self.d_model * self.expand
        
        # Input projection (expand dimension)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 1D convolution for local context
        # Using groups=d_inner makes it a depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,  # Depthwise convolution
            bias=True
        )
        
        # SSM parameters
        # A: State transition matrix (diagonal for stability)
        # Initialize with negative values for stable dynamics
        A = torch.randn(self.d_inner, self.d_state)
        A = -torch.exp(A)  # Ensure negative eigenvalues
        self.A = nn.Parameter(A)
        
        # Selective projections (data-dependent B, C, dt)
        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Skip connection parameter (D in paper)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # 1. Input projection and split for gating
        x_and_gate = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_proj, gate = x_and_gate.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # 2. 1D Convolution for local context
        # Conv1d expects (batch, channels, length)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :seq_len]  # (batch, d_inner, seq_len)
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # 3. Activation
        x_conv = F.silu(x_conv)
        
        # 4. Selective SSM
        # Compute data-dependent parameters
        B = self.B_proj(x_conv)  # (batch, seq_len, d_state)
        C = self.C_proj(x_conv)  # (batch, seq_len, d_state)
        dt = self.dt_proj(x_conv)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)  # Ensure positive step sizes
        
        # Apply selective scan
        y = selective_scan_sequential(x_conv, dt, self.A, B, C, self.D)
        
        # 5. Gated output
        y = y * F.silu(gate)
        
        # 6. Output projection
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y

