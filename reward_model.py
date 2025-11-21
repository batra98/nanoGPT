"""
Reward Model for RLHF.

Architecture: GPT (frozen) + MLP reward head → scalar reward
"""

import torch
import torch.nn as nn
from model import GPT


class RewardModel(nn.Module):
    """
    Reward model that takes text and outputs a scalar reward.
    
    Architecture:
        1. GPT model (frozen) to get representations
        2. Mean pooling over sequence
        3. MLP head to predict scalar reward
    """
    
    def __init__(self, gpt_model: GPT, freeze_gpt: bool = True):
        """
        Initialize reward model.
        
        Args:
            gpt_model: Pretrained GPT model
            freeze_gpt: Whether to freeze GPT weights (default: True)
        """
        super().__init__()
        
        self.gpt = gpt_model
        self.n_embd = gpt_model.config.n_embd
        
        # Freeze GPT if requested
        if freeze_gpt:
            for param in self.gpt.parameters():
                param.requires_grad = False
        
        # Reward head: simple MLP
        self.reward_head = nn.Sequential(
            nn.Linear(self.n_embd, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Output scalar reward
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize reward head weights."""
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, idx: torch.Tensor, return_embeddings: bool = False):
        """
        Forward pass through reward model.
        
        Args:
            idx: Token indices of shape (batch_size, sequence_length)
            return_embeddings: If True, also return the pooled embeddings
        
        Returns:
            rewards: Scalar rewards of shape (batch_size,)
            embeddings (optional): Pooled embeddings of shape (batch_size, n_embd)
        """
        # Get GPT representations (no loss calculation)
        with torch.set_grad_enabled(not self.training or any(p.requires_grad for p in self.gpt.parameters())):
            # Forward through GPT
            device = idx.device
            b, t = idx.size()
            assert t <= self.gpt.config.block_size, f"Sequence length {t} exceeds block size {self.gpt.config.block_size}"
            
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            
            # Get embeddings
            tok_emb = self.gpt.transformer.wte(idx)
            pos_emb = self.gpt.transformer.wpe(pos)
            x = self.gpt.transformer.drop(tok_emb + pos_emb)
            
            # Forward through transformer blocks
            for block in self.gpt.transformer.h:
                x = block(x)
            
            x = self.gpt.transformer.ln_f(x)
            # x is now (batch_size, seq_len, n_embd)
        
        # Mean pooling over sequence dimension
        pooled = x.mean(dim=1)  # (batch_size, n_embd)
        
        # Get reward from reward head
        rewards = self.reward_head(pooled).squeeze(-1)  # (batch_size,)
        
        if return_embeddings:
            return rewards, pooled
        else:
            return rewards
    
    def get_reward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to get rewards (eval mode).
        
        Args:
            idx: Token indices
        
        Returns:
            Scalar rewards
        """
        self.eval()
        with torch.no_grad():
            return self.forward(idx)


def create_reward_model(gpt_checkpoint_path: str, device: str = 'cuda', freeze_gpt: bool = True) -> RewardModel:
    """
    Create a reward model from a GPT checkpoint.
    
    Args:
        gpt_checkpoint_path: Path to GPT checkpoint
        device: Device to load model on
        freeze_gpt: Whether to freeze GPT weights
    
    Returns:
        RewardModel instance
    """
    from model import GPTConfig
    
    # Load checkpoint
    checkpoint = torch.load(gpt_checkpoint_path, map_location=device)
    
    # Create GPT model
    gptconf = GPTConfig(**checkpoint['model_args'])
    gpt = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    gpt.load_state_dict(state_dict)
    gpt.to(device)
    
    # Create reward model
    reward_model = RewardModel(gpt, freeze_gpt=freeze_gpt)
    reward_model.to(device)
    
    print(f"Created reward model:")
    print(f"  GPT parameters: {sum(p.numel() for p in gpt.parameters()) / 1e6:.2f}M")
    print(f"  Reward head parameters: {sum(p.numel() for p in reward_model.reward_head.parameters()) / 1e6:.2f}M")
    print(f"  GPT frozen: {freeze_gpt}")
    
    return reward_model


if __name__ == '__main__':
    # Test reward model creation
    print("Testing reward model...")
    
    # Create a small GPT for testing
    from model import GPTConfig
    
    config = GPTConfig(
        block_size=256,
        vocab_size=50257,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.0,
        bias=False
    )
    
    gpt = GPT(config)
    reward_model = RewardModel(gpt, freeze_gpt=True)
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    dummy_input = torch.randint(0, 50257, (batch_size, seq_len))
    
    rewards = reward_model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Reward shape: {rewards.shape}")
    print(f"  Reward values: {rewards}")
    
    # Test with embeddings
    rewards, embeddings = reward_model(dummy_input, return_embeddings=True)
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in reward_model.parameters())
    trainable_params = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    print(f"  Frozen: {(total_params - trainable_params) / 1e6:.2f}M")
    
    print("\n✓ Reward model test passed!")

