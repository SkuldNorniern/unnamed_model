"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) The official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) Hugging Face's transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import LayerNorm, GELU, RMSNorm
from torch.nn import functional as F
from safetensors.torch import save_file, load_file  # Import missing functions
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from .activation import Swish  # Add this import at the top of the file
from .layers import  MiniRMSNorm
from .attention import CausalSelfAttention





class MLP(nn.Module):
    """Enhanced feed-forward neural network module with configurable expansion."""
    
    def __init__(self, config):
        super().__init__()
        self.expansion_factor = getattr(config, 'mlp_expansion_factor', 4)
        hidden_dim = int(config.n_embd * self.expansion_factor)
        
        self.fc1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.fc2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        
        # Use actual GELU activation
        self.act = Swish() # Using tanh approximation for better performance
        self.dropout = nn.Dropout(config.dropout)
        
        # Proper initialization
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with scaled normal distribution
        std = 0.02
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=std)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block consisting of attention and MLP layers."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    """The GPT Language Model with Adapters."""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # Initialize weights
        self.apply(self._init_weights)
        # Scaled initialization for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        # Forward the GPT model
        tok_emb = self.transformer.wte(idx)  # Token embeddings
        pos_emb = self.transformer.wpe(pos)  # Position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            # Calculate loss if targets are provided
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None
        return logits, loss
    # def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     # Use gradient checkpointing
    #     device = idx.device
    #     b, t = idx.size()
    #     assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

    #     # Embedding and positional encoding
    #     tok_emb = self.transformer.wte(idx)  # Token embeddings
    #     pos_emb = self.transformer.wpe(torch.arange(0, t, dtype=torch.long, device=device))  # Position embeddings
    #     x = self.transformer.drop(tok_emb + pos_emb)

    #     # Define a function for checkpointing
    #     def custom_forward(module):
    #         def forward(*inputs):
    #             return module(*inputs)
    #         return forward

    #     for block in self.transformer.h:
    #         x = torch.utils.checkpoint.checkpoint(custom_forward(block), x)

    #     x = self.transformer.ln_f(x)
    #     logits = self.lm_head(x)

    #     if targets is not None:
    #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    #     else:
    #         loss = None
    #     return logits, loss


    def crop_block_size(self, block_size):
        """Adjust the model's block size."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def save_pretrained(self, save_directory):
        """Save the model to a directory using safetensors."""
        state_dict = self.state_dict()
        save_file(state_dict, f"{save_directory}/model.safetensors")
        config_dict = self.config.__dict__
        torch.save(config_dict, f"{save_directory}/config.pt")

    @classmethod
    def load_from_checkpoint(cls, model_path, override_args=None):
        """Load a model from a directory using safetensors."""
        config_dict = torch.load(f"{model_path}/config.pt")
        config = GPTConfig(**config_dict)
        if override_args:
            for k, v in override_args.items():
                setattr(config, k, v)
        model = cls(config)
        state_dict = load_file(f"{model_path}/model.safetensors")
        model.load_state_dict(state_dict)
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure the optimizer with weight decay for certain parameters."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU)."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, 
                 temperature=0.7,          # Reduced temperature for more focused output
                 top_k=50,                # Added default top-k
                 top_p=0.9,
                 repetition_penalty=1.2,
                 presence_penalty=0.3,     # New: penalize tokens based on presence in history
                 frequency_penalty=0.3,    # New: penalize tokens based on frequency
                 min_tokens=3,            # New: minimum tokens between repetitions
                 eos_token_id=None):      # New: optional end token
        """
        Enhanced text generation with multiple sampling strategies and penalties.
        """
        self.eval()
        generated = []
        token_frequencies = {}  # Track token frequencies
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(idx_cond.shape[1]):
                    token = idx_cond[0, i].item()
                    logits[0, token] /= repetition_penalty
                    
                    # Additional minimum distance between repeated tokens
                    if token in generated[-min_tokens:]:
                        logits[0, token] = -float('inf')
            
            # Apply presence and frequency penalties
            for token in set(idx_cond[0].tolist()):
                freq = token_frequencies.get(token, 0)
                logits[0, token] -= presence_penalty + (freq * frequency_penalty)
            
            # Temperature scaling
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) filtering with improved stability
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Ensure we keep at least one token
                if sorted_indices_to_remove.all():
                    sorted_indices_to_remove[..., 0] = 0
                    
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Update token frequencies
            token = idx_next.item()
            token_frequencies[token] = token_frequencies.get(token, 0) + 1
            generated.append(token)
            
            # Check for EOS token
            if eos_token_id and token == eos_token_id:
                break
                
            idx = torch.cat((idx, idx_next), dim=1)

        return idx