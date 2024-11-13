import os
import torch
import tiktoken
from contextlib import nullcontext
from pathlib import Path
from .model import GPT, GPTConfig

class Inferencer:
    def __init__(self, 
                 model_path: str, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 dtype='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
                 compile=True):
        """Initialize inferencer with model configuration."""
        self.device = device
        self.dtype = dtype
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        
        # Load model and get config
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = GPTConfig(**checkpoint['model_args'])
        self.block_size = self.config.block_size
        
        # Setup device and autocast context
        self.device_type = 'cuda' if 'cuda' in device else 'cpu'
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        
        # Load model
        self.model = self.load_model(checkpoint)
        if compile:
            self.model = torch.compile(self.model)
        
        # Initialize tokenizer
        self.enc = tiktoken.get_encoding("cl100k_base")
        
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token ids."""
        tokens = self.enc.encode(text, allowed_special={"<|endoftext|>"})
        if len(tokens) > self.block_size:
            print(f"Warning: truncating input sequence from {len(tokens)} to {self.block_size} tokens")
            tokens = tokens[:self.block_size]
        return torch.tensor(tokens, dtype=torch.long, device=self.device)[None, ...]

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token ids to text."""
        return self.enc.decode(token_ids.tolist())

    @torch.no_grad()
    def generate(self, 
                prompt: str,
                max_new_tokens: int = 100,
                temperature: float = 0.8,
                top_k: int = 200,
                num_samples: int = 1) -> list[str]:
        """Generate text from a prompt."""
        # Handle file-based prompts
        if prompt.startswith('FILE:'):
            with open(prompt[5:], 'r', encoding='utf-8') as f:
                prompt = f.read()
        
        # Encode the prompt
        x = self.encode(prompt)
        
        # Generate samples
        outputs = []
        with self.ctx:
            for _ in range(num_samples):
                y = self.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                outputs.append(self.decode(y[0]))
        
        return outputs

    def load_model(self, checkpoint: dict) -> GPT:
        """Load the model from checkpoint."""
        try:
            # Initialize model with config
            model = GPT(self.config)
            
            # Get state dict and handle unwanted prefix
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            # Load state dict
            model.load_state_dict(state_dict)
            model.eval()
            model.to(self.device)
            
        except Exception as e:
            raise ValueError(f"Failed to load model weights: {e}")
        
        return model

def inference(model_path: str = "out/ckpt.pt", prompt: str = None):
    """Main inference function."""
    inferencer = Inferencer(model_path)
    
    if prompt is None:
        prompt = "Once upon a time"
    
    print("\nGenerating from prompt:", prompt)
    print("-" * 50)
    
    outputs = inferencer.generate(
        prompt,
        max_new_tokens=500,
        temperature=0.8,
        top_k=200,
        num_samples=1
    )
    
    for i, output in enumerate(outputs, 1):
        print(f"\nSample {i}:")
        print(output)
        print("-" * 50)
