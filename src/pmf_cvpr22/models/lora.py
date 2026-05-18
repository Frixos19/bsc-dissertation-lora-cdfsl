import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update: h = W₀x + BAx * (alpha/r)"""
    def __init__(self, original_linear, r, lora_alpha=1):
        super().__init__()
  
        # store the original frozen linear layer
        self.linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # freeze the original weights - only lora_A and lora_B are trainable
        for param in original_linear.parameters():
            param.requires_grad = False
     
        # create lora_A and lora_B as nn.Parameters
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        # alpha/r keeps update magnitude stable across ranks
        self.scaling = lora_alpha / r
        
        # kaiming uniform for A, zeros for B - ensures ΔW=BA=0 at init
        # follows official LoRA implementation rather than Gaussian in paper
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.merged = False

    def merge_weights(self):
        """Absorbs LoRA update into W0 before query inference -> zero overhead at test time"""
        # absorb LoRA update into frozen weight: W' = W0 + scaling * B @ A
        # after this, forward is a plain linear - no extra computation
        with torch.no_grad():
            self.linear.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()
        self.merged = True
        
    def reset(self):
        # merged flag not restored by load_state_dict - needs explicit reset
        self.merged = False
        
    def forward(self, x):
        # compute the original linear output (frozen)
        result = self.linear(x)

        if self.merged:
            return result

        # compute the lora update: x @ lora_A.T @ lora_B.T * scaling
        DeltaW = x @ self.lora_A.T @ self.lora_B.T * self.scaling

        # return their sum
        return result + DeltaW        

def inject_lora(model, r, lora_alpha=1, targets=('qkv',)):
    """Freezes backbone and injects LoRALinear into target modules across all 12 transformer blocks"""
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        if 'qkv' in targets:
            device = block.attn.qkv.weight.device
            block.attn.qkv = LoRALinear(block.attn.qkv, r, lora_alpha).to(device)
        if 'proj' in targets:
            device = block.attn.proj.weight.device
            block.attn.proj = LoRALinear(block.attn.proj, r, lora_alpha).to(device)
        if 'mlp' in targets:
            device = block.mlp.fc1.weight.device
            block.mlp.fc1 = LoRALinear(block.mlp.fc1, r, lora_alpha).to(device)
            device = block.mlp.fc2.weight.device
            block.mlp.fc2 = LoRALinear(block.mlp.fc2, r, lora_alpha).to(device)

def merge_lora(model):
    """Merges all LoRA updates into frozen weights before query inference"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()

def reset_lora(model):
    """Resets merged flag on all LoRALinear modules at the start of each episode"""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset()

def eject_lora(model):
    """Removes LoRALinear wrappers entirely — used in adaptive LoRA to swap ranks between candidates"""
    for block in model.blocks:
        if isinstance(block.attn.qkv, LoRALinear):
            block.attn.qkv = block.attn.qkv.linear
        if isinstance(block.attn.proj, LoRALinear):
            block.attn.proj = block.attn.proj.linear
        if isinstance(block.mlp.fc1, LoRALinear):
            block.mlp.fc1 = block.mlp.fc1.linear
        if isinstance(block.mlp.fc2, LoRALinear):
            block.mlp.fc2 = block.mlp.fc2.linear