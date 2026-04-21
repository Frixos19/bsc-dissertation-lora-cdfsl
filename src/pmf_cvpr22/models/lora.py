import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r, lora_alpha=1):
        super().__init__()
  
        # store the original frozen linear layer
        self.linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # freeze the original weights
        for param in original_linear.parameters():
            param.requires_grad = False
     
        # create lora_A and lora_B as nn.Parameters
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        # compute scaling 
        self.scaling = lora_alpha / r
        
        # initialise
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.merged = False

    def merge_weights(self):
        # absorb LoRA update into frozen weight: W' = W₀ + scaling * B @ A
        # after this, forward is a plain linear — no extra computation
        with torch.no_grad():
            self.linear.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()
        self.merged = True
        
    def reset(self):
        self.merged = False
        
    def forward(self, x):
        # compute the original linear output (frozen)
        result = self.linear(x)

        if self.merged:
            return result

        # compute the lora update: x @ lora_A.T @ lora_B.T * scaling
        DeltaW = x @ self.lora_A.T @ self.lora_B.T * self.scaling

        return result + DeltaW
        # return their sum

def inject_lora(model, r, lora_alpha=1, targets=('qkv',)):
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
    # walk all modules and merge any LoRALinear layers
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()

def reset_lora(model):
    # loop through all modules in the model
    # if a module is a LoRALinear, call its reset() method
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset()

def eject_lora(model):
    for block in model.blocks:
        if isinstance(block.attn.qkv, LoRALinear):
            block.attn.qkv = block.attn.qkv.linear
        if isinstance(block.attn.proj, LoRALinear):
            block.attn.proj = block.attn.proj.linear
        if isinstance(block.mlp.fc1, LoRALinear):
            block.mlp.fc1 = block.mlp.fc1.linear
        if isinstance(block.mlp.fc2, LoRALinear):
            block.mlp.fc2 = block.mlp.fc2.linear