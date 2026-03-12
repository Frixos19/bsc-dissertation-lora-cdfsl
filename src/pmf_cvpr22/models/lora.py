import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    def __init__(self, original_linear, r, lora_alpha=1):
        super().__init__()
  
       
        # store the original frozen linear layer
        # hint: you need in_features and out_features from it

        self.linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # freeze the original weights
        # hint: requires_grad

        for param in original_linear.parameters():
            param.requires_grad = False
     
        # create lora_A and lora_B as nn.Parameters
        # hint: lora_A shape is (r, in_features)
        # hint: lora_B shape is (out_features, r)
        
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        # compute scaling = lora_alpha / r

        self.scaling = lora_alpha / r
        
        # initialise: lora_A kaiming uniform, lora_B zeros

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def reset(self):
        # reinitialise lora_A and lora_B back to starting values
        # same init as __init__
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        

    def forward(self, x):
        # compute the original linear output (frozen)

        result = self.linear(x)

        # compute the lora update: x @ lora_A.T @ lora_B.T * scaling
        
        DeltaW = x @ self.lora_A.T @ self.lora_B.T * self.scaling

        return result + DeltaW

        # return their sum
        


def inject_lora(model, r, lora_alpha=1):
    # loop through all blocks in model.blocks

    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, r, lora_alpha)
    # for each block, replace block.attn.qkv with LoRALinear(block.attn.qkv, r)



def reset_lora(model):
    # loop through all modules in the model
    # if a module is a LoRALinear, call its reset() method

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.reset()