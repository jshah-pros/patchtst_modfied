import torch
from model.patchtst import PatchTST

# Model parameters
config = {
    'enc_in': 512,
    'seq_len': 512,
    'pred_len': 24,
    'patch_len': 16,
    'stride': 8,
    'padding_patch': 'end',
    'e_layers': 2,
    'n_heads': 8,
    'd_model': 512,
    'd_ff': 2048,
    'dropout': 0.05,
    'fc_dropout': 0.05,
    'head_dropout': 0.0,
    'individual': 0,
    
}       

# Initialize Model
patchTST = PatchTST(config)

# Test
x = torch.rand(128, 512, 1)
y = torch.rand(128, 24, 1)
y_hat = patchTST(x)

print(y_hat.shape) # [BS, Output Length, Channels]
assert y.shape == y_hat.shape