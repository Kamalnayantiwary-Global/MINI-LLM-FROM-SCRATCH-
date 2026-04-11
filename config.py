import torch

# Settings for our Mini LLM
batch_size = 16    
block_size = 32    # Context length
max_iters = 3000   
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64        # Embedding size
n_head = 4         # Number of attention heads
n_layer = 4        # Number of transformer blocks
dropout = 0.1