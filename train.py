import torch.backends
import torch.nn as nn
import inspect
import torch
import torch.nn.functional as F
import numpy as np
import math
import os 

device = torch.device("cpu")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps" 
dataset_location = "fineweb10B"
print(f"Using device: {device}")
vocab_size = 50304 
weight_decay = 0.1
block_size = 1024
batch_size = 4 
epochs = 19073
total_batch_size = 524288
lr = 3e-4
max_steps = epochs
warmup_steps = 150
max_lr = lr
min_lr = max_lr * 0.1

assert total_batch_size % (batch_size * block_size) == 0, "total_batch_size must be divisble by B * T"
grad_accum_steps = total_batch_size // (batch_size * block_size)
print(grad_accum_steps)

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        data_root = dataset_location
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

class RMSNorm(nn.Module):
    def __init__(self, input_shape, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(input_shape))
        self.b = nn.Parameter(torch.ones(input_shape))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        output = x / rms 
        output = (output * self.g) + self.b
        return output 

# TODO: Review implementation of CausalSelfAttention
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        self.n_embd = n_embed
        self.n_head = n_head
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super(MLP, self).__init__()
        
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, 4 *  embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim), 
        )
        self.scale_init = 1
    def forward(self, x):
        return self.MLP(x)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.rmsnorm1 = RMSNorm(embed_dim) 
        self.MultiheadAttention = CausalSelfAttention(self.embed_dim, self.num_heads)
        self.rmsnorm2 = RMSNorm(embed_dim)
        self.MLP = MLP(embed_dim)
        
    def forward(self, x):
        x = x + self.MultiheadAttention(self.rmsnorm1(x))
        x = x + self.MLP(self.rmsnorm2(x))
        return x

class Model1(nn.Module):
    def __init__(self, block_size, vocab_size, embed_dim, num_heads, num_blocks, dropout):
        super(Model1, self).__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim  
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.positional_embedding = nn.Embedding(self.block_size, self.embed_dim)

        self.blocks = nn.ModuleList([Block(self.embed_dim, self.num_heads, self.dropout) for _ in range(self.num_blocks)])

        self.rmsnorm3 = RMSNorm(self.embed_dim)
        self.lm_linear = nn.Linear(self.embed_dim, self.vocab_size)
        
        #weight sharing scheme
        self.lm_linear.weight = self.embedding.weight

        # Apply parameter initialization as per GPT2 model
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.num_blocks) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x)
        x = x + self.positional_embedding(torch.arange(T, device=device))
        for block in self.blocks:
            x = block(x)
        output = self.rmsnorm3(x)
        output = self.lm_linear(output)
        return output


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

train_dataloader = DataLoaderLite(B=batch_size, T=block_size, split="train")
val_dataloader = DataLoaderLite(B=batch_size, T=block_size, split="val")

model = Model1(block_size=block_size, vocab_size=vocab_size, embed_dim=768, num_heads=16, num_blocks=8, dropout=0.2).to(device)
if device == "mps":
    pass
else:
    model = torch.compile(model)

fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device == "cuda"

loss = nn.CrossEntropyLoss()

param_dict = [p for p in model.parameters()]
param_dict = [p for p in param_dict if p.requires_grad]
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
decay_params = [p for p in param_dict if p.dim() >= 2]
nodecay_params = [p for p in param_dict if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optim_groups, lr=lr, fused=use_fused)
print("Parameter weight decay set")

for epoch in range(epochs):
    optimizer.zero_grad()
    loss_acum = 0.0
    
    for micro_step in range(grad_accum_steps):
        x, y = train_dataloader.next_batch()
        x, y = x.to(device), y.to("cpu")
        # x, y = x.to(device), y.to(device)
        output = model(x).to("cpu")
        # output = model(x)
        train_loss = loss(output.view(-1, output.size(-1)), y.view(-1))
        train_loss = train_loss.to(device) 
        output = output.to(device)

        train_loss = train_loss / grad_accum_steps
        loss_acum += train_loss.detach()
        train_loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

    output = output.to(device)
    
    # for param in model.parameters():
        # param.grad = None

    optimizer.step()
    
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    with torch.no_grad():
        x, y = val_dataloader.next_batch()
        x, y = x.to(device), y.to("cpu")
        output = model(x)
        output = output.to("cpu")
        val_loss = loss(output.view(-1, output.size(-1)), y.view(-1))

    print_statement = f"Epoch: {epoch}| Train Loss: {loss_acum.item()} | Val Loss: {val_loss.item()} | Norm: {norm} | lr: {lr}"

    with open("log.txt", 'a') as fd:
        fd.write(f"{print_statement}\n")

    print(print_statement)