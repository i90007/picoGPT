"""
This training script for running on a single gpu 
"""
import os
from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
batch_size = 8 # max for T4 in Google Colab
real_data = True
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
max_iters = 600000 # total number of training iterations
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024 # how far back does the model look? i.e. context size
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12 # size of the model
    n_head: int = 12 # size of the model
    n_embd: int = 768 # size of the model
    dropout: float = 0.1 # for determinism
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    max_knn_memories: int = 81920
    topk_retrieved_memories: int = 3
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading init
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# model init
model = GPT(GPTConfig())
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0

# simple benchmarking
torch.cuda.synchronize()
iter_num = 0
while True:
    t0 = time.time()
    X, Y = get_batch('train')
    for k in range(num_steps):
        with ctx:
            logits, loss = model(X, Y)
        X, Y = get_batch('train')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lossf = loss.item()
        print(f"{k}/{num_steps} loss: {lossf:.4f}")
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
    if stage == 1:
        print(f"iter {stage}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    # termination conditions
    if iter_num > max_iters:
        break
