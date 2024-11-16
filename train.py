"""
The training script for running on a single gpu
Little logs:
1) block_size = 512
tokens per iteration will be: 7,680
number of parameters: 118.96M
iter 187: loss 6.3322
For T4 on Google Colab (tinyshakespeare) should be: gradient_accumulation_steps = 5 and batch_size = 3
2) ...
"""
import os
import time
import math
from contextlib import nullcontext
import numpy as np

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# import sys
# sys.exit()
from model import MemorizingGPT
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 20 # 2000 for openwebtext, 20 for tinyshakespeare
eval_iters = 2 # 200 for openwebtext, 2 for tinyshakespeare
init_from = 'scratch' # 'scratch' or 'resume'
# data
dataset = 'tinyshakespeare' # 'openwebtext'
gradient_accumulation_steps = 5 # used to simulate larger batch sizes (5 for tinyshakespeare and TP4 on Google Colab)
batch_size = 3 # if gradient_accumulation_steps > 1, this is the micro-batch size
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 100 # total number of training iterations (600000 for openwebtext, 200 for tinyshakespeare)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2 # how many steps to warm up for (2000 for openwebtext, 2 for tinyshakespeare)
lr_decay_iters = 100 # should be ~= max_iters per Chinchilla (600000 for openwebtext, 200 for tinyshakespeare)
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# attempt to autodetect device
device = "cpu" # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True if device == "cuda" else False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
class GPTConfig:
    block_size: int = 1024 # (1024) how far back does the model look? i.e. context size
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12 # size of the model
    n_head: int = 12 # size of the model
    n_embd: int = 768 # size of the model
    dropout: float = 0.1 # for determinism
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    max_knn_memories: bool = 130943 # the maximum number of memories that will be stored locally

# we are running on a single gpu, and one process
tokens_per_iter = gradient_accumulation_steps * batch_size * GPTConfig.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - GPTConfig.block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+GPTConfig.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+GPTConfig.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

    else:
        x, y = x.to(device), y.to(device)

    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = MemorizingGPT(GPTConfig)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    # create the model
    model = MemorizingGPT(GPTConfig)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# crop down the model block size if desired, using model surgery
if GPTConfig.block_size < model.config.block_size:
    model.crop_block_size(GPTConfig.block_size)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device, enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        model.knn.clear()
        for k in range(eval_iters):
            print(f'Estimate_loss - {k} from {eval_iters}')
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num
                }
                print(f"saving checkpoint to {out_dir}")
                checkpoint_loss = round(best_val_loss * 10000)
                torch.save(checkpoint, os.path.join(out_dir, f"val_loss__0_{checkpoint_loss}.pt"))

    # Clear XL memories
    logits = None
    # Clear KNN memory
    model.knn.clear()
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y, xl_memories=logits)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    # get loss as float. note: this is a CPU-GPU sync point
    # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
    lossf = loss.item() * gradient_accumulation_steps
    print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1
    local_iter_num += 1
    # termination conditions
    if iter_num > max_iters:
        break
