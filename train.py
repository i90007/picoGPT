"""
The training script for running on a single gpu
Little logs:
1) block_size = 512
dropout = 0.5
number of parameters: 759.66M
iter 100: loss 6.3455
For T4 on Google Colab should be: gradient_accumulation_steps = 2 and batch_size = 1
--- after refactoring and optimization ---
2) 
"""
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import glob
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from model import MemorizingGPT, CastedLinear

init_from = 'scratch' # 'scratch' or 'resume'
dataset = 'tinyshakespeare' # openwebtext, tinyshakespeare
# attempt to autodetect device
device = "cuda"
if not torch.cuda.is_available():
    print("Trainig possible with CUDA only!")
    os.sys.exit()
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size : int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer : int = 48 # size of the model (48, 36, 24, 12)
    n_head : int = 25 # size of the model (25, 20, 16, 12)
    n_embd: int = 1600 # size of the model (1600, 1280, 1024, 768)
    dropout: float = 0.5 # for determinism
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    max_knn_memories: bool = 130943 # the maximum number of memories that will be stored locally
configGpt = GPTConfig()
@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = f'data/{dataset}/train*.bin' # input .bin to train on
    input_val_bin : str = f'data/{dataset}/val*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 2 # batch size, in sequences, across all devices
    device_batch_size : int = 1 # batch size, in sequences, per device
    sequence_length : int = 64*1024 # sequence length, in tokens
    num_iterations : int = 180 # number of iterations to run (180 for tinyshakespeare, 1800 for openwebtext)
    warmup_iters : int = 1 # (1 for tinyshakespeare, 10 for openwebtext)
    cooldown_iters : int = 64 # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule (64 for tinyshakespeare, 640 for openwebtext)
    # evaluation and logging hyperparams
    val_loss_every : int = 10 # every how many steps to evaluate val loss? 0 for only at the end (10 for tinyshakespeare, 100 for openwebtext)
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()
# we are running on a single gpu, and one process
tokens_per_iter = args.batch_size * args.device_batch_size * args.sequence_length
print(f"tokens per iteration will be: {tokens_per_iter:,}")
# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader
def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        batch_size = self.B * self.T * self.num_processes
        buf = self.tokens[self.current_position:self.current_position+self.B*self.T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1] # inputs
        y = buf[1:] # targets
        # advance current position and load next shard if necessary
        self.current_position += batch_size
        if self.current_position + batch_size >= len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# convenience variables
T = args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % T == 0
val_steps = args.val_tokens // T
# load tokens
train_loader = DistributedDataLoader(args.input_bin, T, 0, 1)
val_loader = DistributedDataLoader(args.input_val_bin, T, 0, 1)
print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
print('='*100, logonly=True)
x, y = train_loader.next_batch()
# -----------------------------------------------------------------------------
# Muon optimizer
def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ['WORLD_SIZE']) == int(os.environ['RANK']):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# model init
model_args = dict(n_layer=configGpt.n_layer, n_head=configGpt.n_head, n_embd=configGpt.n_embd, 
    bias=configGpt.bias, vocab_size=configGpt.vocab_size, dropout=configGpt.dropout, max_knn_memories=configGpt.max_knn_memories)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = MemorizingGPT(GPTConfig())
elif init_from == 'resume':
    print(f"Resuming training from 'out' directory")
    # resume training from a checkpoint.
    ckpt_path = os.path.join('out', 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    for k in ['n_layer', 'n_head', 'n_embd', 'dropout', 'bias', 'vocab_size', 'max_knn_memories']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = MemorizingGPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
model.to(device)

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
num_vocab = 50304
model = MemorizingGPT(GPTConfig())
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True
# compile the model
print("compiling the model... (takes a ~minute)")
model = torch.compile(model)

# init the optimizer(s)
optimizer1 = torch.optim.Adam([model.transformer.wte.weight], lr=0.6,   betas=(0.9, 0.95), fused=True)
optimizer2 = torch.optim.Adam([model.lm_head.weight],         lr=0.008, betas=(0.9, 0.95), fused=True)
params = list(model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2] + [model.skip_weights]
optimizer3 = Muon(matrix_params, lr=0.04, momentum=0.95)
optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.9, 0.95), fused=True) # note that this learning rate is neither sensitive nor tuned
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
# learning rate decay scheduler (linear warmup and cooldown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    # 3) linear cooldown
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio
# resume optimizers
if init_from == 'resume':
    optimizer1.load_state_dict(checkpoint['optimizers'][0])
    optimizer2.load_state_dict(checkpoint['optimizers'][1])
    optimizer3.load_state_dict(checkpoint['optimizers'][2])
    optimizer4.load_state_dict(checkpoint['optimizers'][3])
checkpoint = None # free up memory

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset and write checkpoints
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step or (args.save_every > 0 and step % args.save_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        checkpoint = dict(step=step, code=code, model=model.state_dict(), model_args=model_args, optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(checkpoint, 'out/ckpt.pt')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, args.batch_size+1):
        # forward pass
        loss = model(x, y)
        train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < args.batch_size:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= args.batch_size
    # momentum warmup for Muon
    frac = min(step/500, 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.
    approx_time = training_time_ms + 1000 * (time.time() - t0)
    print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
    
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")