"""
https://github.com/i90007/picoGPT
The training script for running on a single gpu
Little logs:
1) openwebtext 0.8B, 1 T4 GPU, Google Colab, 53m.
number of parameters: 1208.23M
sequence_length = 1152
n_layer         = 22
n_head          = 16
n_embd          = 1024
dropout         = 0.4
max_knn_memories= 500000
num_iterations  = 490
step: 0-400 val_loss: 6.500 train_time:2278108ms step_avg: 5841.30ms
step: 400-900 val_loss: 6.312 train_time:2953.8072106838226s step_avg: 6.03ms
"""
import gc
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import time
import contextlib
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch._inductor.config as config
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from model import MemorizingGPT, CastedLinear, KNN
def is_colab():
  try:
      from google.colab import drive
      return True
  except ImportError:
      return False
is_colab = is_colab()
# if not hasattr(torch.compiler, "set_stance"):
#   !pip uninstall torch
#   !pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
print(torch.__version__)

init_from = 'scratch' # 'scratch', 'resume', 'resume_google_drive'
def mount_drive_and_get_path():
    drive.mount('/content/drive', force_remount=True)
    return '/content/drive/My Drive/'
dataset = mount_drive_and_get_path() # 'data/openwebtext/', 'data/tinyshakespeare/', mount_drive_and_get_path()
# attempt to autodetect device
device = "cuda"
if not torch.cuda.is_available():
    print("Trainig possible with CUDA only!")
    os.sys.exit()
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    sequence_length : int = 2048 # sequence length, in tokens (2048 is max reasonable for openwebtext dataset)
    vocab_size : int      = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer : int         = 18 # size of the model (n_head * 2)
    n_head : int          = 9 # size of the model (hed size = 128)
    n_embd : int          = 1152 # size of the model
    dropout : float       = 0.1 #
    # the maximum number of memories (~2.7GB) that will be stored locally
    max_knn_memories: bool= 500000
    num_iterations : int  = 200 # number of iterations to run (100 for tinyshakespeare, 2000 for openwebtext)
configGpt = GPTConfig()
@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str        = f'{dataset}train*.bin' # input .bin to train on
    input_val_bin : str    = f'{dataset}val*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int       = 1 # batch size, in sequences, across all devices (shold be as low as possible)
    device_batch_size : int= 1 # batch size, in sequences, per device
    warmup_iters : int     = 0
    cooldown_iters : int   = 60 # number of iterations of linear warmup for triangular or trapezoidal schedule (60 for tinyshakespeare, 600 for openwebtext 1B)
    # evaluation and logging hyperparams
    val_loss_every : int   = 100 # every how many steps to evaluate val loss? 0 for only at the end (10 for tinyshakespeare, 100 for openwebtext 1B)
    val_steps : int        = 10
    save_every : int       = 100 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()
# we are running on a single gpu, and one process
tokens_per_iter = args.batch_size * args.device_batch_size * configGpt.sequence_length
print(f"tokens per iteration will be: {tokens_per_iter:,}")
# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader
def next_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(dataset, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(dataset, 'val.bin'), dtype=np.uint16, mode='r')
    ix       = torch.randint(len(data) - configGpt.sequence_length, (args.device_batch_size,))
    sequences= [torch.from_numpy(data[i:i+configGpt.sequence_length].astype(np.int64)) for i in ix]
    targets  = [torch.from_numpy(data[i+1:i+1+configGpt.sequence_length].astype(np.int64)) for i in ix]
    x        = torch.cat(sequences, dim=0)
    y        = torch.cat(targets, dim=0)
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y     = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y
# load tokens
print('='*100)
torch.cuda.empty_cache()
inputs_train, targets_train = next_batch('train') # fetch the very first batch
# -----------------------------------------------------------------------------
# Muon optimizer
def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile # (backend="onnxrt")
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
    X = G.half() # or bfloat16()
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
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.float16) # or bfloat16
            curr_idx = 0
            for i, p in enumerate(group['params']):
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

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()
# model init
model_args = dict(n_layer=configGpt.n_layer, n_head=configGpt.n_head, n_embd=configGpt.n_embd,
    vocab_size=configGpt.vocab_size, dropout=configGpt.dropout, max_knn_memories=configGpt.max_knn_memories)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = MemorizingGPT(GPTConfig())
else:
  if init_from == 'resume':
      print(f"Resuming training from local directory")
      path = 'ckpt.pt'
  elif init_from == 'resume_google_drive':
      print(f"Resuming training from google_drive")
      if not os.path.exists('/content/drive'):
        drive.mount('/content/drive', force_remount=True)
      path = '/content/drive/My Drive/ckpt.pt'
  torch.cuda.empty_cache()
  gc.collect()
  torch.cuda.ipc_collect()
  checkpoint = torch.load(path)
  checkpoint_model_args = checkpoint['model_args']
  # force these config attributes to be equal otherwise we can't even resume training
  for k in ['n_layer', 'n_head', 'n_embd', 'dropout', 'vocab_size', 'max_knn_memories']:
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
model = model.cuda().half() # or bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True
# compile the model
def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)
model.apply(initialize_weights)
print("compiling the model... (takes a ~minute)")
model = torch.compile(model) # backend="onnxrt"

# KNN
knn = KNN(configGpt.n_embd, configGpt.max_knn_memories, configGpt.vocab_size, device)
def compute_knn_loss(knn_results, logits, lambda_knn=0.1):
    knn_logits = np.mean(knn_results, axis=(1, 2))
    knn_logits = torch.tensor(knn_logits, device=logits.device, dtype=logits.dtype)
    knn_logits = knn_logits @ knn.projection_layer.weight
    knn_logits = knn_logits.squeeze()

    assert logits.shape == knn_logits.shape, f"Mismatch: logits {logits.shape}, knn_logits {knn_logits.shape}"
    mse_loss = F.mse_loss(logits, knn_logits)
    return lambda_knn * mse_loss

# init the optimizer(s)
embed_params = [*model.embed.parameters(), *model.value_embeds.parameters(), *knn.get_parameters()]
optimizer1 = torch.optim.Adam(embed_params, lr=0.6, betas=(0.8, 0.95), fused=True)
optimizer2 = torch.optim.Adam([model.lm_head.weight, model.skip_weights], lr=0.008, betas=(0.8, 0.95), fused=True)
params = list(model.blocks.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2] + [model.skip_weights]
optimizer3 = Muon(matrix_params, lr=0.05, momentum=0.95)
optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True) # note that this learning rate is neither sensitive nor tuned
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
# learning rate decay scheduler (linear warmup and cooldown)
def get_lr(it):
    assert it <= configGpt.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < configGpt.num_iterations - args.cooldown_iters:
        return 1.0
    # 3) linear cooldown
    else:
        return (configGpt.num_iterations - it) / args.cooldown_iters
# resume optimizers
if init_from == 'resume':
    optimizer1.load_state_dict(checkpoint['optimizers'][0])
    optimizer2.load_state_dict(checkpoint['optimizers'][1])
    optimizer3.load_state_dict(checkpoint['optimizers'][2])
    optimizer4.load_state_dict(checkpoint['optimizers'][3])
checkpoint = None # free up memory

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

sliding_window_num_blocks = torch.tensor(1, dtype=torch.int32, device="cuda")
sw_num_blocks_prev = 1
# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()

embed = model.embed.weight
# begin training
torch.cuda.empty_cache()
gc.collect()
torch.cuda.ipc_collect()
for step in range(configGpt.num_iterations + 1):
    last_step = (step == configGpt.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # Linearly increase the sliding window size over training in chunks of 64 from 64 -> 1792. By @fernbear.bsky.social
    frac_done = step / configGpt.num_iterations # training progress
    sw_num_blocks = int(((1 - frac_done) * 64 + frac_done * 1792 + 64) // 128)
    if sw_num_blocks != sw_num_blocks_prev:
        sliding_window_num_blocks.copy_(sw_num_blocks, non_blocking=True)
        sw_num_blocks_prev = sw_num_blocks

    # once in a while evaluate the validation dataset and write checkpoints
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += time.time() - t0
        # run validation batches
        model.eval()
        val_loss = 0.0
        for _ in range(args.val_steps):
            with torch.no_grad():
                model.skip_weights.data.uniform_(0.1, 0.5)
                x_val, y_val = next_batch('val')
                logits = model(sliding_window_num_blocks, x_val)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_val.view(-1))
                val_loss += loss
        val_loss /= args.val_steps
        # log val loss to console and to logfile
        print(f'step: {step}/{configGpt.num_iterations} val_loss: {val_loss:.3f} train_time: {training_time_ms:.0f}s step_avg: {training_time_ms/(timed_steps-1):.1f}s')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step or (step > 0 and step % args.save_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += time.time() - t0
        # save the state of the training process
        checkpoint = dict(step=step, code=code, model=model.state_dict(), model_args=model_args, optimizers=[opt.state_dict() for opt in optimizers])
        if drive != False:
            try:
                if not os.path.exists('/content/drive'):
                  drive.mount('/content/drive', force_remount=True)
                torch.save(checkpoint, '/content/drive/My Drive/ckpt.pt')
                print(f'Checkpoint saved to /content/drive/My Drive/ckpt.pt')
            except Exception as e:
                print(f'Error saving to Google Drive: {e}')
        else:
            torch.save(checkpoint, 'ckpt.pt')
            print(f'Checkpoint saved locally to ckpt.pt')
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
    for i in range(1, args.batch_size + 1):
        with contextlib.ExitStack() as stack:
            if step >= args.warmup_iters:
                stack.enter_context(torch.compiler.set_stance("default"))
            logits = model(sliding_window_num_blocks, inputs_train)
            # Search KNN
            if step > 0:
              query_vecs = logits @ embed  # Dimensionality reduction
              knn_results = knn.search_and_retrieve(query_vecs, current_step=step, total_steps=configGpt.num_iterations)
              # Additional KNN-based loss function (regularization)
              knn_loss = compute_knn_loss(knn_results, logits)
            else:
              knn_loss = 0
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_train.view(-1)) + knn_loss
            loss.backward()
            # Adding new data in memory
            inputs_train = inputs_train.unsqueeze(-1).expand(-1, logits.size(-1)) 
            kv_memories = torch.stack([inputs_train, logits], dim=-2)
            knn.add_async(kv_memories)

            inputs_train, targets_train = next_batch('train')
    if args.batch_size != 1:
        for p in model.parameters():
            p.grad /= args.batch_size
    # momentum warmup for Muon
    frac = min(step/300, 1)
    for group in optimizer3.param_groups:
        group['momentum'] = (1 - frac) * 0.85 + frac * 0.95

    # step the optimizers and schedulers
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.
    approx_time = training_time_ms + time.time() - t0
    print(f"step: {step+1}/{configGpt.num_iterations} train_time: {approx_time:.0f}s step_avg: {approx_time/timed_steps:.0f}s")
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    torch.cuda.empty_cache()