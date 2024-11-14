"""
Full definition of the Language Model, all of it in this single file.
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import inspect
import time
from einops import rearrange, einsum
#!conda install faiss-gpu
import faiss
from torch.compiler import allow_in_graph

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        B, T, C = input.size()
        if T != self.weight.shape[0]:
            B, C, T = input.size()
        return F.layer_norm(input.view(B, C, T), self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(
        self, x, # batch_size, block_size, n_embd
        xl_memory = None
    ):
        if xl_memory is not None:
            B, T, M, C = xl_memory.size()
            xl_memory = xl_memory.view(B*M, T, C)
            q, k_xl, v  = self.c_attn(xl_memory).split(self.n_embd, dim=2)
            k = k_xl.view(B*M, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B*M, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B*M, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

            xl_sequence_length = k_xl.shape[1]
        else:
            M = 1
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B*M, T, C) # re-assemble all head outputs side by side

        # output projection
        out = self.resid_dropout(self.c_proj(y))

        # new XL memories
        k = rearrange(k, 'b h t d -> b t (h d)', h = self.n_head)
        v = rearrange(v, 'b h t d -> b t (h d)', h = self.n_head)
        kv_memories = torch.stack((k, v), dim=-2) # (batch, sequence_len, 2, dimension)

        if xl_memory is not None:
            _, kv_memories = kv_memories[:, :-xl_sequence_length], kv_memories[:, -xl_sequence_length:]

        return out, kv_memories

@allow_in_graph
def add_to_faiss_index(n_embd):
    return faiss.IndexFlatL2(n_embd)

# k-nearest-neibhor layer for the external memory
class KNN():
    def __init__(self, n_embd, max_memories):
        self.shape = (max_memories, 2, n_embd)
        self.db_offset = 0
        self.db_filepath = "./memory.memmap"
        self.db = np.memmap(self.db_filepath, mode = 'w+', dtype = np.float32, shape = self.shape)
        self.index = add_to_faiss_index(n_embd)

    def add_to_db(self, new_data):
        new_data_len = new_data.shape[0]
        ids = (np.arange(new_data_len) + self.db_offset)
        self.db[ids] = new_data.detach().cpu().numpy()
        self.db_offset += new_data_len
        # Write to file
        self.db.flush()

    def search_and_retrieve(self, query_vecs, topk):
        _, indices = self.index.search(query_vecs, topk)
        kvs = self.db[indices]
        return kvs

    def add(self, new_data):
        # Input is b n 2 d, flatten to (b n) 2 d
        new_data = new_data.flatten(0,1)
        # Add to db
        self.add_to_db(new_data)
        # Only keys are used in knn index
        keys, _ = new_data.unbind(dim=-2)
        keys = keys.detach().cpu().numpy()
        # Add (b n) d tensors to index
        keys = np.ascontiguousarray(keys)
        # Add to index
        self.index.add(keys)

    def search(self, query_vecs, topk):
        query_batch_size, query_seq_len = query_vecs.shape[0], query_vecs.shape[1]
        device = query_vecs.device
        # Input is b n d, flatten to (b n) d
        query_vecs = query_vecs.flatten(0,1)
        kvs = self.search_and_retrieve(np.ascontiguousarray(query_vecs.detach().cpu().numpy()), topk)
        # kvs are (b n) k 2 d, unflatten to b n k 2 d
        kvs = torch.tensor(kvs)
        kvs = torch.unflatten(kvs, 0, (query_batch_size, query_seq_len))
        return kvs.to(device)

    def clear(self):
        self.index.reset()
        self.db[:] = 0
        self.db_offset = 0

# k-nearest-neibhor attention block for the external memory
class KNNAttention(nn.Module):
    def __init__(self, config, knn):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.max_memories = config.max_knn_memories
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self.gate_bias = nn.Parameter(torch.randn(config.n_head, 1, 1))
        self.knn = knn

    def forward(
        self, x, # batch_size, sequence_length, embedding_dimension
        xl_memory = None
    ):
        if xl_memory is not None:
            B, T, M, C = xl_memory.size()
            xl_memory = xl_memory.view(B*M, T, C)
            q, k_xl, v  = self.c_attn(xl_memory).split(self.n_embd, dim=2)
            k = k_xl.view(B*M, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B*M, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B*M, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            xl_sequence_length = k_xl.shape[1]
        else:
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        ### KNN ATTENTION
        # If there are knn memories (we're not on the first segment) then perform knn attention
        if self.knn.index.ntotal >= self.max_memories:
            self.knn.clear()
        if self.knn.index.ntotal > 0:
            t1 = time.time()
            print ("Begin KNN operations")
            # Convert queries to search form
            queries = rearrange(q, 'b h t d -> b t (h d)')
            mem_kv = self.knn.search(queries, topk = 3) # returns b t k 2 d
            mem_k, mem_v = mem_kv.unbind(dim = -2)
            mem_k = rearrange(mem_k, 'b t k (h d) -> b h t k d', h = self.n_head)
            mem_v = rearrange(mem_v, 'b t k (h d) -> b h t k d', h = self.n_head)

            # Convert queries to attention form
            queries = rearrange(queries, 'b t (h d) -> b h t d', h = self.n_head)
            mem_qk = einsum(queries, mem_k, 'b h t d, b h t k d -> b h t k')
            mem_qk = mem_qk * 0.125

            mem_qk = F.softmax(mem_qk, dim = -1)
            mem_qk = self.resid_dropout(mem_qk)
            mem_qkv = einsum(mem_qk, mem_v, 'b h t k, b h t k d -> b h t d')

            # Combined attentions
            combined_qkv = mem_qkv * self.gate_bias + y * (1 - self.gate_bias)
            combined_qkv = rearrange(combined_qkv, 'b h t d -> b t (h d)')
            out = self.resid_dropout(self.c_proj(combined_qkv))
            t2 = time.time()
            print ("End KNN operations, time taken:", t2 - t1)
        else:
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
            # output projection
            out = self.c_proj(y)
            out = self.resid_dropout(out)

        # new XL memories
        k = rearrange(k, 'b h t d -> b t (h d)', h = self.n_head)
        v = rearrange(v, 'b h t d -> b t (h d)', h = self.n_head)
        kv_memories = torch.stack((k, v), dim=-2) # (batch, sequence_len, 2, dimension)

        if xl_memory is not None:
            _, kv_memories = kv_memories[:, :-xl_sequence_length], kv_memories[:, -xl_sequence_length:]

        self.knn.add(kv_memories)
        return out, kv_memories

# Multi level perceptron
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class MemBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x, xl_memory):
        attn_out, new_xl_memories = self.attn(self.ln_1(x), xl_memory=xl_memory)
        mlp = self.mlp(self.ln_2(attn_out))
        return mlp, new_xl_memories


class MemorizingGPT(nn.Module):  
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.knn = KNN(config.n_embd, config.max_knn_memories)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([MemBlock(config) for _ in range(config.n_layer - 1)]),
            knn_attention = KNNAttention(config, self.knn),         
            ln_f = LayerNorm(config.n_embd, bias=config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model. The position embeddings (wpe) get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, xl_memories=None):
        # If no XL memories (start of a sequence) then None type for each layer.
        # There is one set of XL memories for each layer
        # xl_memories = default(xl_memories, (None,) * self.num_xl_memory_layers)
        if xl_memories is None:
            xl_memories = (None,) * self.config.n_layer
        # Iterator
        xl_memories_iter = iter(xl_memories)
        # Embeddings
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Store the XL memories for each pass
        new_memories = []
        for i, block in enumerate(self.transformer.h):
            if i == self.config.n_layer - 2:
                x, xl_mem = self.transformer.knn_attention(x, next(xl_memories_iter))
                new_memories.append(xl_mem.detach())
                x, xl_mem = block(x, next(xl_memories_iter))
            else:
                x, xl_mem = block(x, next(xl_memories_iter))

            if xl_mem is not None:
                new_memories.append(xl_mem.detach())

        x = self.transformer.ln_f(x)

        # Training
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            B, T, C = logits.size()
            logits = logits.view(T*B, C)
            targets = targets.view(-1)
            if logits.size(0) > targets.size(0):
              logits = logits[-targets.size(0):]
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if len(new_memories) > 0:
            return new_memories, loss
        return logits, loss
    
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
