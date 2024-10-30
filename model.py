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
#!conda install einops
from einops import rearrange, einsum
#!conda install faiss-gpu
import faiss

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class KNN():
    def __init__(self, dim, max_memories):
        self.dim = dim
        self.max_memories = max_memories
        self.shape = (max_memories, 2, dim)
        self.db_offset = 0
        self.db_filepath = "./memory.memmap"
        self.db = np.memmap(self.db_filepath, mode = 'w+', dtype = np.float32, shape = self.shape)
        self.index = faiss.IndexFlatL2(dim)

    def add_to_db(self, new_data):
        new_data_len = new_data.shape[0]
        ids = (np.arange(new_data_len) + self.db_offset)
        self.db[ids] = new_data.detach().cpu().numpy() #######
        self.db_offset += new_data_len
        # Write to file
        self.db.flush()

    def search_and_retrieve(self, query_vecs, topk):
        query_vecs = query_vecs
        _, indices = self.index.search(query_vecs, topk)
        kvs = self.db[indices]
        return kvs

    def add(self, new_data):
        # Input is b n 2 d, flatten to (b n) 2 d
        new_data = new_data.flatten(0,1)
        # Add to db
        self.add_to_db(new_data)
        # Only keys are used in knn index
        keys, vals = new_data.unbind(dim=-2)
        keys = keys.detach().cpu().numpy() ######
        # Add (b n) d tensors to index
        keys = np.ascontiguousarray(keys)
        # Add to index
        self.index.add(keys)

    def search(self, query_vecs, topk):
        query_batch_size, query_seq_len = query_vecs.shape[0], query_vecs.shape[1]
        device = query_vecs.device ######
        # Input is b n d, flatten to (b n) d
        query_vecs = query_vecs.flatten(0,1)
        kvs = self.search_and_retrieve(np.ascontiguousarray(query_vecs.detach().cpu().numpy()), topk) ######
        # kvs are (b n) k 2 d, unflatten to b n k 2 d
        kvs = torch.tensor(kvs)
        kvs = torch.unflatten(kvs, 0, (query_batch_size, query_seq_len))
        return kvs.to(device)

    def clear(self):
        self.index.reset()
        self.db[:] = 0
        self.db_offset = 0

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dimension, heads=8, head_dimension=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dimension ** -0.5

        self.query_matrix = nn.Linear(embedding_dimension, self.heads * head_dimension)
        self.key_matrix = nn.Linear(embedding_dimension, self.heads * head_dimension)
        self.value_matrix = nn.Linear(embedding_dimension, self.heads * head_dimension)
        self.output_matrix = nn.Linear(self.heads * head_dimension, embedding_dimension)

    def forward(
        self, x, # batch_size, sequence_length, embedding_dimension
        relative_positions = None, xl_memory = None
    ):

        device = x.device
        queries = self.query_matrix(x)
        keys = self.key_matrix(x)
        values = self.value_matrix(x)

        queries = queries * self.scale

        if xl_memory is not None:
            k_xl, v_xl = xl_memory.unbind(dim = -2) # assume stacked
            keys = torch.cat((k_xl, keys), dim = -2) # prepend XL memory
            values = torch.cat((v_xl, values), dim = -2) # prepend XL memory
            xl_sequence_length = k_xl.shape[1]

        queries = rearrange(queries, 'b t (h d) -> b h t d', h = self.heads)
        keys    = rearrange(keys, 'b t (h d) -> b h t d', h = self.heads)
        qk      = einsum(queries, keys, 'b h i d, b h j d -> b h i j')

        i, j = qk.shape[-2:]
        if relative_positions is not None:
            qk = relative_positions[..., -i:, -j:] + qk

        qk = qk * self.scale

        mask = torch.ones((i,j), dtype = torch.bool, device=device).triu(j-i+1)
        qk = qk.masked_fill(mask, float('-inf'))

        qk = F.softmax(qk, dim=-1)
        qk = self.dropout(qk)

        values = rearrange(values, 'b t (h d) -> b h t d', h=self.heads)
        qkv = qk@values
        qkv = rearrange(qkv, 'b h t d -> b t (h d)')

        out = self.output_matrix(qkv)

        # new XL memories
        keys = rearrange(keys, 'b h t d -> b t (h d)', h = self.heads)
        values = rearrange(values, 'b h t d -> b t (h d)', h=self.heads)
        kv_memories = torch.stack((keys, values), dim=-2) # (batch, sequence_len, 2, dimension)

        if xl_memory is not None:
            _, current_input = kv_memories[:, :-xl_sequence_length], kv_memories[:, -xl_sequence_length:]
            kv_to_add_xl = current_input
        else:
            kv_to_add_xl = kv_memories

        return out, kv_to_add_xl

class KNNAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_head
        self.scale = config.head_dimension ** -0.5
        self.dropout = nn.Dropout(config.dropout)

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.query_matrix = nn.Linear(embedding_dimension, heads * head_dimension)
        self.key_matrix = nn.Linear(embedding_dimension, heads * head_dimension)
        self.value_matrix = nn.Linear(embedding_dimension, heads * head_dimension)
        self.output_matrix = nn.Linear(heads * head_dimension, embedding_dimension)
        self.final_attn = Block(config)

        self.gate_bias = nn.Parameter(torch.randn(self.heads, 1, 1))
        self.topk_retrieved_memories = topk_retrieved_memories
        self.knn = knn

    def forward(
        self, x, # batch_size, sequence_length, embedding_dimension
        relative_positions = None, xl_memory = None
    ):
        device = x.device
        x = self.ln_1(x)
        queries = self.query_matrix(x)
        keys = self.key_matrix(x)
        values = self.value_matrix(x)

        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)

        if xl_memory is not None:
            k_xl, v_xl = xl_memory.unbind(dim = -2) # unstack
            keys = torch.cat((k_xl, keys), dim = -2) # prepend XL memory
            values = torch.cat((v_xl, values), dim = -2) # prepend XL memory
            xl_sequence_length = k_xl.shape[1]

        ### LOCAL ATTENTION
        queries = rearrange(queries, 'b t (h d) -> b h t d', h = self.heads)
        keys    = rearrange(keys, 'b t (h d) -> b h t d', h = self.heads)
        qk      = einsum(queries, keys, 'b h i d, b h j d -> b h i j')

        i, j = qk.shape[-2:]
        if relative_positions is not None:
            qk = relative_positions[..., -i:, -j:] + qk

        qk = qk * self.scale

        mask = torch.ones((i,j), dtype = torch.bool, device=device).triu(j-i+1) ########
        qk = qk.masked_fill(mask, float('-inf'))

        qk = F.softmax(qk, dim=-1)

        qk = self.dropout(qk)

        values = rearrange(values, 'b t (h d) -> b h t d', h=self.heads)
        qkv = qk@values

        ### KNN ATTENTION
        # If there are knn memories (we're not on the first segment) then perform knn attention
        if self.knn.index.ntotal > 0:
            t1 = time.time()
            print ("Begin KNN operations")
            # Convert queries to search form
            queries = rearrange(queries, 'b h t d -> b t (h d)')
            mem_kv = self.knn.search(queries, topk = self.topk_retrieved_memories) # returns b t k 2 d
            mem_k, mem_v = mem_kv.unbind(dim = -2)
            mem_k = rearrange(mem_k, 'b t k (h d) -> b h t k d', h=self.heads)
            mem_v = rearrange(mem_v, 'b t k (h d) -> b h t k d', h=self.heads)

            # Convert queries to attention form
            queries = rearrange(queries, 'b t (h d) -> b h t d', h = self.heads)
            mem_qk = einsum(queries, mem_k, 'b h t d, b h t k d -> b h t k')
            mem_qk = mem_qk * self.scale

            mem_qk = F.softmax(mem_qk, dim=-1)
            mem_qk = self.dropout(mem_qk)
            mem_qkv = einsum(mem_qk, mem_v, 'b h t k, b h t k d -> b h t d')

            # Combined attentions
            combined_qkv = mem_qkv * self.gate_bias + qkv * (1 - self.gate_bias)
            combined_qkv = rearrange(combined_qkv, 'b h t d -> b t (h d)')
            out = self.output_matrix(combined_qkv)
            t2 = time.time()
            print ("End KNN operations, time taken:", t2-t1)
        else:
            qkv = rearrange(qkv, 'b h t d -> b t (h d)')
            out = self.output_matrix(qkv)
            out = self.final_attn(out)

        # New XL memories
        keys = rearrange(keys, 'b h t d -> b t (h d)', h = self.heads)
        values = rearrange(values, 'b h t d -> b t (h d)', h=self.heads)
        kv_memories = torch.stack((keys, values), dim=-2) # (batch, sequence_len, 2, dimension)

        if xl_memory is not None:
            # if we're on a middle/end segment of a document (there are previous XL memories)
            _, current_kv = kv_memories[:, :-xl_sequence_length], kv_memories[:, -xl_sequence_length:]
        else:
            # if we're at the first segment
            current_kv = kv_memories

        self.knn.add(current_kv)
        return out, current_kv


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

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x, xl_memory):
        attn_out, new_xl_memories = self.attn(self.ln_1(x), xl_memory=xl_memory)
        x = x + attn_out

        x = x + self.mlp(self.ln_2(x))
        return x, new_xl_memories

class MemorizingGPT(nn.Module):  
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            knn_attention = KNNAttention(config),            
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

    def forward(self, idx, xl_memory=None, targets=None):
        # If no XL memories (start of a sequence) then None type for each layer.
        # There is one set of XL memories for each layer
        # xl_memories = default(xl_memories, (None,) * self.num_xl_memory_layers)
        if xl_memory is None:
            xl_memory = (None,) * self.config.n_embd
        # Iterator
        xl_memories_iter = iter(xl_memory)

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
        for block in self.transformer:
            x, xl_mem = block(x, next(xl_memories_iter))
            if xl_mem is not None:
                new_memories.append(xl_mem.detach())

        x = self.transformer.ln_f(x)

        # Training
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if len(new_memories) > 0:
            return new_memories, loss
        return logits, loss
    
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
