import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp
import time
from einops import rearrange, einsum
#!conda install faiss-gpu
import faiss
from torch.compiler import allow_in_graph
# Use of FlexAttention
from torch.nn.attention.flex_attention import flex_attention, BlockMask

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(torch.nn.Module):
    def __init__(self, n_embd, n_head, base=10000):
        super().__init__()
        assert n_embd % n_head == 0, (
            f"n_embd {n_embd} must be divisible by n_head {n_head}"
        )
        head_dim = n_embd // n_head
        assert head_dim % 2 == 0, (
            f"head_dim {head_dim} must be divisible by 2"
        )
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, head_dim, 2) / head_dim))
        self.cos_cached = None
        self.sin_cached = None
        self.seq_len_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if self.cos_cached is None or seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device).float()
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        x1, x2 = x.chunk(2, dim=3)
        assert x1.shape[-1] == cos.shape[-1], (
            f"Mismatch: x1 last dim {x1.shape[-1]} vs cos last dim {cos.shape[-1]}"
        )
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        assert dim % n_head == 0, (
            f"Embedding dim {dim} must be divisible by n_head {n_head}"
        )
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.embedding_dim = dim
        # Linear layers
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        # value residual lambda
        self.lamb = nn.Parameter(torch.tensor(0.5))
        # rotary embeddings
        self.rotary = Rotary(dim, n_head)
        # output projection
        self.c_proj = CastedLinear(dim, dim)
        nn.init.xavier_uniform_(self.c_proj.weight)

    def forward(self, x, vi, block_mask):
        B, _, T, _ = x.size()  # batch size, sequence length (T), embedding_dim
        assert B == 1, "Must use batch size = 1 for FlexAttention"

        # Apply linear transformations
        q = self.c_q(x)  # [B, T, embedding_dim]
        k = self.c_k(x)  # [B, T, embedding_dim]
        v = self.c_v(x)  # [B, T, embedding_dim]

        # Ensure that embedding_dim is divisible by n_head
        head_dim = self.embedding_dim // self.n_head

        # Reshape to [B, T, n_head, head_dim]
        q = q.view(B, T, self.n_head, head_dim)  # [B, T, n_head, head_dim]
        k = k.view(B, T, self.n_head, head_dim)  # [B, T, n_head, head_dim]
        v = v.view(B, T, self.n_head, head_dim)  # [B, T, n_head, head_dim]

        if vi is None:
            vi = v.clone()  # This happens if we are in the first block
        elif vi.shape != v.shape: # Reshape or expand vi to match v if necessary
            vi = vi.view_as(v)  # Ensure compatible dimensions
        # Calculate value residual and apply rotary embeddings
        v = (1 - self.lamb) * v + self.lamb * vi
        q, k = norm(q), norm(k)  # QK norm
        q, k = self.rotary(q), self.rotary(k)

        # Apply attention mechanism
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().view_as(x)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


@allow_in_graph
def add_to_faiss_index(n_embd):
    return faiss.IndexFlatL2(n_embd)

# k-nearest-neighbor layer for the external memory
class KNN:
    def __init__(self, n_embd, max_memories):
        self.n_embd = n_embd
        self.max_memories = max_memories
        self.db_offset = 0
        # FAISS Index
        self.index = faiss.IndexFlatL2(n_embd)
        # Memmap initialization
        self.db = np.memmap(
            "./memory.memmap", mode='w+', dtype=np.float32, shape=(max_memories, 2, n_embd)
        )
        # Asynchronous update
        self.update_queue = mp.Queue()
        self.update_process = mp.Process(
            target=self._update_memory_worker, args=("./memory.memmap", self.update_queue, max_memories, n_embd)
        )
        self.update_process.start()

    def _update_memory_worker(self, memmap_file, queue, max_memories, embd_dim):
        """
        Asynchronous process for updating the memmap file.
        """
        db = np.memmap(memmap_file, mode='r+', dtype=np.float32, shape=(max_memories, 2, embd_dim))
        while True:
            new_data = queue.get()
            if new_data is None:
                break  # End job
            idx = new_data["idx"]
            db[idx % max_memories] = new_data["value"]
        db.flush()

    def add_async(self, new_data):
        """
        Asynchronously adding new keys/values ​​to memmap and FAISS.
        """
        batch_size = new_data.shape[0]
        if self.db_offset + batch_size > self.max_memories:
            print("Memory limit reached. Clearing database.")
            self.clear()
        # Adding keys in FAISS
        keys = new_data[:, 0].reshape(-1, self.n_embd).detach().cpu().to(torch.float32).numpy()
        self.index.add(keys)
        # Adding tasks in queue
        self.update_queue.put({
            "idx": np.arange(batch_size) + self.db_offset,
            "value": new_data.detach().cpu().to(torch.float32).numpy()
        })
        self.db_offset += batch_size

    def search_and_retrieve(self, query_vecs, topk):
        _, indices = self.index.search(query_vecs.squeeze(0), topk)
        kvs = self.db[indices]
        return kvs

    def search(self, query_vecs, topk):
        query_batch_size, query_seq_len = query_vecs.shape[0], query_vecs.shape[1]
        device = query_vecs.device
        # Input is b n d, flatten to (b n) d
        query_vecs = query_vecs.to(torch.float32).detach().cpu().numpy()
        kvs = self.search_and_retrieve(np.ascontiguousarray(query_vecs), topk)
        # kvs are (b n) k 2 d, unflatten to b n k 2 d
        kvs = torch.tensor(kvs)
        kvs = torch.unflatten(kvs, 0, (query_batch_size, query_seq_len))
        return kvs.to(device)

    def clear(self):
        """
        Clearing FAISS and memmap.
        """
        self.index.reset()
        self.db[:] = 0
        self.db.flush()
        self.db_offset = 0

# k-nearest-neibhor attention block for the external memory
class KNNAttention(nn.Module):
    def __init__(self, config, knn):
        super().__init__()
        self.n_head = config.n_head
        assert config.n_embd % config.n_head == 0
        self.scale = config.n_embd / config.n_head ** -0.5
        self.max_memories = config.max_knn_memories
        # key, query, value projections for all heads, but in a batch
        self.query_matrix = nn.Linear(config.n_embd, config.n_embd)
        self.key_matrix = nn.Linear(config.n_embd, config.n_embd)
        self.value_matrix = nn.Linear(config.n_embd, config.n_embd)
        self.output_matrix = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.dropout = nn.Dropout(config.dropout)
        self.gate_bias = nn.Parameter(torch.randn(config.n_head, 1, 1))
        self.knn = knn

    def forward(self, x): # batch_size, sequence_length, embedding_dimension
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        device = x.device
        _, sequence_length = x.shape[:2]
        x = x.squeeze(1) # deleting batch
        # Getting queries, keys, and values
        queries = self.query_matrix(x)
        keys = self.key_matrix(x)
        values = self.value_matrix(x)
        # Normalizing
        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
        ### LOCAL ATTENTION
        queries = rearrange(queries, 'b t (h d) -> b h t d', h = self.n_head)
        keys    = rearrange(keys, 'b t (h d) -> b h t d', h = self.n_head)
        qk      = einsum(queries, keys, 'b h i d, b h j d -> b h i j')
        # Masking
        i, j = qk.shape[-2:]
        qk = qk * self.scale
        mask = torch.ones((i,j), dtype = torch.bool, device=device).triu(j-i+1)
        qk = qk.masked_fill(mask, float('-inf'))

        qk = F.softmax(qk, dim=-1)
        qk = self.dropout(qk)

        values = rearrange(values, 'b t (h d) -> b h t d', h = self.n_head)
        qkv = qk@values
        ### KNN ATTENTION
        # If there are knn memories (we're not on the first segment) then perform knn attention
        if self.knn.index.ntotal >= self.max_memories:
            logger.info("Clearing KNN memories due to limit reached.")
            self.knn.clear()
        if self.knn.index.ntotal > 0:
            # Convert queries to search form
            queries_flat = rearrange(queries, 'b h t d -> b t (h d)')
            kv_memories = self.knn.search(queries_flat, topk=3) # returns b t k 2 d
            mem_k, mem_v = kv_memories.unbind(dim=-2)
            # Convert queries to attention form
            mem_k = rearrange(mem_k, 'b t k (h d) -> b h t k d', h = self.n_head)
            mem_v = rearrange(mem_v, 'b t k (h d) -> b h t k d', h = self.n_head)
            # Calculating weights
            mem_qk = einsum(queries.to(torch.float32), mem_k, 'b h t d, b h t k d -> b h t k')
            mem_qk = mem_qk * self.scale

            mem_qk = F.softmax(mem_qk, dim = -1)
            mem_qk = self.dropout(mem_qk)
            mem_qkv = einsum(mem_qk, mem_v, 'b h t k, b h t k d -> b h t d')
            # Combined attentions
            combined_qkv = mem_qkv * self.gate_bias + qkv * (1 - self.gate_bias)
            combined_qkv = rearrange(combined_qkv.to(torch.bfloat16), 'b h t d -> b t (h d)')
            out = self.output_matrix(combined_qkv)
        else:
            logger.info("No KNN memories available.")
            qkv = rearrange(qkv, 'b h t d -> b t (h d)')
            out = self.output_matrix(qkv)
        ### MEMORY UPDATE (ASYNCHRONY)
        keys = rearrange(keys, 'b h t d -> b t (h d)', h=self.n_head)
        values = rearrange(values, 'b h t d -> b t (h d)', h=self.n_head)
        kv_memories = torch.stack((keys, values), dim=-2) # (batch, sequence_len, 2, dimension)
        _, kv_memories = kv_memories[:, :-sequence_length], kv_memories[:, -sequence_length:]
        self.knn.add_async(kv_memories) # asynchrony adding

        return out

# Multi level perceptron
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_() # zero init

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.n_embd, config.n_head)
        self.mlp = MLP(config.n_embd)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, vi, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = self.attn(norm(x), vi, block_mask)
        x = x + self.mlp(norm(x))
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.__setattr__
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.n_embd)
            for _ in range(config.n_head)
        ])
    def forward(self, inputs) -> "list[torch.Tensor]":
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve


class MemorizingGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.head_dim = config.n_embd // config.n_head
        self.sliding_window_size = torch.tensor(self.head_dim, dtype=torch.int32, device="cuda")
        self.n_layer = config.n_layer
        # U-net design
        self.num_encoder_layers = config.n_layer // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        knn = KNN(config.n_embd, config.max_knn_memories)
        self.drop = nn.Dropout(config.dropout)
        self.knn_attention = KNNAttention(config, knn)

        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.value_embeds = ValueEmbedding(config)

        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_()

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model. The position embeddings (wpe) get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(
            self,
            sliding_window_num_blocks: torch.Tensor,
            idx: torch.Tensor,
            target: torch.Tensor = None
        ):
        idx = torch.squeeze(idx)
        seq_len = len(idx)
        assert seq_len % self.head_dim == 0, (
            f"seq_len = {seq_len}, self.head_dim = {self.head_dim}"
        )
        total_num_blocks = seq_len // self.head_dim
        docs = (idx == 50256).cumsum(0)
        assert idx.ndim == 1
        docs_low = docs.view(-1, self.head_dim)[:, 0].contiguous()
        docs_high = docs.view(-1, self.head_dim)[:, -1].contiguous()
        def document_causal(_, __, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_mask: torch.Tensor):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = dense_mask.to(torch.int32).argsort(dim=-1, descending=True, stable=True)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        def create_doc_swc_block_mask(sliding_window_num_blocks: torch.Tensor):
            kv_idx = block_idx = torch.arange(total_num_blocks, dtype=torch.int32, device="cuda")
            q_idx = block_idx[:, None]
            causal_bm = q_idx >= kv_idx
            causal_full_bm = q_idx > kv_idx
            window_bm = q_idx - kv_idx < sliding_window_num_blocks
            window_full_bm = window_bm
            document_bm = (docs_low[:, None] <= docs_high) & (docs_low <= docs_high[:, None])
            document_full_bm = (docs_low[:, None] == docs_high) & (docs_low == docs_high[:, None])
            nonzero_bm = causal_bm & window_bm & document_bm
            full_bm  = causal_full_bm & window_full_bm & document_full_bm
            kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm ^ full_bm)
            full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)
            return BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                BLOCK_SIZE=self.head_dim,
                mask_mod=document_causal
            )
        block_mask = create_doc_swc_block_mask(sliding_window_num_blocks)

        # forward the GPT model itself
        x = self.embed(idx[None]) # token embeddings of shape (b, t, n_embd)
        x = norm(x)
        x = self.drop(x)
        x0 = x
        ve = self.value_embeds(idx)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_mask)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            if i == self.num_decoder_layers - 2:
                x = x + self.skip_weights[i] * skip_connections.pop()
                x = self.knn_attention(x)
                x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)
            else:
                x = x + self.skip_weights[i] * skip_connections.pop()
                x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)

        x = norm(x)
        if target is not None:
            # if we are given some desired target also calculate the loss
            logits = self.lm_head(x)
            logits = 30 * torch.tanh(logits / 30)
            target = target.view(-1)
            if logits.size(0) > target.size(0):
              logits = logits[-target.size(0):]
            logits = logits.view(-1, logits.size(-1))
            target = target.view(-1)
            loss = F.cross_entropy(logits, target, ignore_index=-1)
        else:
            loss = None

        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # sample from the distribution
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Faster than multinomial
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx