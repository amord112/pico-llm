# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import List, Tuple, Optional

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # Using embedding is more efficient than one-hot encoding
        self.embedding = nn.Embedding(vocab_size, embed_size // k)
        
        # Calculate flattened input size to MLP
        self.input_size = (embed_size // k) * k
        
        # Build MLP layers
        layers = []
        # First layer from flattened embeddings to embed_size
        layers.append(nn.Linear(self.input_size, embed_size))
        layers.append(nn.SiLU())  # SiLU activation (Swish)
        
        # Inner MLP layers
        for _ in range(num_inner_layers):
            layers.append(nn.Linear(embed_size, embed_size))
            layers.append(nn.SiLU())
        
        # Output layer to vocab_size
        layers.append(nn.Linear(embed_size, vocab_size))
        
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        # Padding for beginning of sequence
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    # Instead of one-hot encoding, use embeddings
                    context_emb = self.embedding(torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device))
                    context_flat = context_emb.flatten().unsqueeze(0)  # Flatten to [1, k*embed_dim//k]
                    logits_b = self.net(context_flat)  # [1, vocab_size]
                    batch_logits.append(logits_b)
                
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # [1, batch, vocab_size]

            block_outputs = torch.cat(block_outputs, dim=0)  # [chunk_size, batch, vocab_size]
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # [seq_len, batch, vocab_size]
        return outputs

################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        q = self.q_proj(x)  
        k = self.k_proj(x)  
        v = self.v_proj(x)  
        
        q = q.view(seq_len, batch_size, self.n_heads, self.d_head)
        k = k.view(seq_len, batch_size, self.n_heads, self.d_head)
        v = v.view(seq_len, batch_size, self.n_heads, self.d_head)
        
        q = q.transpose(0, 1).transpose(1, 2)
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask[None, None, :, :], float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        context = torch.matmul(attn_probs, v)  
        context = context.transpose(1, 2).transpose(0, 1).contiguous()
        context = context.view(seq_len, batch_size, self.d_model)
        
        out = self.out_proj(context)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  
            
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(), 
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = RMSNorm(d_model)  
        self.norm2 = RMSNorm(d_model)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, n_heads=2, n_blocks=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_blocks)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, tokens_seq):
        x = self.embedding(tokens_seq)  
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)  
        return logits

################################################################################
# 5.4 Transformer with KV-cache 
################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / norm)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(1) 
        pos_enc = self.pe[start_pos : start_pos + seq_len, :].unsqueeze(0)
        x = x + pos_enc
        return self.dropout(x)

class KVCache():
    def __init__(self, n_layers, bsz, max_seq_length, n_heads, head_dim, device): 
        self.n_layers = n_layers
        self.bsz = bsz
        self.max_seq_length = max_seq_length
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device # Store device
        self.cache_k: List[torch.Tensor] = []
        self.cache_v: List[torch.Tensor] = []
        self.reset()

    def reset(self):
        self.cache_k = []
        self.cache_v = []
        for _ in range(self.n_layers):
            self.cache_k.append(torch.zeros((self.bsz, self.n_heads, 0, self.head_dim), device=self.device))
            self.cache_v.append(torch.zeros((self.bsz, self.n_heads, 0, self.head_dim), device=self.device))

    def update(self, layer, new_k, new_v):
        new_k = new_k.to(self.device)
        new_v = new_v.to(self.device)
        self.cache_k[layer] = torch.cat([self.cache_k[layer], new_k], dim=2)
        self.cache_v[layer] = torch.cat([self.cache_v[layer], new_v], dim=2)
        current_cache_len = self.cache_k[layer].shape[2]
        if current_cache_len > self.max_seq_length:
             self.cache_k[layer] = self.cache_k[layer][:, :, -self.max_seq_length:, :]
             self.cache_v[layer] = self.cache_v[layer][:, :, -self.max_seq_length:, :]

    def get(self, layer):
        return self.cache_k[layer], self.cache_v[layer]

class mha(nn.Module): 
    def __init__(self, dim, n_heads, dropout_rate = 0.1):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.n_heads = n_heads
        self.Wq = nn.Linear(dim,dim,bias=False)
        self.Wk = nn.Linear(dim,dim,bias=False)
        self.Wv = nn.Linear(dim,dim,bias=False)
        self.out = nn.Linear(dim,dim,bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.resid_dropout = nn.Dropout(dropout_rate)
        self.use_flash_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, layer = 0, start_pos = 0, cache: Optional[KVCache] = None, mask = None):
        bsz, seq_len, _ = x.shape
        q = self.Wq(x); k = self.Wk(x); v = self.Wv(x)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            cache.update(layer, k, v)
            k, v = cache.get(layer)
        q_len = q.shape[2]; kv_seq_len = k.shape[2]
        if self.use_flash_attn:
            is_causal = mask is None and q_len > 1 and q_len == kv_seq_len
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None if is_causal else mask,
                dropout_p=self.dropout.p if self.training else 0.0, is_causal=is_causal)
        else:
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                 relevant_mask = mask[:, :, start_pos : start_pos + q_len, :kv_seq_len]
                 scores = scores + relevant_mask
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn_output = torch.matmul(attn, v)
        out = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.dim)
        out = self.resid_dropout(self.out(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, multiple_of=256, dropout_rate = 0.1):
        super().__init__()
        hidden_dim = int(2 * (4 * dim) / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self, n_heads, dim, dropout_rate = 0.1):
        super().__init__()
        self.attn = mha(dim, n_heads, dropout_rate)
        self.ffn = FeedForward(dim=dim, dropout_rate=dropout_rate)
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    def forward(self, x, layer, cache = None, mask= None, start_pos = 0):
        attn_out = self.attn(self.attn_norm(x), layer, start_pos, cache, mask)
        h = x + attn_out
        ffn_out = self.ffn(self.ffn_norm(h))
        out = h + ffn_out
        return out

class TransformerModel(nn.Module): # Original KV Cache TransformerModel class
    def __init__(self, vocab_size=50257, d_model=256, n_heads=4, n_blocks=4, dropout_rate = 0.1, max_seq_len = 256, init_std=0.02):
        super().__init__()
        self.vocab_size = vocab_size; self.d_model = d_model; self.n_heads = n_heads; self.n_blocks = n_blocks;
        self.dropout_rate = dropout_rate; self.max_seq_len = max_seq_len; self.head_dim = d_model // n_heads; self.init_std = init_std
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim=d_model)
        self.blocks = nn.ModuleList([Block(n_heads=n_heads, dim=d_model, dropout_rate=dropout_rate) for _ in range(n_blocks)])
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout_rate, max_len=max_seq_len)
        self.norm = RMSNorm(d_model); self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.embedding.weight = self.output.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=self.init_std / math.sqrt(2 * self.n_blocks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def forward(self, tokens, cache = None, mask = None, start_pos = 0):
        tokens = tokens.transpose(0, 1) 
        bsz, seq_len = tokens.shape
        h = self.embedding(tokens)
        h = self.pos_encoding(h, start_pos=start_pos) 
        for layer, block in enumerate(self.blocks):
            h = block(h, layer=layer, cache=cache, mask=mask, start_pos=start_pos)
        h = self.norm(h)
        logits = self.output(h)
        logits = logits.transpose(0, 1) 
        return logits

    @staticmethod
    def create_causal_mask(seq_len, device):
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril(diagonal=0)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.unsqueeze(0).unsqueeze(0) 

    def init_kv_cache(self, bsz, device):
         return KVCache(self.n_blocks, bsz, self.max_seq_len, self.n_heads, self.head_dim, device)


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    """
    Implements nucleus sampling (top-p) as described in:
    "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019)
    
    Args:
        logits: tensor of shape (vocab_size,) containing token logits
        p: probability threshold (default 0.95)
        
    Returns:
        Sampled token ID
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find indices where cumulative probability exceeds p
    # We want the smallest k such that the first k tokens have cumulative prob >= p
    nucleus = cumulative_probs < p
    
    # Add one more token to ensure we surpass the threshold p
    # (This handles the case described in the instructions where p(1) + ... + p(k-1) < p <= p(1) + ... + p(k))
    nucleus[-1] = True
    nucleus = torch.cat([nucleus[1:], torch.tensor([False], device=logits.device)])
    
    # Get the indices of tokens in the nucleus
    nucleus_indices = sorted_indices[nucleus]
    
    # Get the probabilities of tokens in the nucleus
    nucleus_probs = sorted_probs[nucleus]
    
    # Renormalize the probabilities
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    
    # Sample from the nucleus
    sample_idx = torch.multinomial(nucleus_probs, num_samples=1).item()
    
    # Get the actual token id
    token_id = nucleus_indices[sample_idx].item()
    
    return token_id

def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a"):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 32
    num_epochs = 1
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(
    ).to(device)

    models = {
      #"kgram_mlp_seq": kgram_model,
       "lstm_seq": lstm_model,
      # "kvcache_transformer": kv_transformer,
      #"transformer": transformer,
    }


    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt  # <--- Pass the user-specified prompt here
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()