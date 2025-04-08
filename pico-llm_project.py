# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple
import matplotlib.pyplot as plt

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

        # fill in

        # Our initial Linear and ReLU layer
        layers = []
        layers.append(nn.Linear(k * vocab_size, embed_size))
        layers.append(nn.SiLU())

        # Now create the remaining inner layers
        for i in range(num_inner_layers):
            layers.append(nn.Linear(embed_size, embed_size))
            layers.append(nn.SiLU())
        
        # Finally, add one final layer to get our vocab
        layers.append(nn.Linear(embed_size, vocab_size))

        # Set net as our layers
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
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
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
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # [d_model / 2]
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer so it's part of the model's state_dict,
        # but not trained by the optimizer. Shape [max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            start_pos: The starting position index for the sequence (used with KV cache)
        """
        seq_len = x.size(1)
        # Retrieve the required positional embeddings [seq_len, d_model]
        # Unsqueeze to add batch dimension for broadcasting: [1, seq_len, d_model]
        pos_enc = self.pe[start_pos : start_pos + seq_len, :].unsqueeze(0)
        x = x + pos_enc
        return self.dropout(x)
    
class KVCache():
    def __init__(self, n_layers, bsz, max_seq_length, n_heads, head_dim):
        self.n_layers = n_layers
        self.bsz = bsz
        self.max_seq_length = max_seq_length
        self.n_heads = n_heads 
        self.head_dim = head_dim 

        self.cache_k: List[torch.Tensor] = []
        self.cache_v: List[torch.Tensor] = []
        self.reset()

    def reset(self):
        self.cache_k = []
        self.cache_v = []
        for _ in range(self.n_layers):
            self.cache_k.append(torch.zeros((self.bsz, self.n_heads, 0, self.head_dim)))
            self.cache_v.append(torch.zeros((self.bsz, self.n_heads, 0, self.head_dim)))
        
    def update(self, layer, new_k, new_v, seq_len):
        self.cache_k[layer] = torch.cat([self.cache_k[layer], new_k], dim=2)
        self.cache_v[layer] = torch.cat([self.cache_v[layer], new_v], dim=2)

    def get(self, layer, seq_len):
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

    def forward(self,x,layer = 0, start_pos = 0, cache = None, mask = None):
        bsz, seq_len, _ = x.shape

        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        q = q.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            cache.update(layer, k, v, start_pos)

            seq_len = start_pos + seq_len
            k, v = cache.get(layer, seq_len)

        q_len = q.shape[2]
        kv_seq_len = k.shape[2]
        
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            relevant_mask = mask[:, :, start_pos : start_pos + q_len, :kv_seq_len]
            scores = scores + relevant_mask     

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        """
        print(f"DEBUG: mha layer {layer}, start_pos {start_pos}")
        print(f"  Input x shape: {x.shape}")
        print(f"  Shape before reshape: {out.shape}")
        print(f"  Target reshape vars: bsz={bsz}, seq_len={seq_len}, self.dim={self.dim}")
        print(f"  Target reshape shape: ({bsz}, {seq_len}, {self.dim})")
        """
        out = out.transpose(1, 2).reshape(bsz, q_len, self.dim)
        out = self.dropout(self.out(out))

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

    def forward(self, x): #swiglu
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x

class Block(nn.Module):
    def __init__(self, n_heads, dim, dropout_rate = 0.1):
        super().__init__()
        self.attn = mha(dim, n_heads, dropout_rate)
        self.ffn = FeedForward(dim=dim, dropout_rate=dropout_rate)
        self.attnNorm = RMSNorm(dim)
        self.ffnNorm = RMSNorm(dim)

    def forward(self,x, layer, cache = None, mask= None, start_pos = 0):
        x = self.attnNorm(x)
        attn = self.attn(x,layer, start_pos, cache, mask)
        x = x + attn
        x = x + self.ffn(self.ffnNorm(x))
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4, dropout_rate = 0.1, max_seq_len = 1024, init_std=0.02):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads
        self.init_std = init_std

        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim=d_model)
        self.blocks = nn.ModuleList([Block(n_heads=n_heads, dim=d_model) for i in range(n_blocks)])
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout_rate, max_len=max_seq_len)

        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.embedding.weight = self.output.weight
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('output.weight') or pn.endswith('w2.weight'): # Attn output proj, FFN down proj
                torch.nn.init.normal_(p, mean=0.0, std=self.init_std / math.sqrt(2 * self.n_blocks))

    def _init_weights(self, module):
        """ Initializes weights according to common practices (e.g., GPT-2)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embeddings initialized slightly differently or same as Linear
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
        # Note: RMSNorm 'weight' parameter is already initialized to 1 by default in its __init__

    def forward(self, x, cache = None, mask = None, start_pos = 0):
        bsz, seq_len = x.shape
        x = self.embedding(x)
        x = self.pos_encoding(x, start_pos)
        for layer, block in enumerate(self.blocks):
            x = block(x, layer, cache, mask, start_pos)
        x = self.norm(x)
        return self.output(x)
    
    def create_causal_mask(seq_len):
        """Creates a causal attention mask."""
        # Creates a lower triangular mask (allowing attention to previous positions)
        mask = torch.full((seq_len, seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1) # Set upper triangle (excluding diagonal) to -inf
         # Add dimensions for batch and heads: [1, 1, seq_len, seq_len]
        return mask.unsqueeze(0).unsqueeze(0)
    
    def init_kv_cache(self, bsz):
         return KVCache(self.n_blocks, bsz, self.max_seq_len, self.n_heads, self.head_dim)


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cutoff index
    cutoff = cumulative_probs > p
    cutoff_indices = torch.where(cutoff)[0]

    if len(cutoff_indices) == 0:
        # If no cutoff found (like p=1.0), keep all tokens
        cutoff_index = len(sorted_probs)
    else:
        cutoff_index = cutoff_indices[0].item() + 1

    top_probs = sorted_probs[:cutoff_index]
    top_indices = sorted_indices[:cutoff_index]

    top_probs = top_probs / top_probs.sum()
    sampled_index = torch.multinomial(top_probs, 1).item()
    return top_indices[sampled_index].item()
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  temperature=1.0, # Added temperature for flexibility
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step:
         - IF TransformerModel: Feed only the last token + use KV cache.
         - ELSE: Feed the entire context as (seq_len,1) to model(...).
      - We get model(...)-> We take the final step's logits => next_logits.
      - We pick next token (greedy or top-p), append to context_tokens list.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    model.to(device) 

    prompt_tokens = enc.encode(init_text)
    context_tokens = list(prompt_tokens) 
    annotation_list = [] 

    is_transformer = isinstance(model, TransformerModel)
    cache = None
    loop_context_tensor = None

    if is_transformer:
        with torch.no_grad():
            loop_context_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
            bsz, current_seq_len = loop_context_tensor.shape if len(prompt_tokens) > 0 else (1, 0)
            cache = model.init_kv_cache(bsz=bsz)
            if current_seq_len > 0:
                prompt_mask = TransformerModel.create_causal_mask(current_seq_len) if current_seq_len > 1 else None
                _ = model(loop_context_tensor, cache=cache, mask=prompt_mask, start_pos=0)

    # --- Generation Loop ---
    with torch.no_grad():
        for step_i in range(max_new_tokens):
            next_logits = None 

            if is_transformer:
                if loop_context_tensor is None or loop_context_tensor.shape[1] == 0:
                    break 
                current_seq_len = loop_context_tensor.shape[1]
                start_pos = current_seq_len - 1
                next_token_input = loop_context_tensor[:, -1:] 
                logits = model(next_token_input, cache=cache, mask=None, start_pos=start_pos)
                next_logits = logits[:, -1, :] 

            else:
                if not context_tokens:
                    break 
                seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
                logits_seq = model(seq_tensor)              
                next_logits = logits_seq[-1, 0, :]        

            if next_logits.dim() == 1:
                next_logits = next_logits.unsqueeze(0)

            chosen_token_item = None 
            chosen_token_tensor = None 

            # Greedy or Nucleus Sampling
            if top_p is None:
                chosen_token_tensor = torch.argmax(next_logits, dim=-1, keepdim=True) 
                chosen_token_item = chosen_token_tensor.item()
            else:
                chosen_token_item = nucleus_sampling(next_logits.squeeze(0), p=top_p)
                chosen_token_tensor = torch.tensor([[chosen_token_item]], dtype=torch.long, device=device)

            context_tokens.append(chosen_token_item)

            if is_transformer:
                loop_context_tensor = torch.cat([loop_context_tensor, chosen_token_tensor], dim=1)
            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token_item, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token_item, neighbors))
            else:
                annotation_list.append((chosen_token_item, []))

            if is_transformer and loop_context_tensor.shape[1] >= model.max_seq_len:
                 print("Warning: Reached transformer max sequence length.")
                 break

    model.train(was_training) 
    final_text = enc.decode(context_tokens)
    num_actually_generated = len(context_tokens) - len(prompt_tokens)
    prefix_text = enc.decode(prompt_tokens) 

    annotated_strs = [prefix_text]
    for i in range(num_actually_generated):
        if i < len(annotation_list):
             tid, neighs = annotation_list[i]
             token_str = enc.decode([tid])
             if neighs: 
                 neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
                 annotated = f"{token_str}[NN={neighbor_strs}]"
             else:
                 annotated = token_str
             annotated_strs.append(annotated)
        else:
             print(f"Warning: Mismatch between generated tokens and annotation list at index {i}")

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

    epoch_avg_losses = []
    partial_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1
            epoch_loss = 0.0

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_loss += loss.item()
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
        epoch_avg_losses.append(avg_loss)
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")

    print(f"Training finished for {model_name}.")
    # Plot the collected losses
    plot_losses(epoch_avg_losses, model_name)


################################################################################
# 8.5 Loss plot
################################################################################

def plot_losses(epoch_losses, model_name):
    """Plots the average loss per epoch and loss at specific steps."""
    plt.figure(figsize=(12, 5))

    # Plot average loss per epoch
    plt.subplot(1, 2, 1)
    epochs = range(1, len(epoch_losses) + 1)
    plt.plot(epochs, epoch_losses, marker='o', linestyle='-', label='Avg Epoch Loss')
    plt.title(f'{model_name} - Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.xticks(epochs) # Ensure integer ticks for epochs
    plt.grid(True)
    plt.legend()

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig(f"{model_name}_loss_curves.png") # Save the plot
    print(f"Saved loss plot to {model_name}_loss_curves.png")
    plt.show() # Optionally display the plot interactively


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 3
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

    #transformer = TransformerModel(
    #).to(device)

    transformer = TransformerModel(
        vocab_size=vocab_size,
        d_model=256,  # Smaller than original 1024
        n_heads=4,
        n_blocks=2
    ).to(device)

    #Uncomment the ones you want to run, we did not implement a kvcache_transformer
    models = {
<<<<<<< Updated upstream
    #   "kgram_mlp_seq": kgram_model,
    #   "lstm_seq": lstm_model,
=======
       #"kgram_mlp_seq": kgram_model,
       "lstm_seq": lstm_model,
>>>>>>> Stashed changes
      # "kvcache_transformer": kv_transformer,
       # "transformer": transformer
    }


    ############################################################################
    # Train each model
    ############################################################################

    print("\n=== Testing Transformer with dummy input ===")
    dummy_input = torch.randint(0, vocab_size, (10, 1), device=device)  # (seq_len=10, batch=1)
    dummy_output = transformer(dummy_input)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Dummy output shape: {dummy_output.shape}")
    print("Transformer basic test passed!\n")
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