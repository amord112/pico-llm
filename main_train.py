# File: main_kv_subset.py
# Trains the KV Cache Transformer model on a subset of TinyStories
# Includes train/test split and evaluation.
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from functools import partial

from datasets import load_dataset # Needed for TinyStories
import tiktoken

# --- Constants ---
EARLY_STOP_THRESHOLD = 5.2 # The loss threshold for early stopping

################################################################################
# 1. Command-line arg parsing (Adapted for Subset Training)
################################################################################
# --- No changes needed here ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train KV Cache Transformer model on a TinyStories subset.")
    # Data Args
    parser.add_argument("--tinystories_subset", type=str, default="train[:5000]", # Default to 1000 examples
                        help="Load a subset/split of TinyStories (e.g., 'train[:1%]', 'train[:5000]'). Default='train[:5000]'.")
    parser.add_argument("--test_split_ratio", type=float, default=0.1,
                        help="Fraction of the loaded subset data to use for testing (0.0 to 1.0). Default=0.1.")
    parser.add_argument("--block_size", type=int, default=256,
                        help="Maximum sequence length for examples. Default=256.")
    # Model Args
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension. Default=256.")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads. Default=4.") # Changed default
    parser.add_argument("--n_blocks", type=int, default=4, help="Number of transformer blocks. Default=4.") # Changed default
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate. Default=0.1.") # Changed default
    # Training Args
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per device. Default=16.") # Changed default
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs. Default=10.") # Increased epochs
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (AdamW). Default=3e-4.") # Adjusted LR
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW. Default=0.01.") # Adjusted WD
    parser.add_argument("--log_steps", type=int, default=1, help="Log training loss every N steps. Default=20.") # Log more often
    parser.add_argument("--sample_interval_epochs", type=int, default=1, help="Generate text samples every N epochs. Default=1.")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None, help="Limit steps per epoch for quick tests.")
    # Generation Args
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for generation.")
    parser.add_argument("--gen_max_new_tokens", type=int, default=50, help="Max new tokens for generation samples.")
    parser.add_argument("--gen_top_p", type=float, default=0.8, help="Top-p (nucleus) sampling probability.")
    parser.add_argument("--gen_temperature", type=float, default=1.0, help="Generation temperature.")
    # System Args
    parser.add_argument("--device_id", type=str, default="cuda:0", help="Torch device ('cuda:0', 'cpu').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")

    args = parser.parse_args()
    args.embed_size = args.d_model # Alias for consistency if needed elsewhere
    return args

################################################################################
# 2. Data handling (Original Structure - Seq Len First Collate)
################################################################################
# --- No changes needed here ---
class SimpleTokenDataset(Dataset):
    def __init__(self, sequences): self.sequences = sequences
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return torch.tensor(self.sequences[idx], dtype=torch.long)

def seq_collate_fn(batch, pad_token_id=0):
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)
    padded = torch.full((max_len, batch_size), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq
    return padded

################################################################################
# 3. Loss Function (Original Structure - Seq Len First)
################################################################################
# --- No changes needed here ---
def compute_next_token_loss(logits, tokens, ignore_index=-100):
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2: return torch.tensor(0.0, device=logits.device, requires_grad=True)
    preds = logits[:-1, :, :]
    gold = tokens[1:, :]
    loss = F.cross_entropy(preds.reshape(-1, vocab_size), gold.reshape(-1), ignore_index=ignore_index)
    return loss

################################################################################
# 4. KV Cache Transformer Model Definition (Copied from main_train_original_kv.py)
################################################################################
# --- Assume model definitions are present and unchanged ---
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
        # Input tokens: (seq_len, batch) from original loader
        tokens = tokens.transpose(0, 1) # Transpose to (batch, seq_len)
        bsz, seq_len = tokens.shape
        h = self.embedding(tokens)
        h = self.pos_encoding(h, start_pos=start_pos) # PE expects batch_first
        for layer, block in enumerate(self.blocks):
            h = block(h, layer=layer, cache=cache, mask=mask, start_pos=start_pos) # Blocks expect batch_first
        h = self.norm(h)
        logits = self.output(h) # (batch, seq_len, vocab_size)
        logits = logits.transpose(0, 1) # Transpose back to (seq_len, batch, vocab_size)
        return logits

    @staticmethod
    def create_causal_mask(seq_len, device):
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril(diagonal=0)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.unsqueeze(0).unsqueeze(0) # Add batch and head dims

    def init_kv_cache(self, bsz, device):
         return KVCache(self.n_blocks, bsz, self.max_seq_len, self.n_heads, self.head_dim, device)
# --- End Model Definitions ---

################################################################################
# 5. Text Generation (KV Cache Version from main_train_original_kv.py)
################################################################################
# --- No changes needed here ---
def nucleus_sampling(logits, p=0.95):
    if p >= 1.0: probs = F.softmax(logits, dim=-1); return torch.multinomial(probs, 1).item()
    if p <= 0: return torch.argmax(logits).item()
    probs = F.softmax(logits, dim=-1); sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1); cutoff_mask = cumulative_probs > p
    cutoff_indices = torch.where(cutoff_mask)[0]
    cutoff_index = cutoff_indices[0].item() + 1 if len(cutoff_indices) > 0 else len(sorted_probs)
    top_probs = sorted_probs[:cutoff_index]; top_indices = sorted_indices[:cutoff_index]
    if top_probs.numel() == 0: return sorted_indices[0].item()
    renormalized_probs = top_probs / top_probs.sum()
    sampled_rel_index = torch.multinomial(renormalized_probs, 1).item()
    return top_indices[sampled_rel_index].item()

def generate_text(model, enc, init_text, max_new_tokens=50, device="cpu",
                  top_p=0.95, temperature=1.0):
    was_training = model.training; model.eval(); model.to(device)
    is_transformer = isinstance(model, TransformerModel) # Check if KV Cache model
    prompt_tokens = enc.encode(init_text)
    context_tokens_list = list(prompt_tokens)
    cache = None; transformer_context_tensor = None
    if is_transformer:
        with torch.no_grad():
             transformer_context_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
             if transformer_context_tensor.shape[1] > 0:
                 cache = model.init_kv_cache(bsz=1, device=device)
                 prompt_len = transformer_context_tensor.shape[1]
                 prefill_mask = model.create_causal_mask(prompt_len, device=device) if prompt_len > 1 else None
                 _ = model(transformer_context_tensor.transpose(0,1), cache=cache, mask=prefill_mask, start_pos=0) # Prefill call
    generated_ids_for_annotation = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            next_logits = None
            if is_transformer:
                if transformer_context_tensor is None or transformer_context_tensor.shape[1] == 0: break
                current_seq_len = transformer_context_tensor.shape[1]
                start_pos = current_seq_len - 1 # Generate from the last position
                last_token_tensor_bf = transformer_context_tensor[:, -1:] # Batch first
                # Pass mask=None during generation steps (KV cache handles causality)
                logits_seq_len_first = model(last_token_tensor_bf.transpose(0,1), cache=cache, mask=None, start_pos=start_pos)
                next_logits = logits_seq_len_first.squeeze() # Get logits for the next token
            else: # Fallback logic (won't be used if model is TransformerModel)
                if not context_tokens_list: break
                context_tensor_sl = torch.tensor(context_tokens_list, dtype=torch.long, device=device).unsqueeze(1)
                # This assumes a non-transformer model takes seq_len_first
                if hasattr(model, 'max_seq_len') and context_tensor_sl.shape[0] > model.max_seq_len: break # Check length
                logits_seq = model(context_tensor_sl)
                next_logits = logits_seq[-1, 0, :]
            if temperature != 1.0: next_logits = next_logits / temperature
            chosen_token_item = nucleus_sampling(next_logits, p=top_p)
            context_tokens_list.append(chosen_token_item)
            generated_ids_for_annotation.append(chosen_token_item)
            if is_transformer:
                chosen_token_tensor = torch.tensor([[chosen_token_item]], dtype=torch.long, device=device)
                transformer_context_tensor = torch.cat([transformer_context_tensor, chosen_token_tensor], dim=1)
            if chosen_token_item == enc.eot_token: break
            if is_transformer and transformer_context_tensor.shape[1] >= model.max_seq_len: break
    model.train(was_training); final_text = enc.decode(context_tokens_list)
    return final_text, final_text # Return text twice
# --- End Generation ---

################################################################################
# 6. Training & Evaluation Functions (Restored Early Stopping Logic)
################################################################################

def evaluate(model, loader, device, pad_token_id, static_mask=None):
    # --- No changes needed here ---
    model.eval(); total_loss = 0.0
    with torch.no_grad():
        for batch_tokens in loader:
            batch_tokens = batch_tokens.to(device) # (seq_len, batch)
            current_seq_len = batch_tokens.shape[0]
            current_mask = None
            if static_mask is not None and current_seq_len <= static_mask.shape[-1]:
                 current_mask = static_mask[:, :, :current_seq_len, :current_seq_len]
            logits = model(batch_tokens, cache=None, mask=current_mask, start_pos=0)
            loss = compute_next_token_loss(logits, batch_tokens, ignore_index=pad_token_id)
            total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0.0

def train_one_model(model, train_loader, test_loader, epochs, model_name, device, lr=1e-3, weight_decay=0.01,
                    log_steps=100, sample_interval_epochs=1, max_steps_per_epoch=None, enc=None,
                    prompt="Once upon a time", gen_max_new_tokens=50, gen_top_p=0.9, gen_temperature=1.0,
                    pad_token_id=0, early_stop_threshold=None): # <<< Restored parameter name
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_epoch_losses = []; test_epoch_losses = []; global_step = 0
    intermediate_train_losses = []; intermediate_global_steps = []
    # <<< Restored early stop variables >>>
    early_stop_triggered = False
    early_stop_step = -1
    early_stop_epoch = -1

    print(f"\n=== Starting Training for {model_name} ===")
    train_mask = None
    if isinstance(model, TransformerModel) and model.max_seq_len > 1:
         train_mask = model.create_causal_mask(model.max_seq_len, device=device)

    for epoch in range(1, epochs + 1):
        model.train(); total_train_loss = 0.0; log_loss_tracker = 0.0; log_start_time = time.time(); step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(train_loader, start=1):
            step_in_epoch += 1; global_step += 1
            batch_tokens = batch_tokens.to(device)
            current_seq_len = batch_tokens.shape[0]
            current_mask = None
            if train_mask is not None and current_seq_len <= train_mask.shape[-1]:
                 current_mask = train_mask[:, :, :current_seq_len, :current_seq_len]

            logits = model(batch_tokens, cache=None, mask=current_mask, start_pos=0)
            loss = compute_next_token_loss(logits, batch_tokens, ignore_index=pad_token_id)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            batch_loss = loss.item(); total_train_loss += batch_loss; log_loss_tracker += batch_loss

            if global_step % log_steps == 0:
                avg_log_loss = log_loss_tracker / log_steps if log_steps > 0 else batch_loss
                elapsed = time.time() - log_start_time
                steps_per_sec = log_steps / elapsed if elapsed > 0 else 0
                print(f"[{model_name}] Ep{epoch}/{epochs}, Step{step_in_epoch}/{len(train_loader)} (Glob:{global_step}) | Loss:{avg_log_loss:.4f} | Step/s:{steps_per_sec:.2f}")
                intermediate_train_losses.append(avg_log_loss)
                intermediate_global_steps.append(global_step)
                log_loss_tracker = 0.0; log_start_time = time.time()

                # <<< Check for early stopping condition >>>
                if early_stop_threshold is not None and avg_log_loss < early_stop_threshold:
                    print(f"--- Early stopping triggered at global step {global_step} (Epoch {epoch}) ---")
                    print(f"    Average loss {avg_log_loss:.4f} < threshold {early_stop_threshold}")
                    early_stop_triggered = True
                    early_stop_step = global_step
                    early_stop_epoch = epoch
                    break # <<< RE-ADDED break: Exit inner (batch) loop >>>

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"Reached max_steps_per_epoch ({max_steps_per_epoch}), ending epoch early.")
                break # Exit inner (batch) loop - this is for step limit

        # Record epoch losses (even if partial due to early stop/max_steps)
        # Only record if the loop ran at least once for this epoch
        if step_in_epoch > 0:
            avg_epoch_train_loss = total_train_loss / step_in_epoch
            train_epoch_losses.append(avg_epoch_train_loss)
            # Evaluate test loss only if the epoch wasn't stopped early *during* the training loop
            # Or if it was the epoch where stopping occurred (to get final test loss)
            avg_epoch_test_loss = float('nan')
            if test_loader and (not early_stop_triggered or epoch == early_stop_epoch):
                avg_epoch_test_loss = evaluate(model, test_loader, device, pad_token_id, train_mask)
                test_epoch_losses.append(avg_epoch_test_loss)
            elif test_loader: # Append NaN if stopped in a previous epoch
                 test_epoch_losses.append(float('nan'))


            # Don't print end epoch summary if early stopping happened in this epoch's inner loop
            if not early_stop_triggered:
                 print(f"[{model_name}] *** End Epoch {epoch} *** Train Loss: {avg_epoch_train_loss:.4f} | Test Loss: {avg_epoch_test_loss:.4f}")
            elif epoch == early_stop_epoch: # Print final summary if stopping occurred this epoch
                 print(f"[{model_name}] *** Stopping Epoch {epoch} *** Final Train Loss (partial): {avg_epoch_train_loss:.4f} | Final Test Loss: {avg_epoch_test_loss:.4f}")


        # Generate samples only if epoch completed normally and interval matches
        if enc is not None and epoch % sample_interval_epochs == 0 and not early_stop_triggered:
             print(f"\n[{model_name}] Generating sample text @ Epoch {epoch}...")
             text_sample, _ = generate_text(model, enc, prompt, max_new_tokens=gen_max_new_tokens, device=device, top_p=gen_top_p, temperature=gen_temperature)
             print(f" Prompt: '{prompt}'\n Sample: {text_sample}\n")

        # <<< RE-ADDED check to exit outer (epoch) loop >>>
        if early_stop_triggered:
            break

    print(f"Training finished for {model_name}.")
    # <<< Return early stopping info >>>
    return train_epoch_losses, test_epoch_losses, intermediate_global_steps, intermediate_train_losses, early_stop_step, early_stop_epoch

################################################################################
# 7. Loss Plotting (Restored Version for Early Stopping Marker)
################################################################################

def plot_losses(train_losses, test_losses, model_name):
    # --- No changes needed here ---
    if not train_losses and not any(not math.isnan(l) for l in test_losses): print("No loss data to plot."); return
    plt.figure(figsize=(10, 6)); epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker='o', linestyle='-', label='Avg Train Loss')
    valid_test_losses = [(i, loss) for i, loss in enumerate(test_losses, 1) if not math.isnan(loss)]
    if valid_test_losses:
        test_epochs, test_vals = zip(*valid_test_losses)
        plt.plot(test_epochs, test_vals, marker='s', linestyle='--', label='Avg Test Loss')
    plt.title(f'{model_name} - Average Loss per Epoch'); plt.xlabel('Epoch'); plt.ylabel('Average Loss')
    if epochs: plt.xticks(list(epochs)); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"{model_name}_loss_curves_epoch.png"); print(f"Saved epoch loss plot to {model_name}_loss_curves_epoch.png")
    plt.close() # Close plot

# <<< Restored parameters for early stopping marker >>>
def plot_intermediate_losses(steps, losses, model_name, log_steps_freq,
                             early_stop_step=None, early_stop_threshold=None):
    """Plots loss values against global training steps."""
    if not steps or not losses:
        print("No intermediate loss data to plot.")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, marker='.', linestyle='-', markersize=4, label=f'Avg Train Loss (every {log_steps_freq} steps)')

    # <<< Add vertical line for early stopping >>>
    if early_stop_step is not None and early_stop_step > 0:
        label_text = f'Early Stop @ {early_stop_step}'
        if early_stop_threshold is not None:
             label_text += f' (Loss < {early_stop_threshold:.1f})'
        plt.axvline(x=early_stop_step, color='r', linestyle='--', label=label_text)
        # Optionally, add a point marker at the stop location
        try:
            stop_idx = steps.index(early_stop_step)
            plt.plot(early_stop_step, losses[stop_idx], 'ro', markersize=8) # Red circle
        except ValueError:
            print(f"Warning: Could not find exact step {early_stop_step} in intermediate steps for marking.")


    plt.title(f'{model_name} - Intermediate Training Loss per Step')
    plt.xlabel('Global Training Step')
    plt.ylabel(f'Average Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Use a specific filename for the intermediate plot
    plt.savefig(f"{model_name}_intermediate_loss_curve_steps.png")
    print(f"Saved intermediate loss plot to {model_name}_intermediate_loss_curve_steps.png")
    # plt.show() # Uncomment to display plot interactively
    plt.close() # Close the plot to free memory

################################################################################
# 8. Main Execution (Restored Early Stopping Logic)
################################################################################

def main():
    args = parse_args()

    # --- Setup ---
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    if args.device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA specified but not available. Using CPU."); device = torch.device("cpu")
    else: device = torch.device(args.device_id)
    print(f"Using device: {device}")
    print(f"Block size (max_seq_len): {args.block_size}")
    print(f"Early stopping threshold: {EARLY_STOP_THRESHOLD}") # <<< Print the threshold

    ############################################################################
    # Data Loading & Preparation (Loads TinyStories Subset)
    ############################################################################
    # --- No changes needed here ---
    print("\n--- Loading and Preparing Data ---")
    all_sequences = []
    enc = tiktoken.get_encoding("gpt2"); vocab_size = enc.n_vocab; pad_token_id = enc.eot_token
    print(f"Tokenizer: gpt2 | Vocab size: {vocab_size} | Pad token ID: {pad_token_id}")
    print(f"Loading TinyStories dataset (subset: {args.tinystories_subset})...")
    try:
        ts_dataset = load_dataset("roneneldan/TinyStories", split=args.tinystories_subset)
        print(f"Tokenizing {len(ts_dataset)} TinyStories examples (block_size={args.block_size})...")
        count = 0
        for sample in ts_dataset:
             text = sample['text']
             if text:
                 tokens = enc.encode(text)[:args.block_size] # Tokenize and truncate
                 if len(tokens) > 1: all_sequences.append(tokens); count += 1
                 if count % 10000 == 0 and count > 0: print(f"  ... tokenized {count} stories")
        print(f"Finished tokenizing. Found {len(all_sequences)} valid sequences.")
    except Exception as e:
        print(f"Error loading/tokenizing TinyStories: {e}. Exiting."); return
    if not all_sequences: print("FATAL: No data sequences loaded. Exiting."); return
    print(f"Total sequences for dataset: {len(all_sequences)}")
    full_dataset = SimpleTokenDataset(all_sequences)
    test_size = int(len(full_dataset) * args.test_split_ratio)
    train_size = len(full_dataset) - test_size
    if train_size <= 0 or test_size <= 0:
         print(f"Warning: Dataset size ({len(full_dataset)}) too small for split. Using all data for training.")
         train_dataset = full_dataset; test_dataset = None
         train_size = len(train_dataset)
    else:
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(args.seed))
        print(f"Split dataset: Train={len(train_dataset)}, Test={len(test_dataset)}")
    collate_fn_with_pad = partial(seq_collate_fn, pad_token_id=pad_token_id)
    effective_batch_size = min(args.batch_size, train_size) if train_size > 0 else args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_with_pad, pin_memory=(device.type == 'cuda'))
    test_loader = None
    if test_dataset:
        effective_test_batch_size = min(args.batch_size * 2, test_size) if test_size > 0 else args.batch_size * 2
        test_loader = DataLoader(test_dataset, batch_size=effective_test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_with_pad, pin_memory=(device.type == 'cuda'))
    print("DataLoaders created.")

    ############################################################################
    # Model Initialization (KV Cache Transformer)
    ############################################################################
    # --- No changes needed here ---
    print("\n--- Initializing KV Cache Model ---")
    model = TransformerModel( # Use the KV Cache version
        vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        n_blocks=args.n_blocks, dropout_rate=args.dropout_rate, max_seq_len=args.block_size
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"KV Cache TransformerModel initialized. Parameter count: {num_params/1e6:.2f}M")

    ############################################################################
    # Training Loop
    ############################################################################
    model_name = f"TransformerKV_Subset_ES{EARLY_STOP_THRESHOLD:.1f}" # Specific name including ES threshold

    # <<< Call train_one_model and capture early stopping results >>>
    train_losses, test_losses, inter_steps, inter_losses, stopped_step, stopped_epoch = train_one_model(
        model=model, train_loader=train_loader, test_loader=test_loader, epochs=args.num_epochs,
        model_name=model_name, device=device, lr=args.learning_rate, weight_decay=args.weight_decay,
        log_steps=args.log_steps, sample_interval_epochs=args.sample_interval_epochs, max_steps_per_epoch=args.max_steps_per_epoch,
        enc=enc, prompt=args.prompt, gen_max_new_tokens=args.gen_max_new_tokens, gen_top_p=args.gen_top_p,
        gen_temperature=args.gen_temperature, pad_token_id=pad_token_id,
        early_stop_threshold=EARLY_STOP_THRESHOLD # Pass the threshold
    )

    ############################################################################
    # Final Steps
    ############################################################################
    print("\n--- Post-Training ---")
    # <<< Report if early stopping occurred >>>
    if stopped_step > 0:
        print(f"EARLY STOPPING triggered at Epoch {stopped_epoch}, Global Step {stopped_step}.") # Indicate stopping happened
    else:
        print("Training completed without triggering early stopping.")

    # Plotting - pass early stopping info
    plot_losses(train_losses, test_losses, model_name) # Plot train/test losses per epoch
    plot_intermediate_losses(inter_steps, inter_losses, model_name, args.log_steps,
                             early_stop_step=stopped_step, early_stop_threshold=EARLY_STOP_THRESHOLD) # Plot intermediate loss per step

    print("\n--- Final Generation Sample (Using model state at end of training/stopping) ---") # Clarify state
    final_text, _ = generate_text(model, enc, args.prompt, max_new_tokens=args.gen_max_new_tokens * 2, device=device, top_p=args.gen_top_p, temperature=args.gen_temperature)
    print(f"Prompt: '{args.prompt}'\nGenerated: {final_text}"); print("-" * 50)
    print("\n*** Script Finished ***")

if __name__ == "__main__":
    main()