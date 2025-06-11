import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader

# ===== TOKENIZER =====
class SimpleBPETokenizer:
    """Implementasi sederhana dari BPE tokenizer"""
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        
        # Inisialisasi token spesial
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
        self.vocab_size_actual = len(self.special_tokens)  # Mulai dari jumlah token spesial
    
    def train(self, texts, min_freq=2, max_tokens=None):
        """Melatih tokenizer dengan algoritma BPE sederhana"""
        # Mulai dengan karakter sebagai token dasar
        char_freq = {}
        for text in texts:
            for char in text:
                if char not in char_freq:
                    char_freq[char] = 0
                char_freq[char] += 1
        
        # Tambahkan karakter yang memenuhi frekuensi minimum ke vocabulary
        for char, freq in char_freq.items():
            if freq >= min_freq and self.vocab_size_actual < self.vocab_size:
                self.token_to_id[char] = self.vocab_size_actual
                self.id_to_token[self.vocab_size_actual] = char
                self.vocab_size_actual += 1
        
        # Implementasi BPE sederhana
        # Dalam implementasi lengkap, kita akan menggabungkan pasangan yang paling sering muncul
        # dan memperbarui vocabulary sampai mencapai ukuran yang diinginkan
        # Untuk kesederhanaan, kita hanya menggunakan karakter individual
        
        print(f"Tokenizer trained with {self.vocab_size_actual} tokens")
    
    def encode(self, text):
        """Encode teks menjadi token IDs"""
        # Implementasi sederhana: encode per karakter
        ids = [self.special_tokens["<bos>"]]
        for char in text:
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            else:
                ids.append(self.special_tokens["<unk>"])
        ids.append(self.special_tokens["<eos>"])
        return ids
    
    def decode(self, ids):
        """Decode token IDs menjadi teks"""
        text = ""
        for id in ids:
            if id in self.id_to_token and id not in [0, 2, 3]:  # Skip pad, bos, eos
                text += self.id_to_token[id]
        return text

# ===== POSITIONAL EMBEDDING =====
class RotaryPositionalEmbedding(nn.Module):
    """Implementasi Rotary Positional Embedding (RoPE) seperti di LLaMA"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute freqs
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, freqs)  # [seq_len, dim/2]
        
        # Cache cos dan sin untuk efisiensi
        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))
    
    def forward(self, x, seq_len=None):
        # x: [batch, seq_len, dim]
        seq_len = x.shape[1] if seq_len is None else seq_len
        cos = self.cos_cached[:seq_len].view(1, seq_len, -1)  # [1, seq, dim/2]
        sin = self.sin_cached[:seq_len].view(1, seq_len, -1)  # [1, seq, dim/2]
        
        # Reshape untuk rotasi
        x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
        
        # Aplikasikan rotasi
        # [batch, seq, dim/2] * [1, seq, dim/2] -> [batch, seq, dim/2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        
        # Reshape kembali
        rx = torch.stack([rx1, rx2], dim=-1).reshape(*x.shape)
        
        return rx

# ===== ATTENTION =====
class LLaMAAttention(nn.Module):
    """Multi-head attention dengan RoPE seperti di LLaMA"""
    def __init__(self, dim, num_heads=4, head_dim=None, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        
        # Proyeksi query, key, value
        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Rotary positional embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Proyeksi dan reshape untuk multi-head
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Aplikasikan RoPE pada query dan key
        q = self.rope(q.reshape(-1, seq_len, self.head_dim)).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        k = self.rope(k.reshape(-1, seq_len, self.head_dim)).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, heads, seq, seq]
        
        # Causal mask (autoregressive)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Attention mask jika ada
        if attention_mask is not None:
            scores.masked_fill_(~attention_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # Softmax dan dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aplikasikan attention weights ke values
        context = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]
        
        # Reshape dan proyeksi output
        context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.o_proj(context)
        
        return output

# ===== FEEDFORWARD =====
class LLaMAFeedForward(nn.Module):
    """Feedforward dengan SwiGLU activation seperti di LLaMA"""
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 4 * dim
        
        # SwiGLU membutuhkan dua proyeksi
        self.gate_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU activation: gate * GELU(up)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwiGLU: gate * GELU(up)
        activated = gate * F.gelu(up)
        
        # Proyeksi kembali ke dimensi asli
        output = self.down_proj(activated)
        output = self.dropout(output)
        
        return output

# ===== LAYER NORMALIZATION =====
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization seperti di LLaMA"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # RMSNorm: x * w / sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight

# ===== TRANSFORMER LAYER =====
class LLaMALayer(nn.Module):
    """Layer transformer dengan pre-normalization seperti di LLaMA"""
    def __init__(self, dim, num_heads, ffn_dim=None, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Pre-normalization untuk attention
        self.attention_norm = RMSNorm(dim)
        self.attention = LLaMAAttention(dim, num_heads, dropout=dropout)
        
        # Pre-normalization untuk feedforward
        self.ffn_norm = RMSNorm(dim)
        self.ffn = LLaMAFeedForward(dim, ffn_dim, dropout=dropout)
    
    def forward(self, x, attention_mask=None):
        # Pre-norm + attention + residual
        x = x + self.attention(self.attention_norm(x), attention_mask)
        
        # Pre-norm + feedforward + residual
        x = x + self.ffn(self.ffn_norm(x))
        
        return x

# ===== FULL MODEL =====
class LLaMAModel(nn.Module):
    """Model LLaMA lengkap dengan embedding, layers, dan language modeling head"""
    def __init__(self, 
                 vocab_size, 
                 dim=512, 
                 num_layers=6, 
                 num_heads=8, 
                 max_seq_len=1024, 
                 ffn_dim=None, 
                 dropout=0.1):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LLaMALayer(dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights antara embedding dan lm_head
        self.lm_head.weight = self.token_embedding.weight
        
        # Inisialisasi parameter
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        x = self.token_embedding(input_ids)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits

# ===== PRETRAINING DATASET =====
class TextDataset(Dataset):
    """Dataset untuk pretraining"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Tokenize semua teks
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 2:  # Pastikan ada konten (bukan hanya BOS+EOS)
                self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Truncate atau pad sesuai kebutuhan
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Buat input_ids dan labels untuk causal language modeling
        input_ids = tokens[:-1]  # Semua kecuali token terakhir
        labels = tokens[1:]      # Semua kecuali token pertama
        
        # Padding
        padding_length = self.max_length - 1 - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.special_tokens["<pad>"]] * padding_length
            labels = labels + [-100] * padding_length  # -100 akan diabaikan oleh loss function
        
        # Buat attention mask (1 untuk token asli, 0 untuk padding)
        attention_mask = [1] * (len(tokens) - 1) + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask)
        }

# ===== PRETRAINING FUNCTION =====
def pretrain_llama(model, tokenizer, texts, batch_size=8, epochs=3, lr=5e-5, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Fungsi untuk pretraining model LLaMA"""
    print(f"Pretraining on {device}...")
    model.to(device)
    
    # Siapkan dataset dan dataloader
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer dan scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(dataloader))
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Pindahkan batch ke device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Hitung loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, model.vocab_size), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Log progress
        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    return model