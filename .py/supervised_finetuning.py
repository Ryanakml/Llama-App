import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import List, Dict, Any

# Import model dari file llama_model.py
from llama_model import LLaMAModel, SimpleBPETokenizer

# ===== DATASET UNTUK SFT =====
class QADataset(Dataset):
    """Dataset untuk supervised fine-tuning dengan format QA"""
    def __init__(self, qa_data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Format dan tokenize data QA
        for item in qa_data:
            # Format: "Question: {question} Answer: {answer}"
            text = f"Question: {item['instruction']} Answer: {item['output']}"
            tokens = tokenizer.encode(text)
            
            if len(tokens) > 2:  # Pastikan ada konten
                self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Truncate jika perlu
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Buat input_ids dan labels
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # Padding
        padding_length = self.max_length - 1 - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.special_tokens["<pad>"]] * padding_length
            labels = labels + [-100] * padding_length
        
        # Attention mask
        attention_mask = [1] * (len(tokens) - 1) + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask)
        }

# ===== SUPERVISED FINE-TUNING FUNCTION =====
def supervised_finetune(model, tokenizer, qa_data, batch_size=4, epochs=3, lr=2e-5, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Fungsi untuk supervised fine-tuning model LLaMA"""
    print(f"Supervised fine-tuning on {device}...")
    model.to(device)
    
    # Siapkan dataset dan dataloader
    dataset = QADataset(qa_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer dengan learning rate lebih kecil untuk fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
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
            
            total_loss += loss.item()
        
        # Log progress
        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    return model

# ===== CONTOH DATA QA =====
def prepare_qa_data():
    """Buat contoh dataset QA sederhana"""
    qa_data = [
        {"instruction": "Apa itu AI?", "output": "AI adalah kecerdasan buatan yang dirancang untuk meniru kecerdasan manusia dalam menyelesaikan tugas."},
        {"instruction": "Jelaskan gunung tertinggi di dunia", "output": "Gunung Everest adalah gunung tertinggi di dunia dengan ketinggian 8.848 meter di atas permukaan laut."},
        {"instruction": "Siapa penemu listrik?", "output": "Benjamin Franklin dikenal karena eksperimennya dengan listrik, meskipun tidak tepat menyebutnya sebagai penemu listrik karena listrik adalah fenomena alam."},
        {"instruction": "Apa ibu kota Indonesia?", "output": "Jakarta adalah ibu kota Indonesia, terletak di pulau Jawa."},
        {"instruction": "Berapa jumlah planet di tata surya?", "output": "Ada delapan planet di tata surya: Merkurius, Venus, Bumi, Mars, Jupiter, Saturnus, Uranus, dan Neptunus."}
    ]
    return qa_data

# ===== FUNGSI UNTUK GENERATE TEKS =====
def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Generate teks dari model yang sudah di-fine-tune"""
    model.eval()
    model.to(device)
    
    # Tokenize prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    
    # Generate token satu per satu
    for _ in range(max_length):
        # Buat attention mask
        attention_mask = torch.ones(input_ids.shape, device=device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        
        # Ambil logits untuk token terakhir
        next_token_logits = logits[:, -1, :]
        
        # Aplikasikan temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        # Sample dari distribusi
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Tambahkan ke input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Hentikan jika generate EOS token
        if next_token.item() == tokenizer.special_tokens["<eos>"]:
            break
    
    # Decode hasil
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text