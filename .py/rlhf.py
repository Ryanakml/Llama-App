import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Dict, Tuple, Any

# Import model dari file sebelumnya
from llama_model import LLaMAModel, SimpleBPETokenizer

# ===== REWARD MODEL =====
class RewardModel(nn.Module):
    """Model untuk memprediksi reward berdasarkan preferensi manusia"""
    def __init__(self, llama_model):
        super().__init__()
        self.llama = llama_model
        
        # Freeze LLaMA model parameters
        for param in self.llama.parameters():
            param.requires_grad = False
        
        # Reward head: satu nilai scalar untuk setiap sequence
        self.reward_head = nn.Linear(llama_model.dim, 1)
    
    def forward(self, input_ids, attention_mask=None):
        # Dapatkan representasi dari LLaMA
        with torch.no_grad():
            # Ambil hidden states dari model (sebelum lm_head)
            logits = self.llama(input_ids, attention_mask)
            hidden_states = self.llama.norm(self.llama.layers[-1].ffn(self.llama.layers[-1].ffn_norm(logits)))
        
        # Ambil representasi token terakhir
        last_hidden = hidden_states[:, -1, :]
        
        # Prediksi reward
        reward = self.reward_head(last_hidden)
        
        return reward

# ===== DATASET UNTUK REWARD MODEL =====
class PreferenceDataset(Dataset):
    """Dataset berisi pasangan respons dengan preferensi"""
    def __init__(self, preference_data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for item in preference_data:
            # Tokenize prompt dan kedua respons
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]
            
            # Format: "Question: {prompt} Answer: {response}"
            chosen_text = f"Question: {prompt} Answer: {chosen}"
            rejected_text = f"Question: {prompt} Answer: {rejected}"
            
            chosen_tokens = tokenizer.encode(chosen_text)
            rejected_tokens = tokenizer.encode(rejected_text)
            
            # Truncate jika perlu
            if len(chosen_tokens) > self.max_length:
                chosen_tokens = chosen_tokens[:self.max_length]
            if len(rejected_tokens) > self.max_length:
                rejected_tokens = rejected_tokens[:self.max_length]
            
            # Padding
            chosen_attn_mask = [1] * len(chosen_tokens)
            rejected_attn_mask = [1] * len(rejected_tokens)
            
            if len(chosen_tokens) < self.max_length:
                pad_length = self.max_length - len(chosen_tokens)
                chosen_tokens = chosen_tokens + [tokenizer.special_tokens["<pad>"]] * pad_length
                chosen_attn_mask = chosen_attn_mask + [0] * pad_length
            
            if len(rejected_tokens) < self.max_length:
                pad_length = self.max_length - len(rejected_tokens)
                rejected_tokens = rejected_tokens + [tokenizer.special_tokens["<pad>"]] * pad_length
                rejected_attn_mask = rejected_attn_mask + [0] * pad_length
            
            self.examples.append({
                "chosen_input_ids": torch.tensor(chosen_tokens),
                "chosen_attention_mask": torch.tensor(chosen_attn_mask),
                "rejected_input_ids": torch.tensor(rejected_tokens),
                "rejected_attention_mask": torch.tensor(rejected_attn_mask)
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# ===== TRAIN REWARD MODEL =====
def train_reward_model(llama_model, tokenizer, preference_data, batch_size=4, epochs=3, lr=1e-5, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Melatih reward model berdasarkan data preferensi"""
    print(f"Training reward model on {device}...")
    
    # Inisialisasi reward model
    reward_model = RewardModel(llama_model)
    reward_model.to(device)
    
    # Hanya update parameter reward head
    optimizer = optim.AdamW(reward_model.reward_head.parameters(), lr=lr)
    
    # Dataset dan dataloader
    dataset = PreferenceDataset(preference_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    reward_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Pindahkan batch ke device
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            
            # Forward pass
            chosen_reward = reward_model(chosen_input_ids, chosen_attention_mask)
            rejected_reward = reward_model(rejected_input_ids, rejected_attention_mask)
            
            # Hitung loss: log(sigmoid(chosen_reward - rejected_reward))
            loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Log progress
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return reward_model

# ===== PPO ALGORITHM =====
class PPOTrainer:
    """Implementasi PPO untuk fine-tuning model LLaMA dengan reward"""
    def __init__(self, model, reward_model, tokenizer, 
                 lr=1e-6, 
                 eps_clip=0.2, 
                 value_coef=0.5, 
                 entropy_coef=0.01,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # PPO hyperparameters
        self.lr = lr
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        
        # Tambahkan value head ke model
        self.value_head = nn.Linear(model.dim, 1).to(device)
        self.value_optimizer = optim.AdamW(self.value_head.parameters(), lr=self.lr)
    
    def compute_rewards(self, input_ids, attention_mask):
        """Hitung rewards menggunakan reward model"""
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)
        return rewards
    
    def compute_values(self, hidden_states):
        """Prediksi values untuk state"""
        # Ambil hidden state terakhir
        last_hidden = hidden_states[:, -1, :]
        values = self.value_head(last_hidden)
        return values
    
    def generate_response(self, prompt, max_length=50):
        """Generate respons dari model saat ini"""
        # Tokenize prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Simpan log probabilities untuk setiap token yang digenerate
        log_probs = []
        generated_ids = []
        hidden_states_list = []
        
        # Generate token satu per satu
        self.model.eval()
        for _ in range(max_length):
            with torch.no_grad():
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Simpan hidden states
                hidden_states = self.model.norm(self.model.layers[-1].ffn(self.model.layers[-1].ffn_norm(logits)))
                hidden_states_list.append(hidden_states)
                
                # Ambil logits untuk token terakhir
                next_token_logits = logits[:, -1, :]
                
                # Sample dari distribusi
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Hitung log probability
                log_prob = F.log_softmax(next_token_logits, dim=-1).gather(1, next_token)
                log_probs.append(log_prob)
                
                # Tambahkan ke generated_ids
                generated_ids.append(next_token.item())
                
                # Update input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                
                # Hentikan jika generate EOS token
                if next_token.item() == self.tokenizer.special_tokens["<eos>"]:
                    break
        
        # Gabungkan semua hidden states
        all_hidden_states = torch.cat(hidden_states_list, dim=1)
        
        # Hitung values
        values = self.compute_values(all_hidden_states)
        
        # Hitung rewards
        rewards = self.compute_rewards(input_ids, attention_mask)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "log_probs": torch.cat(log_probs),
            "values": values,
            "rewards": rewards,
            "generated_text": self.tokenizer.decode(generated_ids)
        }
    
    def ppo_update(self, old_log_probs, states, actions, rewards, values, batch_size=4, epochs=4):
        """Update model menggunakan algoritma PPO"""
        self.model.train()
        
        # Hitung advantages dan returns
        returns = rewards
        advantages = rewards - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(epochs):
            # Forward pass dengan model saat ini
            logits = self.model(states)
            new_log_probs = F.log_softmax(logits, dim=-1).gather(2, actions.unsqueeze(2)).squeeze(2)
            
            # Ratio untuk PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            new_values = self.compute_values(states)
            value_loss = F.mse_loss(new_values, returns)
            
            # Entropy bonus
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.value_optimizer.step()
    
    def train(self, prompts, num_iterations=100, batch_size=4):
        """Latih model dengan PPO"""
        for iteration in range(num_iterations):
            # Collect experiences
            experiences = []
            for prompt in random.sample(prompts, min(batch_size, len(prompts))):
                exp = self.generate_response(prompt)
                experiences.append(exp)
            
            # Batch experiences
            old_log_probs = torch.cat([exp["log_probs"] for exp in experiences])
            states = torch.cat([exp["input_ids"] for exp in experiences])
            attention_masks = torch.cat([exp["attention_mask"] for exp in experiences])
            rewards = torch.cat([exp["rewards"] for exp in experiences])
            values = torch.cat([exp["values"] for exp in experiences])
            
            # Update model dengan PPO
            self.ppo_update(old_log_probs, states, attention_masks, rewards, values)
            
            # Log progress
            if iteration % 10 == 0:
                avg_reward = rewards.mean().item()
                print(f"Iteration {iteration} - Average Reward: {avg_reward:.4f}")
                
                # Tampilkan contoh generate
                sample_idx = random.randint(0, len(experiences) - 1)
                print(f"Sample generation:")
                print(f"Prompt: {prompts[sample_idx]}")
                print(f"Response: {experiences[sample_idx]['generated_text']}")
                print(f"Reward: {experiences[sample_idx]['rewards'].item():.4f}")
                print("-" * 50)

# ===== CONTOH DATA PREFERENSI =====
def prepare_preference_data():
    """Buat contoh data preferensi untuk melatih reward model"""
    preference_data = [
        {
            "prompt": "Apa itu AI?",
            "chosen": "AI atau kecerdasan buatan adalah bidang ilmu komputer yang fokus pada pengembangan sistem yang dapat melakukan tugas-tugas yang biasanya memerlukan kecerdasan manusia, seperti pengenalan visual, pengenalan suara, pengambilan keputusan, dan penerjemahan bahasa.",
            "rejected": "AI adalah robot yang bisa berpikir seperti manusia."
        },
        {
            "prompt": "Jelaskan teori relativitas Einstein",
            "chosen": "Teori relativitas Einstein terdiri dari dua teori: relativitas khusus dan relativitas umum. Relativitas khusus menyatakan bahwa hukum fisika sama untuk semua pengamat yang bergerak dengan kecepatan konstan, dan kecepatan cahaya konstan di semua kerangka acuan. Relativitas umum memperluas ini ke pengamat yang dipercepat dan menjelaskan gravitasi sebagai kelengkungan ruang-waktu.",
            "rejected": "Einstein menemukan E=mc²."
        },
        {
            "prompt": "Bagaimana cara kerja vaksin?",
            "chosen": "Vaksin bekerja dengan memperkenalkan versi yang dilemahkan atau bagian dari patogen (seperti virus atau bakteri) ke dalam tubuh. Ini memicu sistem kekebalan untuk mengenali dan menghasilkan antibodi terhadap patogen tersebut tanpa menyebabkan penyakit. Jika tubuh kemudian terpapar patogen yang sebenarnya, sistem kekebalan sudah siap untuk melawannya.",
            "rejected": "Vaksin menyuntikkan virus ke dalam tubuh untuk membuat kita kebal."
        }
    ]
    return preference_data

# ===== MAIN FUNCTION =====
def main():
    # Inisialisasi tokenizer
    tokenizer = SimpleBPETokenizer(vocab_size=10000)
    
    # Contoh teks untuk pretraining
    texts = [
        "Kecerdasan buatan (AI) adalah simulasi kecerdasan manusia dalam mesin yang diprogram untuk berpikir dan belajar seperti manusia.",
        "Deep learning adalah subset dari machine learning yang menggunakan jaringan saraf tiruan dengan banyak lapisan.",
        "Natural Language Processing (NLP) adalah cabang AI yang fokus pada interaksi antara komputer dan bahasa manusia."
    ]
    
    # Latih tokenizer
    tokenizer.train(texts)
    
    # Inisialisasi model LLaMA kecil
    model = LLaMAModel(
        vocab_size=tokenizer.vocab_size_actual,
        dim=256,
        num_layers=4,
        num_heads=4,
        max_seq_len=512
    )
    
    # 1. Pretraining
    print("Step 1: Pretraining")
    model = pretrain_llama(model, tokenizer, texts, epochs=2)
    
    # 2. Supervised Fine-tuning
    print("\nStep 2: Supervised Fine-tuning")
    qa_data = prepare_qa_data()
    model = supervised_finetune(model, tokenizer, qa_data, epochs=2)
    
    # Test generate setelah SFT
    print("\nTesting generation after SFT:")
    test_prompts = ["Apa itu AI?", "Berapa jumlah planet di tata surya?"]
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 50)
    
    # 3. RLHF
    print("\nStep 3: RLHF")
    # 3.1 Train reward model
    preference_data = prepare_preference_data()
    reward_model = train_reward_model(model, tokenizer, preference_data, epochs=2)
    
    # 3.2 PPO fine-tuning
    ppo_trainer = PPOTrainer(model, reward_model, tokenizer)
    prompts = [item["prompt"] for item in preference_data]
    ppo_trainer.train(prompts, num_iterations=50)
    
    # Test generate setelah RLHF
    print("\nTesting generation after RLHF:")
    for prompt in test_prompts:
        generated = generate_text(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 50)
    
    # Simpan model
    torch.save(model.state_dict(), "llama_mini_model.pt")
    print("Model saved to llama_mini_model.pt")

if __name__ == "__main__":
    main()