Untuk menjalankan implementasi ini, ikuti langkah-langkah berikut:

1. Pastikan semua dependensi terinstal:
```
pip install torch numpy matplotlib
```
2. Jalankan script utama untuk melatih model dari awal hingga RLHF:
```
python main.py --mode all 
--model_path llama_mini.pt
```
3. Atau jalankan tahap tertentu saja:
```
# Hanya pretraining
python main.py --mode pretrain 
--model_path llama_mini.pt

# Hanya SFT
python main.py --mode sft 
--model_path llama_mini.pt

# Hanya RLHF
python main.py --mode rlhf 
--model_path llama_mini_sft.pt
```
Implementasi ini mencakup semua komponen utama dari arsitektur LLaMA dengan skala yang lebih kecil, termasuk RMSNorm, Rotary Positional Embeddings, dan SwiGLU activation. Model ini dilatih melalui tiga tahap: pretraining, supervised fine-tuning, dan reinforcement learning from human feedback, sesuai dengan metodologi yang digunakan dalam pengembangan model bahasa besar modern.