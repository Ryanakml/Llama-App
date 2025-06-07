import torch
import os
import argparse

# Import dari file-file sebelumnya
from llama_model import LLaMAModel, SimpleBPETokenizer, pretrain_llama
from supervised_finetuning import supervised_finetune, generate_text, prepare_qa_data
from rlhf import train_reward_model, PPOTrainer, prepare_preference_data

def main():
    parser = argparse.ArgumentParser(description="Train a mini LLaMA model from scratch")
    parser.add_argument("--mode", type=str, default="all", choices=["pretrain", "sft", "rlhf", "all"], 
                        help="Training mode")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data")
    parser.add_argument("--model_path", type=str, default=None, help="Path to save/load model")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to train on")
    
    args = parser.parse_args()
    
    # Inisialisasi tokenizer
    tokenizer = SimpleBPETokenizer(vocab_size=args.vocab_size)
    
    # Contoh teks untuk pretraining jika tidak ada data path
    if args.data_path is None:
        texts = [
            "Kecerdasan buatan (AI) adalah simulasi kecerdasan manusia dalam mesin yang diprogram untuk berpikir dan belajar seperti manusia.",
            "Deep learning adalah subset dari machine learning yang menggunakan jaringan saraf tiruan dengan banyak lapisan.",
            "Natural Language Processing (NLP) adalah cabang AI yang fokus pada interaksi antara komputer dan bahasa manusia.",
            "Reinforcement learning adalah jenis machine learning di mana agen belajar dengan berinteraksi dengan lingkungan dan menerima reward atau punishment.",
            "Computer vision adalah bidang AI yang memungkinkan komputer untuk mengekstrak informasi dari gambar dan video."
        ]
    else:
        # Load data dari file
        with open(args.data_path, "r", encoding="utf-8") as f:
            texts = f.readlines()
    
    # Latih tokenizer
    print("Training tokenizer...")
    tokenizer.train(texts)
    
    # Inisialisasi atau load model
    if args.model_path and os.path.exists(args.model_path) and args.mode != "pretrain":
        print(f"Loading model from {args.model_path}...")
        model = LLaMAModel(
            vocab_size=tokenizer.vocab_size_actual,
            dim=args.dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads
        )
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    else:
        print("Initializing new model...")
        model = LLaMAModel(
            vocab_size=tokenizer.vocab_size_actual,
            dim=args.dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads
        )
    
    # Pretraining
    if args.mode in ["pretrain", "all"]:
        print("\n===== PRETRAINING =====")
        model = pretrain_llama(
            model, tokenizer, texts, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            device=args.device
        )
        
        # Simpan model setelah pretraining
        if args.model_path:
            torch.save(model.state_dict(), args.model_path)
            print(f"Pretrained model saved to {args.model_path}")
    
    # Supervised Fine-tuning
    if args.mode in ["sft", "all"]:
        print("\n===== SUPERVISED FINE-TUNING =====")
        qa_data = prepare_qa_data()
        model = supervised_finetune(
            model, tokenizer, qa_data, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            device=args.device
        )
        
        # Test generate setelah SFT
        print("\nTesting generation after SFT:")
        test_prompts = ["Apa itu AI?", "Berapa jumlah planet di tata surya?"]
        for prompt in test_prompts:
            generated = generate_text(model, tokenizer, prompt, device=args.device)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
            print("-" * 50)
        
        # Simpan model setelah SFT
        if args.model_path:
            sft_path = args.model_path.replace(".pt", "_sft.pt")
            torch.save(model.state_dict(), sft_path)
            print(f"SFT model saved to {sft_path}")
    
    # RLHF
    if args.mode in ["rlhf", "all"]:
        print("\n===== REINFORCEMENT LEARNING FROM HUMAN FEEDBACK =====")
        # Train reward model
        print("Training reward model...")
        preference_data = prepare_preference_data()
        reward_model = train_reward_model(
            model, tokenizer, preference_data, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            device=args.device
        )
        
        # PPO fine-tuning
        print("\nPPO fine-tuning...")
        ppo_trainer = PPOTrainer(model, reward_model, tokenizer, device=args.device)
        prompts = [item["prompt"] for item in preference_data]
        ppo_trainer.train(prompts, num_iterations=50)
        
        # Test generate setelah RLHF
        print("\nTesting generation after RLHF:")
        test_prompts = ["Apa itu AI?", "Berapa jumlah planet di tata surya?"]
        for prompt in test_prompts:
            generated = generate_text(model, tokenizer, prompt, device=args.device)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
            print("-" * 50)
        
        # Simpan model final
        if args.model_path:
            rlhf_path = args.model_path.replace(".pt", "_rlhf.pt")
            torch.save(model.state_dict(), rlhf_path)
            print(f"RLHF model saved to {rlhf_path}")

if __name__ == "__main__":
    main()