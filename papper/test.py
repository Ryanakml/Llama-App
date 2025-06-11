import numpy as np
import re
import matplotlib.pyplot as plt
import pickle


class TextProcessor:
    """Handles text cleaning and preprocessing"""
    
    def __init__(self):
        self.sentence_pattern = r'[.!?]+\s+'
    
    def clean_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r"[^\w\s.,!?;:-]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def extract_sentences(self, text):
        """Extract sentences from text"""
        sentences = []
        raw_sentences = re.split(self.sentence_pattern, text)
        
        for sent in raw_sentences:
            sent = sent.strip()
            if len(sent) > 10:
                sentences.append(sent)
        
        return sentences
    
    def create_corpus(self, sentences):
        """Convert sentences to word corpus"""
        corpus = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 1:
                corpus.append(words)
        
        # Filter by length
        corpus = [sent for sent in corpus if 3 <= len(sent) <= 100]
        return corpus


class BPETokenizer:
    """Byte Pair Encoding tokenizer"""
    
    def __init__(self, num_merges=2000):
        self.num_merges = num_merges
        self.vocab = {}
        self.merges = []
    
    def _get_pairs(self, vocab):
        """Get all adjacent pairs and their frequencies"""
        pairs = {}
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs
    
    def _merge_vocab(self, pair, vocab):
        """Merge the most frequent pair in vocabulary"""
        new_vocab = {}
        bigram = ''.join(pair)
        
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        
        return new_vocab
    
    def train(self, corpus):
        """Train BPE on corpus"""
        print("Training BPE tokenizer...")
        
        # Initialize vocabulary with character-level tokens
        vocab_freq = {}
        for sentence in corpus:
            for word in sentence:
                chars = list(word) + ["</w>"]
                token = tuple(chars)
                vocab_freq[token] = vocab_freq.get(token, 0) + 1
        
        # Perform merges
        current_vocab = vocab_freq.copy()
        self.merges = []
        
        for merge_step in range(self.num_merges):
            pairs = self._get_pairs(current_vocab)
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break
                
            current_vocab = self._merge_vocab(best_pair, current_vocab)
            self.merges.append(best_pair)
            
            if merge_step % 200 == 0:
                print(f"Merge step {merge_step}, vocab size: {len(current_vocab)}")
        
        # Create final vocabulary
        token_set = set()
        for word in current_vocab.keys():
            for token in word:
                token_set.add(token)
        
        self.vocab = {token: idx for idx, token in enumerate(sorted(token_set))}
        
        # Add special tokens
        special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[MASK]"]
        for token in special_tokens:
            self.vocab[token] = len(self.vocab)
        
        print(f"BPE training completed. Vocabulary size: {len(self.vocab)}")
        return self.vocab, self.merges
    
    def encode_word(self, word):
        """Encode a single word using trained BPE"""
        tokens = list(word) + ["</w>"]
        
        # Apply learned merges
        for pair in self.merges:
            bigram = ''.join(pair)
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(bigram)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                for char in token:
                    token_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
        
        return token_ids
    
    def encode_corpus(self, corpus):
        """Encode entire corpus with BOS/EOS tokens"""
        encoded = []
        for sentence in corpus:
            encoded_sentence = [self.vocab["[BOS]"]]
            for word in sentence:
                encoded_sentence.extend(self.encode_word(word))
            encoded_sentence.append(self.vocab["[EOS]"])
            encoded.append(encoded_sentence)
        return encoded


class DataProcessor:
    """Handles data preparation for language modeling"""
    
    def __init__(self, max_len=128, min_len=10):
        self.max_len = max_len
        self.min_len = min_len
    
    def create_sequences(self, encoded_corpus):
        """Create input-target pairs for language modeling"""
        sequences = []
        targets = []
        
        for sentence in encoded_corpus:
            if len(sentence) < self.min_len:
                continue
                
            # Split long sentences into chunks
            if len(sentence) > self.max_len:
                for i in range(0, len(sentence) - self.max_len + 1, self.max_len // 2):
                    chunk = sentence[i:i + self.max_len]
                    if len(chunk) >= self.min_len:
                        sequences.append(chunk[:-1])  # Input
                        targets.append(chunk[1:])     # Target (shifted)
            else:
                sequences.append(sentence[:-1])  # Input
                targets.append(sentence[1:])     # Target
        
        return sequences, targets
    
    def pad_sequences(self, sequences, pad_value):
        """Pad sequences to uniform length"""
        padded = []
        attention_masks = []
        
        for seq in sequences:
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            
            # Create attention mask
            mask = [1] * len(seq) + [0] * (self.max_len - len(seq))
            padded_seq = seq + [pad_value] * (self.max_len - len(seq))
            
            padded.append(padded_seq)
            attention_masks.append(mask)
        
        return np.array(padded), np.array(attention_masks)


class SimpleLlamaModel:
    """Simple LLaMA-style transformer model"""
    
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=12, max_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.head_dim = d_model // n_heads
        self.ff_dim = d_model * 4
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        scale = 0.02
        
        # Embedding layer
        self.embedding_matrix = np.random.randn(self.vocab_size, self.d_model) * scale
        
        # Transformer layers
        self.attention_weights = []
        self.ff_weights = []
        
        for _ in range(self.n_layers):
            # Attention weights
            wq = np.random.randn(self.d_model, self.d_model) * scale
            wk = np.random.randn(self.d_model, self.d_model) * scale
            wv = np.random.randn(self.d_model, self.d_model) * scale
            wo = np.random.randn(self.d_model, self.d_model) * scale
            self.attention_weights.append((wq, wk, wv, wo))
            
            # Feedforward weights
            w1 = np.random.randn(self.d_model, self.ff_dim) * scale
            w2 = np.random.randn(self.d_model, self.ff_dim) * scale
            w3 = np.random.randn(self.ff_dim, self.d_model) * scale
            self.ff_weights.append((w1, w2, w3))
        
        # Language modeling head
        self.lm_head_weights = np.random.randn(self.d_model, self.vocab_size) * scale
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _swish(self, x):
        return x * self._sigmoid(x)
    
    def _swiglu(self, x1, x2):
        return x2 * self._swish(x1)
    
    def _softmax(self, x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _rms_norm(self, x, gamma=1.0, eps=1e-8):
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        return gamma * x / rms
    
    def _rope(self, x, positions):
        """Apply Rotary Position Embedding"""
        batch_size, seq_len, d = x.shape
        result = np.zeros_like(x)
        positions = positions.reshape(seq_len)

        for i in range(0, d, 2):
            if i + 1 < d:
                theta = positions[:, np.newaxis] / (10000 ** (i / d))
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                cos_theta = np.broadcast_to(cos_theta.T, (batch_size, seq_len))
                sin_theta = np.broadcast_to(sin_theta.T, (batch_size, seq_len))

                x0 = x[:, :, i]
                x1 = x[:, :, i + 1]

                result[:, :, i] = x0 * cos_theta - x1 * sin_theta
                result[:, :, i + 1] = x0 * sin_theta + x1 * cos_theta
            else:
                result[:, :, i] = x[:, :, i]

        return result
    
    def _multi_head_attention(self, x, wq, wk, wv, wo, attention_mask=None):
        """Multi-head attention with causal masking"""
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = x @ wq.T
        K = x @ wk.T
        V = x @ wv.T
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        positions = np.arange(seq_len)
        for head in range(self.n_heads):
            Q[:, head, :, :] = self._rope(Q[:, head, :, :], positions)
            K[:, head, :, :] = self._rope(K[:, head, :, :], positions)
        
        # Attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        # Causal mask
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + causal_mask[np.newaxis, np.newaxis, :, :]
        
        # Apply attention mask
        if attention_mask is not None:
            mask = (1 - attention_mask[:, np.newaxis, np.newaxis, :]) * -1e9
            scores = scores + mask
        
        # Softmax and apply to values
        attn_weights = self._softmax(scores, axis=-1)
        context = np.matmul(attn_weights, V)
        
        # Concatenate heads and output projection
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        output = context @ wo.T
        
        return output, attn_weights
    
    def _feedforward(self, x, w1, w2, w3):
        """Feedforward with SwiGLU activation"""
        gate = x @ w1
        input_proj = x @ w2
        hidden = self._swiglu(gate, input_proj)
        output = hidden @ w3
        return output
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model"""
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        x = self.embedding_matrix[input_ids]
        
        # Transformer layers
        for layer in range(self.n_layers):
            wq, wk, wv, wo = self.attention_weights[layer]
            w1, w2, w3 = self.ff_weights[layer]
            
            # Pre-attention norm
            x_norm = self._rms_norm(x)
            
            # Self-attention
            attn_out, _ = self._multi_head_attention(x_norm, wq, wk, wv, wo, attention_mask)
            x = x + attn_out
            
            # Pre-feedforward norm
            x_norm = self._rms_norm(x)
            
            # Feedforward
            ff_out = self._feedforward(x_norm, w1, w2, w3)
            x = x + ff_out
        
        # Final norm and language modeling head
        x = self._rms_norm(x)
        logits = x @ self.lm_head_weights
        
        return logits
    
    def compute_loss(self, logits, targets, target_mask):
        """Compute cross-entropy loss for language modeling"""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        mask_flat = target_mask.reshape(-1)
        
        # Only compute loss for non-padded tokens
        valid_indices = mask_flat == 1
        
        if np.sum(valid_indices) == 0:
            return 0.0
        
        valid_logits = logits_flat[valid_indices]
        valid_targets = targets_flat[valid_indices]
        
        # Softmax and cross entropy
        probs = self._softmax(valid_logits, axis=-1)
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        loss = -np.mean(np.log(probs[np.arange(len(valid_targets)), valid_targets]))
        
        return loss
    
    def generate_text(self, tokenizer, prompt, max_length=50, temperature=1.0):
        """Generate text using the trained model"""
        # Encode prompt
        words = prompt.lower().split()
        prompt_tokens = [tokenizer.vocab["[BOS]"]]
        for word in words:
            prompt_tokens.extend(tokenizer.encode_word(word))
        
        generated_tokens = prompt_tokens.copy()
        
        for _ in range(max_length):
            # Prepare input
            input_tokens = generated_tokens[-self.max_len:]
            input_array = np.array([input_tokens + [tokenizer.vocab["[PAD]"]] * (self.max_len - len(input_tokens))])
            attention_mask = np.array([([1] * len(input_tokens) + [0] * (self.max_len - len(input_tokens)))])
            
            # Forward pass
            logits = self.forward(input_array, attention_mask)
            
            # Sample next token
            next_token_logits = logits[0, len(input_tokens) - 1, :] / temperature
            probs = self._softmax(next_token_logits)
            next_token = np.random.choice(self.vocab_size, p=probs)
            
            # Stop if EOS token
            if next_token == tokenizer.vocab["[EOS]"]:
                break
                
            generated_tokens.append(next_token)
        
        # Decode tokens back to text
        reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        generated_text = []
        
        for token_id in generated_tokens[len(prompt_tokens):]:
            if token_id in reverse_vocab:
                token = reverse_vocab[token_id]
                if token not in ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]:
                    generated_text.append(token.replace("</w>", " "))
        
        return "".join(generated_text).strip()


class Trainer:
    """Handles model training"""
    
    def __init__(self, model, learning_rate=0.0001, batch_size=8):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_losses = []
        self.val_losses = []
    
    def _create_batches(self, inputs, targets, masks):
        """Create training batches"""
        n_samples = inputs.shape[0]
        indices = np.random.permutation(n_samples)
        
        batches = []
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_inputs = inputs[batch_indices]
            batch_targets = targets[batch_indices]
            batch_masks = masks[batch_indices]
            batches.append((batch_inputs, batch_targets, batch_masks))
        
        return batches
    
    def train(self, train_inputs, train_targets, train_masks, 
              val_inputs, val_targets, val_masks, num_epochs=5):
        """Train the model"""
        print("Starting training...")
        
        for epoch in range(num_epochs):
            # Training
            train_batches = self._create_batches(train_inputs, train_targets, train_masks)
            epoch_train_loss = 0
            
            for batch_idx, (batch_inputs, batch_targets, batch_masks) in enumerate(train_batches):
                # Forward pass
                logits = self.model.forward(batch_inputs, batch_masks)
                loss = self.model.compute_loss(logits, batch_targets, batch_masks)
                epoch_train_loss += loss
                
                if batch_idx % 20 == 0:
                    perplexity = np.exp(loss)
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}, Perplexity: {perplexity:.2f}")
            
            # Average training loss
            avg_train_loss = epoch_train_loss / len(train_batches)
            
            # Validation
            val_logits = self.model.forward(val_inputs, val_masks)
            val_loss = self.model.compute_loss(val_logits, val_targets, val_masks)
            
            # Store metrics
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            
            train_ppl = np.exp(avg_train_loss)
            val_ppl = np.exp(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Perplexity: {val_ppl:.2f}")
            print("-" * 60)
        
        print("Training completed!")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        train_ppl = [np.exp(loss) for loss in self.train_losses]
        val_ppl = [np.exp(loss) for loss in self.val_losses]
        plt.plot(train_ppl, label='Train Perplexity', marker='o')
        plt.plot(val_ppl, label='Val Perplexity', marker='s')
        plt.title('Training and Validation Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main training pipeline"""
    # Load and process text
    with open('input.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    print(f"Raw text loaded, total characters: {len(raw_text)}")
    
    # Process text
    processor = TextProcessor()
    cleaned_text = processor.clean_text(raw_text)[:5000]  # Limit for demo
    sentences = processor.extract_sentences(cleaned_text)
    corpus = processor.create_corpus(sentences)
    
    print(f"Corpus created with {len(corpus)} sentences")
    
    # Train tokenizer
    tokenizer = BPETokenizer(num_merges=2000)
    vocab, merges = tokenizer.train(corpus)
    
    # Encode corpus
    encoded_corpus = tokenizer.encode_corpus(corpus)
    print(f"Corpus encoded with {len(encoded_corpus)} sequences")
    
    # Prepare data
    data_processor = DataProcessor(max_len=128, min_len=10)
    input_sequences, target_sequences = data_processor.create_sequences(encoded_corpus)
    
    padded_inputs, input_masks = data_processor.pad_sequences(input_sequences, tokenizer.vocab["[PAD]"])
    padded_targets, target_masks = data_processor.pad_sequences(target_sequences, tokenizer.vocab["[PAD]"])
    
    # Split data
    split_idx = int(0.9 * len(padded_inputs))
    train_inputs = padded_inputs[:split_idx]
    train_targets = padded_targets[:split_idx]
    train_masks = input_masks[:split_idx]
    val_inputs = padded_inputs[split_idx:]
    val_targets = padded_targets[split_idx:]
    val_masks = input_masks[split_idx:]
    
    print(f"Data split - Train: {len(train_inputs)}, Val: {len(val_inputs)}")
    
    # Create and train model
    model = SimpleLlamaModel(
        vocab_size=len(vocab),
        d_model=512,
        n_heads=8,
        n_layers=12,
        max_len=128
    )
    
    trainer = Trainer(model, learning_rate=0.0001, batch_size=8)
    trainer.train(train_inputs, train_targets, train_masks,
                  val_inputs, val_targets, val_masks, num_epochs=5)
    
    # Plot results
    trainer.plot_training_curves()
    
    # Test text generation
    test_prompts = ["the weather is", "i think that", "machine learning"]
    
    print("\nTesting text generation:")
    for prompt in test_prompts:
        generated = model.generate_text(tokenizer, prompt, max_length=20, temperature=0.8)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print("-" * 40)
    
    # Save model
    model_data = {
        'vocab': vocab,
        'merges': merges,
        'model_state': {
            'embedding_matrix': model.embedding_matrix,
            'attention_weights': model.attention_weights,
            'ff_weights': model.ff_weights,
            'lm_head_weights': model.lm_head_weights
        },
        'config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'n_heads': model.n_heads,
            'n_layers': model.n_layers,
            'max_len': model.max_len
        },
        'training_history': {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses
        }
    }
    
    with open("simple_llama_oop_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully!")


if __name__ == "__main__":
    main()