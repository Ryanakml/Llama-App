import numpy as np
import re
import matplotlib.pyplot as plt
import pickle

# preprocessing and cleaning raw text input
class TextProcessor:
    def __init__(self):
        self.sentence_pattern = r'[.!?]+\s+'


    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s,.?!;:-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_sentence(self, text):
        sentences = []
        raw_sentences = re.split(self.sentence_pettern, text)

        for sent in raw_sentences:
            sent = sent.strip()
            if len(sent) > 10:
                sentences.append(sent)
        return sentences

    def create_corpus(self, sentences):
        corpus = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) > 1:
                corpus.append(words)

        corpus = [sent for sent in corpus if 3 <= len(sent) <= 100]
        return corpus


# tokenizer
class BPETokenizer:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.vocab = {}
        self.merges = []

    def _get_pairs(self, vocab):
        pairs = {}
        for word, freqs in vocab.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = ''.join(pair)

        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(bigram)
                    i+=2
                else:
                    new_word.append(word[i])
                    i += 1
                    
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def train(self, corpus):
        print('Train in progress .....')

        vocab_freq = {}
        for sentence in corpus:
            for word in sentence:
                chars = list(word) + ['</w>']
                token = tuple(chars)
                vocab_freq[token] = vocab.get(token, 0) + 1


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
                print(f'merge steps : {merge_step}, vocab_size : {len(current_vocab)}')

        token_set = set()
        for word in curret_vocab.keys():
            for token in word:
                token_set.add(token)

        self.vocab = {token: idx for idx, token in enumerate(sorted(token_set))}

        special_tokens = ['[UNK]', '[PAD]', '[BOS]', '[EOS]', '[MASK]']
        for token in special_tokens:
            self.vocab[token] = len(vocab)

        print(f'... bpe training completed. vocab size : {len(self.vocab)}')
        return self.vocab, self.merges

    def encpod