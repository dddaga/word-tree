from tokenizers import ByteLevelBPETokenizer, Tokenizer

class SubwordTokenizer:
    def __init__(self, method='BPE', min_freq=5):
        self.method = method
        self.min_freq = min_freq
        self.tokenizer = None

    def train(self, corpus_path, vocab_size=30000):
        if self.method == 'BPE':
            self.tokenizer = ByteLevelBPETokenizer()
            self.tokenizer.train(files=[corpus_path], vocab_size=vocab_size, min_frequency=self.min_freq)
        # Add other methods if needed

    def tokenize(self, text):
        if self.tokenizer is not None:
            return self.tokenizer.encode(text).tokens
        else:
            raise ValueError("Tokenizer not trained yet.")

    def save(self, path):
        if self.tokenizer is not None:
            self.tokenizer.save_model(path)

    def load(self, path):
        if self.method == 'BPE':
            self.tokenizer = ByteLevelBPETokenizer.from_file(f"{path}/vocab.json", f"{path}/merges.txt")
        # Add loading for other methods if needed
