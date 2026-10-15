import urllib.request

import torch

from utils import DATA_DIR

DATASETS = {
    # this is the TinyStories validation split (~20MB), we only use the first 5M chars
    "tinystories": {
        "path": DATA_DIR / "tinystories.txt",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
        "max_chars": 5_000_000,
        "prompt": "Once upon a time",
    },
    "shakespeare": {
        "path": DATA_DIR / "shakespeare.txt",
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "max_chars": None,
        "prompt": "ROMEO:",
    },
}


def download(url, path, max_bytes=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    n = 0
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            n += len(chunk)
            if max_bytes and n >= max_bytes:
                break


def load_text(name):
    cfg = DATASETS[name]
    max_chars = cfg["max_chars"]
    if not cfg["path"].exists():
        # some characters are multi-byte, so grab a bit extra
        download(cfg["url"], cfg["path"], max_bytes=max_chars and int(max_chars * 1.2))

    text = cfg["path"].read_text(encoding="utf-8", errors="ignore")
    if max_chars:
        text = text[:max_chars]
    print(f"{name}: {len(text):,} characters")
    return text


class CharTokenizer:

    def __init__(self, chars):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    def __len__(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]  # unknown chars are dropped

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def train_val_split(text, tokenizer):
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
