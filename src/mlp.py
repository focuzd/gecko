import torch
import torch.nn as nn
from torch.nn import functional as F


class MLPLanguageModel(nn.Module):
    """Bengio et al. 2003 style: embed the last block_size characters, concatenate, MLP."""

    def __init__(self, vocab_size, block_size=32, n_embd=24, n_hidden=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_hidden = n_hidden

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.net = nn.Sequential(
            nn.Linear(block_size * n_embd, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, vocab_size),
        )

    def config(self):
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_embd": self.n_embd,
            "n_hidden": self.n_hidden,
        }

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # predict every position like the other models: position t sees the block_size
        # characters ending at t, left-padded with token 0 ('\n') at the start
        padded = F.pad(idx, (self.block_size - 1, 0), value=0)
        windows = padded.unfold(1, self.block_size, 1)  # (B, T, block_size)
        emb = self.token_embedding_table(windows)  # (B, T, block_size, n_embd)
        logits = self.net(emb.view(B, T, -1))  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
