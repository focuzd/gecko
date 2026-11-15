import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt import FeedForward


# A small learning version of an encoder-decoder (the original 2017 transformer layout), trained on the same
# plain text stream as the other models, so train.py, evaluate.py, sample_stories.py and the benchmark work
# unchanged.
#
# Each training window of block_size characters is split in two, like T5's "prefix LM" objective:
#
#   [ ........ enc_size chars ........ | ........ dec_size chars ........ ]
#     encoder: reads all of it at once   decoder: predicts the next character,
#     (bidirectional, no causal mask)    causal, and cross-attends to the encoder
#
# So the encoder holds the older text and the decoder the most recent text. The loss is only on the decoder
# positions; the encoder never predicts anything, it only builds a representation for the decoder to read.
# While generating, the decoder first works alone (like a small GPT) and the encoder takes over the older
# text once the story is longer than dec_size characters.
#
# Shortcuts compared to a full encoder-decoder (e.g. on TinyStories-Instruct):
# - no separate (input, output) pairs, the split is just a position in the text
# - padding uses a dedicated PAD token that the encoder learns to ignore, instead of attention masks


class Head(nn.Module):
    """One attention head. Queries come from x, keys and values from source:
    x itself for self-attention, the encoder output for cross-attention."""

    def __init__(self, n_embd, head_size, dropout, causal):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.causal = causal
        self.dropout = dropout

    def forward(self, x, source):
        q = self.query(x)
        k = self.key(source)
        v = self.value(source)
        return F.scaled_dot_product_attention(q, k, v, is_causal=self.causal,
                                              dropout_p=self.dropout if self.training else 0.0)


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, n_head, dropout, causal):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout, causal) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, source):
        out = torch.cat([h(x, source) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class EncoderBlock(nn.Module):
    """Same as the GPT block, except attention is not causal: every character sees every other one."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout, causal=False)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.sa(h, h)
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """GPT block plus one extra step in the middle: cross-attention to the encoder output."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout, causal=True)
        self.ca = MultiHeadAttention(n_embd, n_head, dropout, causal=False)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, enc):
        h = self.ln1(x)
        x = x + self.sa(h, h)  # look back at earlier decoder characters
        x = x + self.ca(self.ln2(x), enc)  # look at the whole encoder side
        x = x + self.ffwd(self.ln3(x))
        return x


class EncoderDecoderLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size=512, enc_size=None, n_embd=384, n_head=6,
                 n_enc_layer=3, n_dec_layer=3, dropout=0.2, short_src_frac=2 / 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.enc_size = enc_size or block_size // 2
        self.dec_size = block_size - self.enc_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_enc_layer = n_enc_layer
        self.n_dec_layer = n_dec_layer
        self.dropout = dropout
        # share of training windows whose encoder side gets cut short, see _shorten_src
        self.short_src_frac = short_src_frac
        self.pad = vocab_size  # one extra token id, only ever seen by the encoder

        # one token table for both sides (like T5), with the extra PAD row
        self.token_embedding_table = nn.Embedding(vocab_size + 1, n_embd)
        self.enc_position_embedding_table = nn.Embedding(self.enc_size, n_embd)
        self.dec_position_embedding_table = nn.Embedding(self.dec_size, n_embd)
        self.enc_blocks = nn.ModuleList([EncoderBlock(n_embd, n_head, dropout) for _ in range(n_enc_layer)])
        self.enc_ln_f = nn.LayerNorm(n_embd)
        self.dec_blocks = nn.ModuleList([DecoderBlock(n_embd, n_head, dropout) for _ in range(n_dec_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def config(self):
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "enc_size": self.enc_size,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_enc_layer": self.n_enc_layer,
            "n_dec_layer": self.n_dec_layer,
            "dropout": self.dropout,
            "short_src_frac": self.short_src_frac,
        }

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def split(self, idx):
        """Split a context into encoder and decoder parts.

        The decoder always gets the most recent dec_size characters, the encoder the enc_size before them.
        Training windows are exactly block_size long, so the split is always at enc_size. A shorter context
        (the start of a story while generating) leaves the encoder side empty or partly empty: it is
        left-padded with PAD, so its last slot is always the character right before the decoder.

        (Filling the encoder first instead, so the prompt goes to the encoder, was tried and gave garbled
        story openings: the decoder then starts with a single character, a case that gets very little
        training since only decoder position 0 of each window ever sees it.)
        """
        B, T = idx.shape
        n_dec = min(T, self.dec_size)
        src, tgt = idx[:, :T - n_dec], idx[:, T - n_dec:]
        src = F.pad(src, (self.enc_size - src.shape[1], 0), value=self.pad)
        return src, tgt

    def _shorten_src(self, src):
        # while generating, the encoder side starts empty and fills up only after dec_size characters,
        # but full training windows never have padding. so in short_src_frac of the windows keep only the
        # last k encoder characters and pad the rest. k = max(0, uniform in [-S, S)), so about half of those
        # windows get an empty encoder side (the start of every story) and half a partly filled one
        B, S = src.shape
        keep = torch.randint(-S, S, (B, 1), device=src.device).clamp(min=0)
        short = torch.rand(B, 1, device=src.device) < self.short_src_frac
        return self.pad_src(src, torch.where(short, keep, S))

    def pad_src(self, src, keep):
        """Keep only the last `keep` characters of each encoder side (an int or a (B, 1) tensor), PAD the rest."""
        positions = torch.arange(src.shape[1], device=src.device)
        return src.masked_fill(positions < src.shape[1] - torch.as_tensor(keep, device=src.device), self.pad)

    def encode(self, src):
        x = self.token_embedding_table(src)
        x = x + self.enc_position_embedding_table(torch.arange(src.shape[1], device=src.device))
        for block in self.enc_blocks:
            x = block(x)
        return self.enc_ln_f(x)

    def decode(self, tgt, enc):
        x = self.token_embedding_table(tgt)
        x = x + self.dec_position_embedding_table(torch.arange(tgt.shape[1], device=tgt.device))
        for block in self.dec_blocks:
            x = block(x, enc)
        return self.lm_head(self.ln_f(x))

    def forward(self, idx, targets=None, src_keep=None):
        # logits are only for the decoder positions, i.e. the last tgt.shape[1] characters of idx,
        # so logits[:, -1] is still the prediction for the next character.
        # src_keep (diagnostics only): how many encoder characters to keep, the rest become PAD
        src, tgt = self.split(idx)
        if src_keep is not None:
            src = self.pad_src(src, src_keep)
        elif self.training and targets is not None:
            src = self._shorten_src(src)
        logits = self.decode(tgt, self.encode(src))  # (B, dec positions, vocab_size)

        if targets is None:
            return logits, None

        targets = targets[:, -tgt.shape[1]:]
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def param_breakdown(self):
        """Parameter counts per part. The token table is shared by both sides, so it is counted on its own."""
        count = lambda *modules: sum(p.numel() for m in modules for p in m.parameters())
        return {
            "shared token embedding": count(self.token_embedding_table),
            "encoder": count(self.enc_position_embedding_table, self.enc_blocks, self.enc_ln_f),
            "decoder": count(self.dec_position_embedding_table, self.dec_blocks, self.ln_f, self.lm_head),
            "of which cross-attention": count(*[b.ca for b in self.dec_blocks], *[b.ln2 for b in self.dec_blocks]),
            "total": count(self),
        }


if __name__ == "__main__":
    # python src/encdec.py: parameter table of the encoder-decoder configs vs the GPT baseline
    from gpt import GPTLanguageModel
    from train import CONFIGS

    VOCAB = 96  # the 50M character TinyStories train split
    gpt = sum(p.numel() for p in GPTLanguageModel(VOCAB, block_size=512).parameters())
    print(f"GPT baseline gpt-10.9m-c512-d50m-b64: {gpt:,} parameters (vocab {VOCAB})\n")
    header = ["config", "layers", "width", "heads", "head dim", "encoder", "decoder (incl. cross-attn)",
              "cross-attn", "shared tok emb", "total", "vs GPT"]
    rows = []
    for name, cfg in CONFIGS.items():
        if cfg["model"] != "encdec" or name.endswith("tiny"):
            continue
        a = cfg["model_args"]
        m = EncoderDecoderLanguageModel(VOCAB, block_size=cfg["block_size"], **a)
        p = m.param_breakdown()
        rows.append([name, f"{a['n_enc_layer']}e{a['n_dec_layer']}d", a["n_embd"], a["n_head"],
                     a["n_embd"] // a["n_head"], f"{p['encoder']:,}", f"{p['decoder']:,}",
                     f"{p['of which cross-attention']:,}", f"{p['shared token embedding']:,}",
                     f"{p['total']:,}", f"{100 * (p['total'] - gpt) / gpt:+.1f}%"])
    print("| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
