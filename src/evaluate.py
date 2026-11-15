import argparse
import math

import torch

from torch.nn import functional as F

from data import DATASETS, CharTokenizer, load_text, train_val_split
from train import CONFIGS, SEED, estimate_loss
from utils import get_device, load_checkpoint, resolve_path

SAMPLE_TOKENS = 500
# the diagnostics below use at least this many batches: their buckets share windows, so the differences
# between buckets are precise, but the level of each still moves by ~0.01 between draws of 50 batches
DIAGNOSTIC_MIN_ITERS = 200


def fixed_val_windows(val_data, batch_size, block_size, iters, device):
    """The same val batches for every model with this data, batch size and block size, whatever the architecture.

    Same windows as get_batch would pick, but drawn from their own generator seeded with SEED: building a
    model and the diagnostics themselves use up different amounts of the global randomness per architecture.
    """
    g = torch.Generator().manual_seed(SEED)
    starts = torch.randint(len(val_data) - block_size, (iters, batch_size), generator=g)
    for ix in starts:
        x = torch.stack([val_data[i:i + block_size] for i in ix])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in ix])
        yield x.to(device), y.to(device)


@torch.no_grad()
def encoder_fill_losses(model, windows):
    """Val loss of an encoder-decoder split by how much of its encoder side holds real text.

    Every bucket is scored on the same val windows, only the encoder side changes:
    - empty: all PAD, the decoder works alone. This is how the first dec_size characters of a story are written.
    - partly filled: only the last k encoder characters kept, k uniform in [1, enc_size), as while the story
      grows from dec_size to block_size characters.
    - full: the whole encoder side, the same setting as the normal val loss.
    """
    model.eval()
    S = model.enc_size
    g = torch.Generator().manual_seed(SEED + 1)  # for k, kept apart from the window generator
    sums, n = {"empty": 0.0, "partly filled": 0.0, "full": 0.0}, 0
    for X, Y in windows:
        partial = torch.randint(1, S, (X.shape[0], 1), generator=g).to(X.device)
        for bucket, keep in [("empty", 0), ("partly filled", partial), ("full", S)]:
            _, loss = model(X, Y, src_keep=keep)
            sums[bucket] += loss.item()
        n += 1
    return {bucket: total / n for bucket, total in sums.items()}


@torch.no_grad()
def position_losses(model, windows, split):
    """Val loss of a decoder-only model on positions [0, split) and [split, block_size) of the same windows.

    Position p is predicted from p + 1 characters of context. With split = 256 and block_size = 512 the two
    halves see the same context lengths as a 256 + 256 encoder-decoder's decoder with an empty encoder side
    (1-256) and with a full one (257-512).
    """
    model.eval()
    sums, n = [0.0, 0.0], 0
    for X, Y in windows:
        logits, _ = model(X)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, C), Y.reshape(B * T), reduction="none").view(B, T)
        sums[0] += loss[:, :split].mean().item()
        sums[1] += loss[:, split:].mean().item()
        n += 1
    return {"first half": sums[0] / n, "second half": sums[1] / n}


def evaluate(path):
    torch.manual_seed(SEED)
    device = get_device()
    model, ckpt = load_checkpoint(path, device)
    tok = CharTokenizer(ckpt["chars"])
    cfg = ckpt.get("train_config") or CONFIGS[ckpt["model_name"]]
    print(f"loaded {path} ({ckpt['model_name']}, {ckpt['step']} steps, vocab size {len(tok)})")

    # same text and split as in training so val is still held out
    text = load_text(ckpt["dataset"], ckpt.get("max_chars"))
    train_data, val_data = train_val_split(text, tok)

    losses = estimate_loss(model, train_data, val_data, cfg["batch_size"],
                           ckpt["block_size"], cfg["eval_iters"], device)
    print(f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    print(f"val perplexity {math.exp(losses['val']):.2f} "
          f"(uniform guessing: {len(tok)}, loss {math.log(len(tok)):.4f})")

    # neither diagnostic touches the global randomness, so the samples below come out the same without them
    iters = max(cfg["eval_iters"], DIAGNOSTIC_MIN_ITERS)
    windows = fixed_val_windows(val_data, cfg["batch_size"], ckpt["block_size"], iters, device)

    if ckpt["model_name"] == "gpt":
        half = ckpt["block_size"] // 2
        pos = position_losses(model, windows, half)
        print(f"\nval loss by position (the same {iters} batches as the encdec encoder-fill diagnostic):")
        print(f"  first half   positions 0-{half - 1:<5} context 1-{half:<8} {pos['first half']:.4f}   "
              f"compare with a {half}+{half} encdec, empty bucket")
        print(f"  second half  positions {half}-{2 * half - 1:<3} context {half + 1}-{2 * half:<6} "
              f"{pos['second half']:.4f}   compare with a {half}+{half} encdec, full bucket")

    if ckpt["model_name"] == "encdec":
        fill = encoder_fill_losses(model, windows)
        S, T = model.enc_size, model.dec_size
        rows = [("empty", "0", f"how the first {T} characters of a story are written"),
                ("partly filled", f"1-{S - 1}", ""), ("full", f"{S}", "same setting as the val loss above")]
        print(f"\nval loss by encoder fill ({S} encoder + {T} decoder characters, the same {iters} batches "
              f"for each bucket):")
        for bucket, chars, note in rows:
            print(f"  {bucket:<14} {chars + ' chars':<13} {fill[bucket]:.4f}   {note}")
        print(f"  empty - full gap: {fill['empty'] - fill['full']:+.4f} (what the encoder side is worth)")
        print(f"  overall val loss (full encoder, its own windows): {losses['val']:.4f}")

    prompts = ["\n", DATASETS[ckpt["dataset"]]["prompt"]]
    for prompt in prompts:
        print(f"\n--- sample from {prompt!r} ---")
        context = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        print(tok.decode(model.generate(context, SAMPLE_TOKENS)[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="?", default="weights/gpt-10.8m-c256-d5m-b64.pt")
    args = parser.parse_args()
    evaluate(resolve_path(args.weights))
