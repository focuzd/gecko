import argparse
import math

import torch

from data import DATASETS, CharTokenizer, load_text, train_val_split
from train import CONFIGS, SEED, estimate_loss
from utils import get_device, load_checkpoint, resolve_path

SAMPLE_TOKENS = 500


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

    prompts = ["\n", DATASETS[ckpt["dataset"]]["prompt"]]
    for prompt in prompts:
        print(f"\n--- sample from {prompt!r} ---")
        context = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        print(tok.decode(model.generate(context, SAMPLE_TOKENS)[0].tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="?", default="weights/gpt-10.8m-c256-d5m.pt")
    args = parser.parse_args()
    evaluate(resolve_path(args.weights))
