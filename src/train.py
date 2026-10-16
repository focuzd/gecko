import argparse

import torch

from data import DATASETS, CharTokenizer, get_batch, load_text, train_val_split
from utils import CHECKPOINT_DIR, MODELS, WEIGHTS_DIR, get_device, save_checkpoint

TRAIN_CONFIG = {
    "bigram": dict(batch_size=32, block_size=8, max_iters=3000, eval_interval=300,
                   learning_rate=1e-2, eval_iters=200),
    "mlp": dict(batch_size=64, block_size=32, max_iters=10000, eval_interval=1000,
                learning_rate=1e-3, eval_iters=200),
    "gpt": dict(batch_size=64, block_size=256, max_iters=5000, eval_interval=500,
                learning_rate=3e-4, eval_iters=200),
}

MODEL_CONFIG = {
    "bigram": dict(),
    "mlp": dict(n_embd=24, n_hidden=512),
    "gpt": dict(n_embd=384, n_head=6, n_layer=6, dropout=0.2),
}

SEED = 1337


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(dataset, model_name):
    torch.manual_seed(SEED)
    device = get_device()
    cfg = TRAIN_CONFIG[model_name]
    block_size = cfg["block_size"]

    text = load_text(dataset)
    tok = CharTokenizer.from_text(text)
    train_data, val_data = train_val_split(text, tok)
    print(f"vocab size {len(tok)}, train tokens {len(train_data):,}, val tokens {len(val_data):,}")

    model_args = dict(vocab_size=len(tok), **MODEL_CONFIG[model_name])
    if model_name != "bigram":
        model_args["block_size"] = block_size
    model = MODELS[model_name](**model_args).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, training on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    name = f"{dataset}_{model_name}.pt"

    for step in range(cfg["max_iters"]):
        if step % cfg["eval_interval"] == 0 or step == cfg["max_iters"] - 1:
            losses = estimate_loss(model, train_data, val_data, cfg["batch_size"],
                                   block_size, cfg["eval_iters"], device)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            save_checkpoint(CHECKPOINT_DIR / name, model, model_name, dataset, tok.chars, block_size, step)

        xb, yb = get_batch(train_data, cfg["batch_size"], block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_checkpoint(WEIGHTS_DIR / name, model, model_name, dataset, tok.chars, block_size, cfg["max_iters"])
    print(f"saved {WEIGHTS_DIR / name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="gpt")
    parser.add_argument("--dataset", choices=DATASETS, default="tinystories")
    args = parser.parse_args()
    train(args.dataset, args.model)
