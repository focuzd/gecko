import argparse

import torch

from data import DATASETS, CharTokenizer, get_batch, load_text, train_val_split
from utils import CHECKPOINT_DIR, MODELS, WEIGHTS_DIR, get_device, run_name, save_checkpoint

CONFIGS = {
    "bigram": dict(model="bigram", model_args=dict(),
                   batch_size=32, block_size=8, max_iters=3000, eval_interval=300,
                   learning_rate=1e-2, eval_iters=200),
    "mlp": dict(model="mlp", model_args=dict(n_embd=24, n_hidden=512),
                batch_size=64, block_size=32, max_iters=10000, eval_interval=1000,
                learning_rate=1e-3, eval_iters=200),
    "gpt": dict(model="gpt", model_args=dict(n_embd=384, n_head=6, n_layer=6, dropout=0.2),
                batch_size=64, block_size=256, max_iters=5000, eval_interval=500,
                learning_rate=3e-4, eval_iters=200),
    # same recipe as gpt, only wider (~19M parameters)
    "gpt-20m": dict(model="gpt", model_args=dict(n_embd=512, n_head=8, n_layer=6, dropout=0.2),
                    batch_size=64, block_size=256, max_iters=5000, eval_interval=500,
                    learning_rate=3e-4, eval_iters=200),
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


def make_config(config_name, block_size=None, batch_size=None, max_iters=None):
    cfg = dict(CONFIGS[config_name])
    if block_size and not batch_size:
        # keep the number of characters per step the same, so runs with different
        # context lengths see the same amount of text
        batch_size = max(1, cfg["batch_size"] * cfg["block_size"] // block_size)
    cfg["block_size"] = block_size or cfg["block_size"]
    cfg["batch_size"] = batch_size or cfg["batch_size"]
    cfg["max_iters"] = max_iters or cfg["max_iters"]
    return cfg


def train(dataset, config_name, max_chars=None, block_size=None, batch_size=None, max_iters=None,
          resume=False, amp=False):
    torch.manual_seed(SEED)
    device = get_device()
    # fp16 mixed precision is opt-in so it can be compared against fp32 as its own experiment
    use_amp = amp and device.type == "cuda"
    if amp and not use_amp:
        print("--amp needs a CUDA GPU, training in fp32")
    cfg = make_config(config_name, block_size, batch_size, max_iters)
    block_size, batch_size = cfg["block_size"], cfg["batch_size"]

    max_chars = max_chars or DATASETS[dataset]["max_chars"]
    text = load_text(dataset, max_chars)
    tok = CharTokenizer.from_text(text)
    train_data, val_data = train_val_split(text, tok)
    print(f"vocab size {len(tok)}, train tokens {len(train_data):,}, val tokens {len(val_data):,}")

    model_args = dict(vocab_size=len(tok), **cfg["model_args"])
    if cfg["model"] != "bigram":
        model_args["block_size"] = block_size
    model = MODELS[cfg["model"]](**model_args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    name = run_name(cfg["model"], n_params, block_size, max_chars, batch_size)
    if use_amp:
        name += "-fp16"
    print(f"{name}: {n_params / 1e6:.2f}M parameters, batch {batch_size} x {block_size}, "
          f"{cfg['max_iters']} steps, {'fp16 mixed precision' if use_amp else 'fp32'}, training on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, history = 0, []
    if resume and (CHECKPOINT_DIR / f"{name}.pt").exists():
        ckpt = torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start, history = ckpt["step"], ckpt["history"]
        print(f"resuming from step {start}")

    def info(step):
        return dict(model_name=cfg["model"], config=config_name, dataset=dataset, max_chars=max_chars,
                    chars=tok.chars, block_size=block_size, step=step, n_params=n_params,
                    train_config=cfg, amp=use_amp, history=history)

    for step in range(start, cfg["max_iters"]):
        if step % cfg["eval_interval"] == 0 or step == cfg["max_iters"] - 1:
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size,
                                   cfg["eval_iters"], device)
            history.append(dict(step=step, **losses))
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            save_checkpoint(CHECKPOINT_DIR / f"{name}.pt", model, info(step), optimizer)

        xb, yb = get_batch(train_data, batch_size, block_size, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    save_checkpoint(WEIGHTS_DIR / f"{name}.pt", model, info(cfg["max_iters"]))
    print(f"saved {WEIGHTS_DIR / name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=CONFIGS, default="gpt")
    parser.add_argument("--dataset", choices=DATASETS, default="tinystories")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="how many characters of the dataset to use (default: the dataset's max_chars)")
    parser.add_argument("--block-size", type=int, default=None,
                        help="context length; the batch size shrinks to keep characters per step the same")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue from weights/checkpoints/<run name>.pt if it exists")
    parser.add_argument("--amp", action="store_true",
                        help="fp16 mixed precision on a GPU (faster); off by default, runs are fp32")
    args = parser.parse_args()
    train(args.dataset, args.model, args.max_chars, args.block_size, args.batch_size, args.max_iters,
          args.resume, args.amp)
