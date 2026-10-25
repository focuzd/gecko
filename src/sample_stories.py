import argparse
from datetime import datetime

import torch
from torch.nn import functional as F

from data import CharTokenizer
from utils import OUTPUT_DIR, get_device, load_checkpoint, resolve_path

END_MARKER = "<|endoftext|>"


@torch.no_grad()
def generate_story(model, tok, block_size, prompt, max_new_tokens, temperature=1.0, top_k=None):
    device = next(model.parameters()).device
    ids = tok.encode(prompt) or [0]
    end_ids = tok.encode(END_MARKER)

    for _ in range(max_new_tokens):
        ctx = torch.tensor([ids[-block_size:]], dtype=torch.long, device=device)
        logits, _ = model(ctx)
        logits = logits[0, -1].float().cpu() / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        ids.append(torch.multinomial(probs, num_samples=1).item())

        if ids[-len(end_ids):] == end_ids:
            return tok.decode(ids[:-len(end_ids)]), True

    return tok.decode(ids), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/gpt-10.8m-c256-d5m-b64.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--num-stories", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="output file (default: outputs/<model>_<timestamp>.txt)")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    weights = resolve_path(args.weights)
    model, ckpt = load_checkpoint(weights, get_device())
    tok = CharTokenizer(ckpt["chars"])

    if args.out:
        out_path = resolve_path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"{weights.stem}_{stamp}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (f"weights: {weights.name} ({ckpt['model_name']}, {ckpt['step']} steps)\n"
              f"prompt: {args.prompt!r}, temperature: {args.temperature}, top_k: {args.top_k}, "
              f"seed: {args.seed}\n")
    print(header)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i in range(1, args.num_stories + 1):
            story, ended = generate_story(model, tok, ckpt["block_size"], args.prompt,
                                          args.max_new_tokens, args.temperature, args.top_k)
            note = "" if ended else f"\n[hit the {args.max_new_tokens} token limit]"
            block = f"===== story {i} =====\n{story}{note}\n\n"
            print(block, end="")
            f.write(block)

    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
