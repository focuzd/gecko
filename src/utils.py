from pathlib import Path

import torch

from bigram import BigramLanguageModel
from gpt import GPTLanguageModel
from mlp import MLPLanguageModel

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
CHECKPOINT_DIR = WEIGHTS_DIR / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

MODELS = {
    "bigram": BigramLanguageModel,
    "mlp": MLPLanguageModel,
    "gpt": GPTLanguageModel,
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_path(path):
    # lets paths like weights/x.pt work no matter where the script is run from
    path = Path(path)
    if not path.exists() and not path.is_absolute() and (ROOT / path).exists():
        return ROOT / path
    return path


def size_label(max_chars):
    return "full" if max_chars is None else f"{max_chars / 1e6:g}m"


def param_label(n):
    return f"{n / 1e3:.0f}k" if n < 1e6 else f"{n / 1e6:.1f}m"


def run_name(model_name, n_params, block_size, max_chars):
    # e.g. gpt-10.8m-c256-d50m: model, parameters, context length, training characters
    context = 1 if model_name == "bigram" else block_size  # a bigram only ever looks at one character
    return f"{model_name}-{param_label(n_params)}-c{context}-d{size_label(max_chars)}"


def save_checkpoint(path, model, info, optimizer=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {**info, "model_config": model.config(), "state_dict": model.state_dict()}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = MODELS[ckpt["model_name"]](**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
