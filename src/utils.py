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


def save_checkpoint(path, model, model_name, dataset, chars, block_size, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": model_name,
        "dataset": dataset,
        "model_config": model.config(),
        "state_dict": model.state_dict(),
        "chars": chars,
        "block_size": block_size,
        "step": step,
    }, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = MODELS[ckpt["model_name"]](**ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
