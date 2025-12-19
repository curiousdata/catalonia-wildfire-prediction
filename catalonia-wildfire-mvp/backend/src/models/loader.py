from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

# NOTE:
# - This module MUST NOT import FastAPI.
# - This module MUST NOT hardcode local laptop paths.
# - Keep this boring: load model artifacts, return a ready-to-run torch.nn.Module.


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _pick_weights_file(model_dir: Path) -> Path:
    """Pick a weights file from a directory.

    Order:
    1) MODEL_FILE env var if provided
    2) common filenames
    3) first matching suffix in sorted order
    """
    model_file = os.getenv("MODEL_FILE")
    if model_file:
        p = Path(model_file)
        if not p.is_absolute():
            p = model_dir / p
        if p.exists():
            return p
        raise FileNotFoundError(f"MODEL_FILE was set but not found: {p}")

    candidates = [
        "model.torchscript",
        "model.ts",
        "model.pt",
        "model.pth",
        "weights.pt",
        "weights.pth",
    ]
    for name in candidates:
        p = model_dir / name
        if p.exists():
            return p

    # Fallback: first .torchscript/.ts/.pt/.pth
    matches = sorted(
        [p for p in model_dir.iterdir() if p.is_file() and p.suffix.lower() in {".torchscript", ".ts", ".pt", ".pth"}],
        key=lambda x: x.name,
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No model file found in {model_dir}. "
        "Place a .pt/.pth/.ts/.torchscript file in ./models and mount it to /app/models."
    )


def _import_callable(dotted_path: str) -> Callable[..., Any]:
    """Import a callable from a dotted path like 'pkg.module:func' or 'pkg.module.func'."""
    import importlib

    if ":" in dotted_path:
        mod_name, attr = dotted_path.split(":", 1)
    else:
        mod_name, attr = dotted_path.rsplit(".", 1)

    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"Imported object is not callable: {dotted_path}")
    return fn


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Load and return the model as a singleton.

    Expected environment variables (optional):
    - MODEL_PATH: directory inside container (default: /app/models)
    - MODEL_FILE: file name or path (relative to MODEL_PATH unless absolute)
    - MODEL_FACTORY: dotted path to a function that builds the model architecture.
      Required ONLY if the weights file is a pure state_dict.
    - TORCH_DEVICE: cpu|mps|cuda (handled in inference; we keep model on CPU here by default)

    Returns
    -------
    torch.nn.Module (or torch.jit.ScriptModule)
    """
    import torch

    model_dir = Path(_env("MODEL_PATH", "/app/models"))
    if not model_dir.exists():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {model_dir}. Ensure ./models is mounted to /app/models."
        )

    weights_path = _pick_weights_file(model_dir)

    # 1) Try TorchScript first (most robust in production)
    try:
        m = torch.jit.load(str(weights_path), map_location="cpu")
        m.eval()
        return m
    except Exception:
        pass

    # 2) Try torch.load (may return nn.Module OR a dict)
    obj = torch.load(str(weights_path), map_location="cpu")

    # If it's already a module, great.
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj

    # If it's a state_dict, require a factory to build the architecture.
    if isinstance(obj, dict):
        factory_path = os.getenv("MODEL_FACTORY")
        if not factory_path:
            raise ValueError(
                "Loaded weights look like a state_dict (dict). To load it, set MODEL_FACTORY to a callable "
                "that constructs your model architecture, e.g. 'backend.src.models.arch:build_model'."
            )

        build_model = _import_callable(factory_path)
        model = build_model()
        if not isinstance(model, torch.nn.Module):
            raise TypeError("MODEL_FACTORY must return a torch.nn.Module")

        # Some checkpoints store nested keys
        state_dict = obj.get("state_dict", obj)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Make load issues explicit but non-fatal for MVP.
        if missing or unexpected:
            # Avoid noisy prints in server logs unless explicitly enabled.
            if os.getenv("MODEL_LOAD_DEBUG", "0") == "1":
                print(f"[loader] Missing keys: {missing}")
                print(f"[loader] Unexpected keys: {unexpected}")

        return model

    raise TypeError(
        f"Unsupported model artifact type loaded from {weights_path}: {type(obj)}. "
        "Provide a TorchScript model or a pickled nn.Module, or set MODEL_FACTORY for state_dict checkpoints."
    )


def load_model() -> Any:
    """Backward-compatible alias."""
    return get_model()