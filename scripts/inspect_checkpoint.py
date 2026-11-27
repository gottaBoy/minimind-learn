"""Inspect and optionally load a MiniMind model or resume checkpoint."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_utils import project_path, torch_load


def load_checkpoint(path):
    """Load any course checkpoint onto CPU using the restricted loader."""
    path = Path(path)
    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a dictionary")
    return payload


def is_resume_checkpoint(payload):
    """Resume files contain model, optimizer and progress together."""
    return "model" in payload and "optimizer" in payload and "step" in payload


def load_model_state(payload, hidden_size, num_hidden_layers, use_moe=False,
                     device="cpu"):
    """Create the matching architecture, then copy model tensors into it."""
    state_dict = payload["model"] if is_resume_checkpoint(payload) else payload
    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
    )
    model = MiniMindForCausalLM(config).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    return model


def optimizer_summary(payload, model, device):
    """Load AdamW state and report the first state tensor's device and step."""
    if "optimizer" not in payload:
        return None
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    optimizer.load_state_dict(payload["optimizer"])
    states = list(optimizer.state.values())
    first = states[0] if states else {}
    return {
        "parameter_groups": len(optimizer.param_groups),
        "state_entries": len(states),
        "first_step": first.get("step").item() if "step" in first
        and torch.is_tensor(first["step"]) else first.get("step"),
        "first_state_device": str(next(
            (value.device for value in first.values() if torch.is_tensor(value)),
            torch.device(device),
        )),
        "model_device": str(next(model.parameters()).device),
    }


def inspect(path, hidden_size, num_hidden_layers, use_moe=False, device="cpu"):
    payload = load_checkpoint(path)
    resume = is_resume_checkpoint(payload)
    print("path:", Path(path).resolve())
    print("kind:", "resume checkpoint" if resume else "model weights")
    print("top-level fields:", sorted(payload))
    if resume:
        print("epoch:", payload.get("epoch"))
        print("completed batch step:", payload.get("step"))
        print("saved world_size:", payload.get("world_size", 1))
        print("scaler fields:", sorted(payload.get("scaler", {})))
        print("optimizer fields:", sorted(payload["optimizer"]))
        states = payload["optimizer"].get("state", {})
        if states:
            first = next(iter(states.values()))
            print("optimizer state tensors:",
                  {key: (tuple(value.shape), str(value.dtype))
                   for key, value in first.items() if torch.is_tensor(value)})

    model_state = payload["model"] if resume else payload
    print("model tensor count:", len(model_state))
    first_name, first_tensor = next(iter(model_state.items()))
    print("first model tensor:", first_name, tuple(first_tensor.shape),
          str(first_tensor.dtype), str(first_tensor.device))

    model = load_model_state(
        payload, hidden_size, num_hidden_layers, use_moe=use_moe, device=device,
    )
    print("loaded model device:", next(model.parameters()).device)
    print("loaded model dtype:", next(model.parameters()).dtype)
    if resume:
        print("optimizer after load:", optimizer_summary(payload, model, device))
    return payload, model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="model .pth or *_resume.pth path")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--device", default="cpu",
                        help="target model device, for example cpu or cuda:0")
    args = parser.parse_args(argv)
    path = args.path if args.path.is_absolute() else Path(project_path(args.path))
    if not path.is_file():
        parser.error(f"checkpoint does not exist: {path}")
    inspect(
        path, hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe, device=args.device,
    )


if __name__ == "__main__":
    main()
