"""Fast end-to-end check for imports, forward, loss, and generation."""

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    config = MiniMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=384,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=256,
        flash_attn=True,
    )
    model = MiniMindForCausalLM(config).to(device).eval()
    encoded = tokenizer("MiniMind smoke test", return_tensors="pt").to(device)
    encoded.pop("token_type_ids", None)
    with torch.no_grad():
        output = model(**encoded, use_cache=True)
        assert torch.isfinite(output.logits).all()
        cached = model(
            input_ids=encoded.input_ids[:, -1:],
            past_key_values=model(encoded.input_ids[:, :-1], use_cache=True).past_key_values,
            use_cache=True,
        )
        torch.testing.assert_close(cached.logits[:, -1], output.logits[:, -1], atol=1e-4, rtol=1e-4)
        generated = model.generate(
            **encoded,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(f"device={device}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"logits_shape={tuple(output.logits.shape)}")
    print(f"generated={tokenizer.decode(generated[0], skip_special_tokens=True)!r}")
    model.train()
    loss = torch.nn.functional.cross_entropy(
        model(encoded.input_ids[:, :-1]).logits.flatten(0, 1),
        encoded.input_ids[:, 1:].flatten(),
    )
    loss.backward()
    assert torch.isfinite(loss) and model.lm_head.weight.grad is not None
    print(f"loss={loss.item():.6f}; backward and KV cache checks passed")


if __name__ == "__main__":
    main()
