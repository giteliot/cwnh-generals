import argparse
import pickle

import jax
import numpy as np


def flatten_params(params, prefix=""):
    flat = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, f"{name}/"))
        else:
            flat[name] = np.asarray(value, dtype=np.float32)
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default="competition/agents/ppo_bot/weights.npz")
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as fh:
        blob = pickle.load(fh)

    flat = flatten_params(jax.tree.map(np.asarray, blob["params"])["params"])
    flat["meta/channels"] = np.asarray(blob["channels"], np.float32)
    flat["meta/blocks"] = np.asarray(blob["blocks"], np.float32)
    flat["meta/groups"] = np.asarray(blob.get("groups", 8), np.float32)

    np.savez_compressed(args.out, **flat)
    total = sum(v.size for k, v in flat.items() if not k.startswith("meta/"))
    print(f"wrote {args.out}: {len(flat)} arrays, {total:,} parameters")


if __name__ == "__main__":
    main()
