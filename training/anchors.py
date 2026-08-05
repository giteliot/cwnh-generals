"""Frozen opponents: fixed reference points that eviction can never remove.

The bank is not a yardstick. `choose_opponent` evicts the bank entry we beat
MOST, so the pool always holds the N hardest snapshots and its win rate sits at
~0.50 whatever the policy does. Updates 2500-9626 of the v12 run held bank 0.556
-> 0.494 while the same policy went 95.3% -> 98.1% against the scripted bots:
the pool measured its own composition, not the agent.

The scripted bots are the only fixed reference in the system, and they saturate.
At 98.1% there is 1.9% of headroom left, after which nothing in the log can tell
improvement from decay.

An anchor is a checkpoint loaded from disk. It is never trained, never banked and
never evicted, so its score means exactly one thing: how the live policy does
against a FIXED opponent of known strength. That number has room to move for the
whole run.

Three formats load, so any artefact you kept is usable as an anchor:
  *.pkl  a training checkpoint written by train.save()
  *.npz  exported weights written by training/export.py
  *.zip  a competition submission bundle (reads weights.npz from inside)
"""
import io
import os
import pickle
import zipfile
from typing import NamedTuple

import numpy as np

from .features import GLOBAL_NAMES, PLANE_NAMES
from .network import NUM_OPPONENT_KINDS

VALUE_HEAD = "Dense_2"   # nn.Dense call order in ActorCritic: globals, pass_logit, VALUE, value_out


class Anchor(NamedTuple):
    name: str
    path: str
    params: dict


def _unflatten(flat):
    """Invert export.flatten_params: 'params/Conv_0/kernel' -> nested dict."""
    tree = {}
    for key, value in flat.items():
        if key.startswith("meta/"):
            continue
        node = tree
        parts = key.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = np.asarray(value, dtype=np.float32)
    return tree


def _read_npz(data):
    with np.load(data) as npz:
        return _unflatten({k: npz[k] for k in npz.files})


def read_params(path):
    """Load a parameter tree from a checkpoint, an export or a submission zip."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if "params" not in blob:
            raise SystemExit(f"{path} is a pickle with no 'params' key -- not a checkpoint")
        # A checkpoint records its observation layout, so prefer .pkl over .npz
        # for an anchor: shape checks catch a changed plane COUNT, but only these
        # names catch planes that were reordered or repurposed at the same count,
        # which would load clean and play nonsense.
        for field, current in (("planes", PLANE_NAMES), ("globals", GLOBAL_NAMES)):
            saved = blob.get(field)
            if saved is not None and saved != current:
                raise SystemExit(
                    f"{path} was trained with different {field} than the code now builds.\n"
                    f"  added since:   {[p for p in current if p not in saved] or 'none'}\n"
                    f"  removed since: {[p for p in saved if p not in current] or 'none'}\n"
                    f"  reordered:     {saved != current and sorted(saved) == sorted(current)}\n"
                    "Its observation no longer means what the code feeds it; it cannot be an anchor."
                )
        return {"params": blob["params"]["params"]}
    if ext == ".npz":
        with open(path, "rb") as fh:
            return {"params": _read_npz(io.BytesIO(fh.read()))}
    if ext == ".zip":
        with zipfile.ZipFile(path) as bundle:
            names = [n for n in bundle.namelist() if n.endswith("weights.npz")]
            if not names:
                raise SystemExit(f"{path} has no weights.npz inside: {bundle.namelist()}")
            return {"params": _read_npz(io.BytesIO(bundle.read(names[0])))}
    raise SystemExit(f"{path}: anchors must be .pkl, .npz or .zip, got '{ext}'")


def conform(params, reference, path):
    """Make an anchor's weights loadable by the CURRENT network.

    Only one layer legitimately changes width between runs: the value head takes
    `concat([pooled, opponent_one_hot])`, so its kernel gains a row per opponent
    kind. ppo_v11 was trained with 6 kinds and has 90 rows where the code now
    builds 88. Every other shape must match exactly -- a different channel count
    or plane count is a real incompatibility, not something to paper over.

    The opponent one-hot rows are zeroed rather than copied across. An anchor is
    only ever the OPPONENT, and `make_rollout.net_branch` reads its logits and
    discards its value, so those rows are never used. Zeroing says that plainly
    instead of carrying six stale columns into four slots.
    """
    flat, ref = dict(_leaves(params)), dict(_leaves(reference))
    if set(flat) != set(ref):
        missing = sorted(set(ref) - set(flat))
        extra = sorted(set(flat) - set(ref))
        raise SystemExit(
            f"{path} has a different network structure than the code builds.\n"
            f"  missing: {missing or 'none'}\n"
            f"  extra:   {extra or 'none'}"
        )

    value_kernel = f"params/{VALUE_HEAD}/kernel"
    fixed = {}
    for key, want in ref.items():
        have = flat[key]
        if have.shape == want.shape:
            fixed[key] = np.asarray(have, dtype=np.float32)
            continue
        if key != value_kernel or have.shape[1:] != want.shape[1:]:
            raise SystemExit(
                f"{path}: {key} is {have.shape}, the code builds {want.shape}. Only the value "
                "head's opponent one-hot rows may differ; this is a real architecture change."
            )
        pooled = want.shape[0] - NUM_OPPONENT_KINDS
        if have.shape[0] < pooled:
            raise SystemExit(
                f"{path}: {key} has {have.shape[0]} rows, fewer than the {pooled} board features "
                "the code pools. The observation changed; this checkpoint is not usable."
            )
        rows = np.zeros(want.shape, dtype=np.float32)
        rows[:pooled] = have[:pooled]
        fixed[key] = rows
    return _rebuild(fixed)


def _leaves(tree, prefix=""):
    for key, value in sorted(tree.items()):
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _leaves(value, f"{name}/")
        else:
            yield name, np.asarray(value)


def _rebuild(flat):
    tree = {}
    for key, value in flat.items():
        node = tree
        parts = key.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


def load(spec, reference):
    """Parse a `--anchors` spec into Anchors, conformed to `reference`'s shapes.

    Spec is comma separated. Each item is `path` or `name=path`; without a name
    the file stem is used, because that is what shows up in the league table.
    """
    out = []
    for item in [s.strip() for s in (spec or "").split(",") if s.strip()]:
        name, _, path = item.rpartition("=")
        path = path or item
        name = name or os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(path):
            raise SystemExit(f"anchor {name!r}: no such file {path}")
        if any(a.name == name for a in out):
            raise SystemExit(f"anchor name {name!r} is used twice; disambiguate with name=path")
        params = conform(read_params(path), reference, path)
        out.append(Anchor(name=name, path=os.path.abspath(path), params=params))
    return out
