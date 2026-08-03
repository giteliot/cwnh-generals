import argparse
import pickle
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BOT = REPO / "competition" / "agents" / "ppo_bot"
FILES = ("run.sh", "main.py", "agent.py", "weights.npz")


def main():
    ap = argparse.ArgumentParser(description="checkpoint -> submission zip")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out) if args.out else REPO / f"{ckpt.stem}_submission.zip"

    sys.path.insert(0, str(REPO))
    from training.features import PLANE_NAMES

    blob = pickle.load(open(ckpt, "rb"))
    saved = blob.get("planes")
    if saved is not None and saved != PLANE_NAMES:
        raise SystemExit(
            f"{ckpt} has {len(saved)} observation planes, the code builds {len(PLANE_NAMES)}. "
            "The NumPy bot mirrors the current code, so the exported weights would not match. "
            "Retrain, or check out the code this checkpoint came from."
        )
    print(f"{ckpt.name}: update {blob.get('update')}, {blob['channels']}ch x {blob['blocks']} blocks, "
          f"stage {blob.get('stage')}")

    subprocess.run([sys.executable, "-m", "training.export", str(ckpt)], cwd=REPO, check=True)

    shutil.rmtree(BOT / "__pycache__", ignore_errors=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(BOT / name, name)

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    for name in FILES:
        print(f"   {name}")


if __name__ == "__main__":
    main()
