import argparse
import pickle
from pathlib import Path

DROP = ("opt_state", "key", "bank", "league")


def main():
    ap = argparse.ArgumentParser(description="list or extract the banked policies inside a training checkpoint")
    ap.add_argument("checkpoint")
    ap.add_argument("--index", type=int, default=None, help="bank slot to write out as a standalone checkpoint")
    args = ap.parse_args()

    blob = pickle.load(open(args.checkpoint, "rb"))
    bank = blob["bank"]

    if args.index is None:
        print(f"{args.checkpoint}: update {blob['update']}, {len(bank)} banked policies")
        print("  the score is what the FINAL policy scores against that entry -- below 0.50 means")
        print("  the entry is stronger than the checkpoint's own params\n")
        for i, (wins, draws, games) in enumerate(blob["league"]["bank"]):
            score = (wins + 0.5 * draws) / games if games else 0.0
            flag = "  <- stronger than final" if games and score < 0.5 else ""
            print(f"  b{i}  {score:.2f}  over {games:.0f} games{flag}")
        return

    out = Path(args.checkpoint).with_name(f"{Path(args.checkpoint).stem}_b{args.index}.pkl")
    kept = {k: v for k, v in blob.items() if k not in DROP}
    pickle.dump({**kept, "params": bank[args.index], "bank": [], "league": None}, open(out, "wb"))
    print(f"wrote {out}  (eval/package only -- optimizer state and RNG key are not carried, so it is not resumable)")


if __name__ == "__main__":
    main()
