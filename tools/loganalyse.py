import argparse
import re
from collections import defaultdict

LINE = re.compile(
    r"^upd\s+(\d+).*?"
    r"(?:H\s+([\d.]+))?.*?"
    r"(?:ep\s+([-+][\d.]+))?.*?"
    r"eps\s+(\d+)\s+W/L/D\s+(\d+)/\s*(\d+)/\s*(\d+)"
    r"(?:.*?vs\s+(\S+))?"
)


def parse(path):
    rows = []
    for line in open(path):
        m = LINE.match(line)
        if m:
            upd, h, ep, eps, w, ls, d, opp = m.groups()
            rows.append(dict(
                upd=int(upd), h=float(h) if h else None, ep=float(ep) if ep else None,
                eps=int(eps), w=int(w), l=int(ls), d=int(d),
                opp=(opp or "?").split("[")[0],
            ))
    return rows


def tally(rows):
    w = sum(r["w"] for r in rows)
    ls = sum(r["l"] for r in rows)
    d = sum(r["d"] for r in rows)
    return w, ls, d, w + ls + d


def table(rows, title):
    print(f"\n{title}")
    by = defaultdict(list)
    for r in rows:
        by[r["opp"]].append(r)
    print(f"  {'opponent':<12}{'win%':>7}{'loss%':>7}{'draw%':>7}{'episodes':>10}{'updates':>9}")
    for opp in sorted(by, key=lambda o: -len(by[o])):
        w, ls, d, t = tally(by[opp])
        if t:
            print(f"  {opp:<12}{100*w/t:7.1f}{100*ls/t:7.1f}{100*d/t:7.1f}{t:10d}{len(by[opp]):9d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--recent", type=int, default=600)
    ap.add_argument("--blocks", type=int, default=500)
    args = ap.parse_args()

    rows = parse(args.log)
    lo, hi = rows[0]["upd"], rows[-1]["upd"]
    print(f"{args.log}: {len(rows)} updates, {lo} -> {hi}")

    selfplay = [r for r in rows if r["opp"] == "self"]
    if selfplay:
        w, ls, d, t = tally(selfplay)
        share = 100 * w / t if t else 0
        verdict = "OK" if 40 <= share <= 60 else "*** BROKEN: identical weights must split ~50/50 ***"
        print(f"\nself-play invariant: {share:.1f}% over {t} episodes   {verdict}")

    table(rows, "LIFETIME")
    table([r for r in rows if r["upd"] > hi - args.recent], f"LAST {args.recent} UPDATES")

    print(f"\nWIN% BY {args.blocks}-UPDATE BLOCK")
    opps = sorted({r["opp"] for r in rows if r["opp"] != "?"})
    print("  " + "".join(f"{o[:9]:>10}" for o in ["block"] + opps))
    for start in range(lo, hi + 1, args.blocks):
        cells = [f"{start:>10}"]
        for opp in opps:
            sel = [r for r in rows if opp == r["opp"] and start <= r["upd"] < start + args.blocks]
            w, _, _, t = tally(sel)
            cells.append(f"{100*w/t:9.1f}%" if t > 30 else f"{'-':>10}")
        print("  " + "".join(cells))

    trend = [r for r in rows if r["ep"] is not None]
    if trend:
        print("\nEPISODE REWARD / ENTROPY")
        step = max(len(trend) // 12, 1)
        for r in trend[::step]:
            print(f"  upd {r['upd']:5d}   ep {r['ep']:+7.2f}   H {r['h'] if r['h'] else 0:.2f}   vs {r['opp']}")


if __name__ == "__main__":
    main()
