# Kaggle GPU

Free: 30 GPU-h/week, 12 h/session. Push your code first — Kaggle clones from GitHub.

## Notebook settings

Settings menu → **Accelerator: GPU T4 x2**, **Internet: On**.

**`/kaggle/working` is volatile.** There is no persistence toggle — assume the
directory is gone when the session ends. Either commit a version (Save Version →
**Save & Run All**), which snapshots it as that version's Output, or download
`ckpt.pkl` yourself before you stop. With `--checkpoint-every 50` you lose at
most ~50 updates either way.

---

## Cell 1 — clone

Store the token first: **Add-ons → Secrets**, label `GITHUB_TOKEN`
([create one](https://github.com/settings/personal-access-tokens): only this repo, Contents read-only).

```python
import os, shutil, subprocess
from kaggle_secrets import UserSecretsClient

os.chdir("/kaggle/working")
CLONE, REPO = "/kaggle/working/repo", "giteliot/cwnh-generals"
shutil.rmtree(CLONE, ignore_errors=True)

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
subprocess.run(["git", "clone", "--quiet",
                f"https://x-access-token:{token}@github.com/{REPO}.git", CLONE], check=True)
subprocess.run(["git", "-C", CLONE, "remote", "set-url", "origin",
                f"https://github.com/{REPO}.git"], check=True)

os.chdir(CLONE)
print(os.getcwd(), sorted(os.listdir())[:6])
```

## Cell 2 — check GPU before installing

```python
!nvidia-smi
import jax
print("jax", jax.__version__, jax.devices())
```

- `nvidia-smi` errors → Accelerator not set. Fix and restart.
- Already shows `CudaDevice` → go to Cell 3.
- GPU present but CPU-only jax → `!pip install -q "jax[cuda12]"` (**never** with `-U`).

## Cell 3 — install (pin jax or it breaks CUDA)

```python
import jax
JAXV = jax.__version__
!pip install -q -e ".[train]" "jax=={JAXV}" "jaxlib=={JAXV}"
```

**Restart the kernel**, then re-run Cells 1 and 2 to confirm `CudaDevice`.

## Cell 4 — launch (returns immediately)

```python
import subprocess
LOG = "/kaggle/working/train.log"
cmd = """cd /kaggle/working/repo && python -u -m training.train \
  --stage medium --updates 20000 \
  --envs 2048 --steps 32 \
  --channels 32 --blocks 8 --groups 8 \
  --minibatches 32 --epochs 2 \
  --pool-size 8192 --gamma 0.9999 \
  --anneal-frac 10 --checkpoint-every 50 \
  --resume /kaggle/working/ckpt.pkl \
  --out    /kaggle/working/ckpt.pkl"""
proc = subprocess.Popen(cmd, shell=True, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
print("pid", proc.pid)
```

Checkpoint **must** go to `/kaggle/working/` — it's the only directory that can be
snapshotted into a version's Output. Download it before you stop.

## Cell 5 — watch

```python
!tail -n 20 /kaggle/working/train.log
```

Live-refreshing version:

```python
import time
from IPython.display import clear_output
while proc.poll() is None:
    clear_output(wait=True)
    !tail -n 25 /kaggle/working/train.log
    time.sleep(30)
```

## Cell 5b — auto-save to a Dataset (so a lost session doesn't cost the model)

One-time: kaggle.com → *Settings → API → Create New Token*. Put `username` and
`key` from the downloaded `kaggle.json` into **Add-ons → Secrets** as
`KAGGLE_USERNAME` and `KAGGLE_KEY`.

Run this **before** Cell 4 — the subprocess inherits the env vars:

```python
import os, json
from kaggle_secrets import UserSecretsClient

sec = UserSecretsClient()
os.environ["KAGGLE_USERNAME"] = sec.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"]      = sec.get_secret("KAGGLE_KEY")

UP = "/kaggle/working/upload"
os.makedirs(UP, exist_ok=True)
json.dump({
    "title": "generals-ckpt",
    "id": f"{os.environ['KAGGLE_USERNAME']}/generals-ckpt",
    "licenses": [{"name": "CC0-1.0"}],
}, open(f"{UP}/dataset-metadata.json", "w"))
```

Chain the upload onto Cell 4's command so it fires when training exits:

```
... --out /kaggle/working/ckpt.pkl \
&& cp /kaggle/working/ckpt.pkl /kaggle/working/upload/ckpt.pkl \
&& kaggle datasets version -p /kaggle/working/upload -m "auto"
```

First time only, use `kaggle datasets create -p /kaggle/working/upload` instead.

`&&` only fires on a **clean exit**, which is not the case you are worried about.
Add a periodic uploader too:

```python
import time, subprocess
while proc.poll() is None:
    time.sleep(1800)
    subprocess.run("cp /kaggle/working/ckpt.pkl /kaggle/working/upload/ckpt.pkl && "
                   "kaggle datasets version -p /kaggle/working/upload -m periodic", shell=True)
```

## Cell 6 — stop

```python
proc.terminate()
```

---

## Watch these three

0. **Receptive field >= board size**, printed at startup. `--blocks 4` gives 19
   cells, which cannot see across a 22-wide board; `--blocks 8` gives 35.
1. **`v` under ~1.** It hit 12,964,842 once and the policy collapsed.
2. **Land count** from an eval — two runs ended holding 1 tile, passing every turn. Win rate won't show it.
3. `W/L/D` moving toward W; `H` falling.

## Continuing past 12 h

Keep `--updates` identical across sessions — it's the anneal denominator. Note
it is **cumulative**: resuming at update 2100 with `--updates 12000` gives you
9900 more, not 12000.

**If you committed a version:** Add Data → **Notebook Output** → previous
version, then before Cell 4:

```python
!cp /kaggle/input/<notebook-name>/ckpt.pkl /kaggle/working/ckpt.pkl
```

**If it didn't survive, or the checkpoint is only on your laptop** — upload it as
a Dataset:

1. [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset** →
   drag in `ckpt.pkl` → title `generals-ckpt`.
2. Notebook → **Add Data** → *Your Datasets* → `generals-ckpt`.
3. Before Cell 4:

```python
import shutil, os, pickle
os.makedirs("/kaggle/working", exist_ok=True)
shutil.copy("/kaggle/input/generals-ckpt/ckpt.pkl", "/kaggle/working/ckpt.pkl")
print("resuming from update", pickle.load(open("/kaggle/working/ckpt.pkl","rb"))["update"])
```

`/kaggle/input` is read-only, so the copy is required. For later runs use
**New Version** on the same dataset rather than creating a new one — attached
notebooks pick up the latest automatically.

## Getting the model out

Download `ckpt.pkl` from the version's Output tab, then locally:

```bash
.venv/bin/python tools/watch.py ppo:ckpt.pkl expander --stage medium --games 20 --no-gui
.venv/bin/python tools/play_human.py --agent ppo --checkpoint ckpt.pkl --stage medium
.venv/bin/python -m training.export ckpt.pkl
```

---

## Gotchas

| symptom | fix |
|---|---|
| `cuInit(0) failed: CUDA error 303` | pip upgraded jax past the driver. Fresh session, pinned install (Cell 3). |
| `Unable to read current working directory`, git exit 128 | Kernel in a deleted dir. `os.chdir("/kaggle/working")` first. Not auth. |
| `Username for 'https://github.com'` | Private repo, no stdin. Use the token cell. |
| `jax.devices()` CPU after install | Kernel not restarted. |
| GPU memory reads 11 GiB | JAX preallocates 75%. Not real usage. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing jax to see actual. |
| Only GPU 0 busy | Expected — JAX is single-device. Raise `--envs` before considering `pmap`. |
| OOM | Raise `--minibatches` (it *divides* the batch, so higher = smaller = less memory). |
| `IndexError: axis 0 with size 0` | `--pool-size` below the board-size combo count (16 for `--stage full`). |
| Session died at ~40 min | Interactive idle timeout. For unattended runs use Save & Run All (Commit) with a blocking `!cd ... \| tee` — `Popen` is killed when the notebook ends. |
| Log looks frozen | Missing `python -u`. |
| `/kaggle/working` empty next session | Expected — it does not persist. Commit a version, or download `ckpt.pkl` and re-upload it as a Dataset. |
