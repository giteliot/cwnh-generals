# Kaggle GPU

Free: 30 GPU-h/week, 12 h/session. Push your code first — Kaggle clones from GitHub.

## Notebook settings

Accelerator: **GPU T4 x2** · Internet: **On** · Persistence: **Files only**

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
  --envs 4096 --steps 32 \
  --channels 32 --blocks 4 --groups 8 \
  --minibatches 32 --epochs 2 \
  --pool-size 8192 --gamma 0.997 \
  --anneal-frac 10 --checkpoint-every 50 \
  --resume /kaggle/working/ckpt.pkl \
  --out    /kaggle/working/ckpt.pkl"""
proc = subprocess.Popen(cmd, shell=True, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
print("pid", proc.pid)
```

Checkpoint **must** go to `/kaggle/working/` — nothing else survives the session.

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

## Cell 6 — stop

```python
proc.terminate()
```

---

## Watch these three

1. **`v` under ~1.** It hit 12,964,842 once and the policy collapsed.
2. **Land count** from an eval — two runs ended holding 1 tile, passing every turn. Win rate won't show it.
3. `W/L/D` moving toward W; `H` falling.

## Continuing past 12 h

1. New notebook version → **Add Data → Notebook Output** → previous version.
2. Add before Cell 4:
   ```python
   !cp /kaggle/input/<notebook-name>/ckpt.pkl /kaggle/working/ckpt.pkl
   ```
3. Run Cell 4 unchanged. Keep `--updates` identical — it's the anneal denominator.

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
