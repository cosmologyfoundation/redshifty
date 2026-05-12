# NERSC / Perlmutter scaffolding

Run the spectrum tokenizer pretrain (and later, the transformer) against
DESI DR1 on Perlmutter. Self-contained — everything you need to go from
a fresh login to a submitted job lives in this folder.

## What's here

| File | Purpose |
|---|---|
| `setup_env.sh` | One-time module load + pip install + scratch dirs |
| `build_dr1_index.py` | Walks `/global/cfs/cdirs/desi/public/dr1` and writes a JSONL manifest of healpix coadds |
| `dr1_dataset.py` | `DR1IndexedDataset` — opens FITS on demand from a manifest |
| `dr1_tokenized_dataset.py` | Wraps `DR1IndexedDataset` with spectrum + redshift tokenizers; emits Approach A/B sequences |
| `pretrain_tokenizer.py` | Single-GPU AMP loop; trains `SpectrumTokenizer` from `src/tokenizers/spectrum.py` |
| `train_transformer.py` | Single-GPU AMP loop; trains `SpectrumTransformer` with a pretrained tokenizer |
| `smoke_tokenizer.slurm` / `pretrain_tokenizer.slurm` | 10-min smoke + 24h full pretrain |
| `smoke_transformer.slurm` / `train_transformer.slurm` | 10-min smoke + 24h full transformer training (Approach A or B) |
| `ddp_template.slurm` | 4-GPU template for after the trainer is promoted to DDP |
| `export_ddp_vars.sh` | SLURM env var → torch.distributed env var helper |

## Quickstart

You need: a NERSC account on Perlmutter and an allocation under project
**`deepsrch`** (GPU jobs use the `_g` suffix → `deepsrch_g`).

```bash
# 0. ssh in, clone the repo onto $HOME or $CFS
ssh perlmutter.nersc.gov
cd ~ && git clone <your repo url> FoundationModel
cd FoundationModel

# 1. one-time env setup (loads pytorch module, pip-installs astropy/fitsio/tqdm)
bash nersc/setup_env.sh

# 2. submit the smoke job (10 min, debug-equivalent)
sbatch nersc/smoke_tokenizer.slurm

# 3. once smoke passes, submit the full pretrain
sbatch nersc/pretrain_tokenizer.slurm
```

The smoke job builds its own tiny manifest (5 healpix → ~few hundred
spectra) the first time it runs, then reuses it. The full pretrain does
the same with a 2000-healpix manifest.

## Account / QOS notes

- All scripts default to `--account=deepsrch_g`. If your allocation has a
  different GPU project name, override per-submission:
  ```bash
  sbatch -A <other_account> nersc/smoke_tokenizer.slurm
  ```
  or edit the `#SBATCH -A` line.
- We submit in the **`shared`** QOS so each job uses just 1 of the 4 GPUs
  on a Perlmutter node — you pay 1/4 the allocation hours vs `regular`.
  Single-GPU jobs cap at 2 GPUs in shared. Wallclock is up to 48h.
- For the future 4-GPU DDP run, switch to `regular` (or `debug` for
  short tests) and request a full node with `--gpus-per-node=4`.

QOS reference: <https://docs.nersc.gov/jobs/policy/>.

## Filesystem placement (matters!)

Perlmutter has three filesystems. We use them like this:

| Where | What lives there | Why |
|---|---|---|
| `$HOME` | the cloned repo only | small (40 GB quota) — **don't put checkpoints here** |
| `$SCRATCH/deepsrch/` | manifests, checkpoints, logs *during* training | high-perf Lustre |
| `$CFS/desi/public/dr1` | the dataset (read-only) | already there, world-readable |
| `/global/cfs/cdirs/deepsrch/$USER/checkpoints/<run>/` | best/final checkpoint mirror | multi-TB project quota, survives `$SCRATCH` purge |

**Why not mirror to the repo under `$HOME/redshifty/checkpoints/`?** Because
`$HOME` is 40 GB. With ~300 MB best.pt files written on every val-loss
improvement, you'll exhaust quota mid-run and the training process will
crash on `OSError: [Errno 122] Disk quota exceeded`. The SLURM scripts
default `CFS_OUT` to `/global/cfs/cdirs/deepsrch/$USER/checkpoints/$RUN_NAME`
for this reason. As a safety net, both trainers also wrap the mirror in a
try/except since Phase 10 — a failed mirror prints a warning but does not
kill the training process.

`$SCRATCH` is **purged** after ~8 weeks idle. `pretrain_tokenizer.py`
mirrors the best checkpoint to `$CFS_OUT` (defaulted into the repo
checkpoint dir) on every val improvement and again at end — so your
final artifact is safe even if you don't touch the run for a month.

## DR1 layout (what `build_dr1_index.py` walks)

```
/global/cfs/cdirs/desi/public/dr1/
└── spectro/redux/iron/healpix/
    ├── sv1/{bright,dark,backup}/{hpix_group}/{healpix}/coadd-...fits
    ├── sv3/{bright,dark}/...
    └── main/{bright,dark,backup}/...
```

DR1 production = **iron**. SV3 is the "one-percent" survey we used in the
research log; `main` is the full DR1 main survey (much larger). Default
manifest pulls `sv3 + main × bright + dark`.

If you want to filter by target type or pre-cut on z-quality, the
authoritative catalog is
`/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits`
(~21 GB). The current dataset filters at __getitem__ time using the
per-healpix `redrock-*.fits` `ZWARN` and `COADD_FIBERSTATUS` columns,
which is good enough for tokenizer pretraining.

## Monitoring jobs

```bash
sqs                              # your queue
sacct -j <jobid> --format=...    # post-mortem
tail -F tok-pretrain-<jobid>.out # live log (in $SLURM_SUBMIT_DIR)
```

Per-step metrics are written to
`$SCRATCH/deepsrch/checkpoints/<run_name>/metrics.jsonl` — JSONL, one
record per log step or val pass. Plot it locally with whatever you like.

## Common first-job pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| `sbatch: error: invalid account` | wrong GPU account; missing `_g` | confirm with `iris` portal; pass `-A …_g` |
| Job runs but `torch.cuda.is_available()` is False | forgot `--gpus=…` line | already set in scripts |
| `ModuleNotFoundError: astropy` | `setup_env.sh` not run, or pytorch module changed | re-run `bash nersc/setup_env.sh` |
| FITS read very slow | reading from `$CFS` with many workers | reduce `--num-workers`, or stage a manifest's worth of files to `$SCRATCH` |
| Job killed at 8 weeks | `$SCRATCH` purge ate the checkpoint | mirror to `$CFS_OUT` is what `pretrain_tokenizer.py` does — use that copy |
| `ImportError: cannot import name 'GradScaler'` | very old torch; we use `torch.amp.GradScaler` API | upgrade to `pytorch/2.3.1` or newer module |

## Weights & Biases

Both NERSC trainers (`pretrain_tokenizer.py`, `train_transformer.py`) log
to wandb out of the box. Setup:

1. Put your API key in a `.env` at the repo root (gitignored):
   ```
   WANDB_API_KEY=<your-key>
   ```
   `setup_env.sh` already pip-installs `python-dotenv` and `wandb`.

2. Submit normally; the trainers call `load_dotenv()` and `wandb.init()`
   automatically. Default project is `redshifty` (override with
   `--wandb-project`).

Modes (env var or CLI flag):

| Mode | When to use |
|---|---|
| `online` (default) | Compute node has outbound HTTPS — most Perlmutter jobs |
| `offline` | Network is flaky or you don't want live logging |
| `disabled` | Smoke tests / debugging |

Override per-run:
```bash
# pass via CLI...
... --wandb-mode offline --wandb-project redshifty-debug

# ...or via env var
WANDB_MODE=offline sbatch nersc/train_transformer.slurm
```

If you ran offline, sync from a login node afterwards:
```bash
wandb sync $SCRATCH/deepsrch/wandb/<run-name>
```

## Redshift loss weighting

`train_transformer.py` accepts `--redshift-loss-weight` (default **50.0**).
This multiplies the position-0 (redshift) cross-entropy term relative to
the position-1+ (spectrum) term. With 272 spectrum tokens vs 1 redshift
token, the default of 50 gives redshift ~15% of the effective gradient
share (vs ~0.4% unweighted), which is enough to actually train the
redshift prediction. Set `--redshift-loss-weight 1.0` to disable.

## Encoder masking (BERT-style)

`train_transformer.py` accepts `--encoder-mask-ratio` (default **0.15**).
A fraction of spectrum tokens in the *encoder* input are replaced with
`[MASK]` before the model sees them. The decoder input and target are
unchanged. This forces the model to actually infer spectrum tokens from
context — without it, cross-attention learns a trivial positional copy
and `spec_acc` becomes a dishonest 99%+.

Three spec accuracy numbers are logged per run:

| metric | what it measures |
|---|---|
| `val/spectrum_acc` | accuracy over all decoder spectrum positions (masked + unmasked). Partially inflated by the unmasked copy. |
| `val/masked_spec_acc` | accuracy at decoder positions whose encoder position was `[MASK]`. **The honest training-time number.** |
| `val_ar/spectrum_acc` | autoregressive generation, no teacher forcing. **The honest generation-time number.** Logged at every best-checkpoint update and at end of run. |

Set `--encoder-mask-ratio 0.0` to disable (will reproduce the dishonest 99% pattern).

## Held-out healpix split

`--healpix-holdout-frac` (default **0.05**) reserves entire healpix files
for validation rather than splitting by row. Eliminates same-pointing
leakage (fibers in the same exposure share sky background, conditions).

## Wandb model artifacts (auto-uploaded)

By default, `train_transformer.py` uploads a slim model-only copy of
`best.pt` (with redshift-tokenizer state baked in, no optimizer/scaler)
to wandb on each best-checkpoint update and at end of run. Stored as a
versioned `wandb.Artifact`, viewable in the project's Artifacts tab.

Notebook usage (no `scp` required):

```python
ARTIFACT_URI = 'jonathansamuel/redshifty/approach_a_<run-name>:best'
api = wandb.Api()
art = api.artifact(ARTIFACT_URI, type='model')
art.download(root='checkpoints/wandb_artifacts/...')
```

The visualization notebook `notebooks/07_visualize_predictions.ipynb`
has a cell for this. Set `ARTIFACT_URI = None` to fall back to a local
`scp`-ed checkpoint.

Disable artifact upload with `--no-push-wandb-artifact` (e.g. for sweep
runs where you don't want to flood storage).

**Auto-pruning of old versions.** `log_model_artifact()` defaults to
`keep_only_latest=True`: after each successful upload it deletes prior
versions of the same artifact (stripping any protected aliases like
`latest` first). So a run that improves val-loss 10 times will only
have *one* `:vN` version in wandb at the end — the best one — and your
project storage stays small. Set `keep_only_latest=False` if you want
the full version history.

## Staging DR1 from CFS to SCRATCH (for I/O speedup)

`$CFS` is *not tuned* for the random small-file FITS reads that
DataLoader workers do — it was the bottleneck in Phase 8 (1.05 step/s
on a 24M-param tokenizer; ~200-500 spec/s would be GPU-bound). Staging
the relevant healpix files to `$SCRATCH` (Lustre) once before a long
training run typically gives a **5-10× step-rate speedup**.

Workflow:

```bash
# 1. (One-time) build the manifest you want to train on
python nersc/build_dr1_index.py \
    --root /global/cfs/cdirs/desi/public/dr1 \
    --surveys sv3 main --programs bright dark \
    --max-healpix 2000 \
    --out $SCRATCH/deepsrch/manifests/dr1_2k.jsonl

# 2. Stage CFS -> SCRATCH (CPU-only SLURM job; ~5-10 min for ~50 GB)
SRC_MANIFEST=$SCRATCH/deepsrch/manifests/dr1_2k.jsonl \
    sbatch nersc/stage_to_scratch.slurm

# 3. Train using the rewritten manifest pointing at SCRATCH paths
MANIFEST=$SCRATCH/deepsrch/manifests/dr1_2k_scratch.jsonl \
    TOKENIZER_CKPT=$TOK APPROACH=a \
    sbatch nersc/train_transformer.slurm
```

The staging script is **idempotent** — re-running on the same source
manifest skips files already present in the destination. You can stage
small manifests directly from a login node too:

```bash
python nersc/stage_to_scratch.py \
    --src-manifest $SCRATCH/deepsrch/manifests/dr1_smoke.jsonl \
    --dst-root     $SCRATCH/deepsrch/dr1_staged \
    --dst-manifest $SCRATCH/deepsrch/manifests/dr1_smoke_scratch.jsonl
```

**Watch out for `$SCRATCH` purge**: files not accessed within ~8 weeks
get deleted. Stage right before a training campaign, not months ahead.

References: <https://docs.nersc.gov/filesystems/>,
<https://docs.nersc.gov/filesystems/perlmutter-scratch/>.

## SLURM env-var overrides

`train_transformer.slurm` reads these env vars at submit time so you can
sweep without editing scripts:

| env var | flag | default |
|---|---|---|
| `REDSHIFT_LOSS_WEIGHT` | `--redshift-loss-weight` | 50.0 |
| `ENCODER_MASK_RATIO` | `--encoder-mask-ratio` | 0.15 |
| `HEALPIX_HOLDOUT_FRAC` | `--healpix-holdout-frac` | 0.05 |
| `AR_EVAL_BATCHES` | `--ar-eval-batches` | 4 |
| `WANDB_MODE` | `--wandb-mode` | online |
| `WANDB_PROJECT` | `--wandb-project` | redshifty |
| `STEPS`, `BATCH_SIZE`, `LR`, `NUM_WORKERS` | — | see script |
| `MAX_HEALPIX`, `MANIFEST`, `RUN_NAME`, `CFS_OUT` | — | see script |

Example: weight sweep for Approach A across 7 values:

```bash
for w in 1 5 10 20 50 100 200; do
    REDSHIFT_LOSS_WEIGHT=$w APPROACH=a STEPS=8000 \
        RUN_NAME=sweep_a_w${w} \
        sbatch -t 03:00:00 nersc/train_transformer.slurm
done
```

## Stage 2: Train the transformer with a pretrained tokenizer

Once `pretrain_tokenizer.slurm` (or the trial run) lands a `best.pt` you
trust, train Approach A and Approach B against it.

```bash
# point at the tokenizer checkpoint that came out of stage 1
TOK=$SCRATCH/deepsrch/checkpoints/<tokenizer_run>/best.pt

# 10-min smoke first (validates the loader + tokenizer load path)
TOKENIZER_CKPT=$TOK APPROACH=a sbatch nersc/smoke_transformer.slurm

# full Approach A (24h, shared QOS, 1 GPU)
TOKENIZER_CKPT=$TOK APPROACH=a sbatch nersc/train_transformer.slurm

# full Approach B (independent run; same data, frozen tokenizer)
TOKENIZER_CKPT=$TOK APPROACH=b sbatch nersc/train_transformer.slurm
```

The trainer mirrors `best.pt` and `final.pt` to
`$REPO/checkpoints/nersc/approach_<a|b>_<jobid>/` so the result survives
`$SCRATCH` purge and is git-track-able.

Override knobs (env vars on the `sbatch` line):
`MAX_HEALPIX`, `STEPS`, `BATCH_SIZE`, `LR`, `NUM_WORKERS`, `D_MODEL`,
`N_LAYERS`, `N_HEADS`. See `train_transformer.slurm` for defaults.

You can also load the pretrained tokenizer in the **local** trainer for
debugging: `python scripts/train.py --approach a --tokenizer_ckpt <path>`.

## Going from smoke → real → DDP

1. **Smoke** (`smoke_tokenizer.slurm`): 50 steps, 200 spectra, ~few minutes.
   Confirms environment is correct and `best.pt` is written.
2. **Real single-GPU** (`pretrain_tokenizer.slurm`): 100k steps over
   ~hundreds of thousands of spectra in 24h. Mirrors `best.pt` and
   `final.pt` to `$CFS_OUT`. **This is what unblocks the transformer
   downstream.**
3. **DDP scale-up** (`ddp_template.slurm`): only after step 2 is happy.
   Requires a small code change in `pretrain_tokenizer.py`
   (DistributedDataParallel + DistributedSampler) which is intentionally
   not in this scaffold yet.

## After tokenizer is trained

The tokenizer checkpoint replaces the random init that's currently
forced in `src/datasets/tokenized_dataset.py`. From there:

- Add a `--tokenizer-ckpt PATH` flag to `scripts/train.py` that loads the
  pretrained weights before training the transformer.
- Re-run the Approach A and Approach B trainings at `d_model=768,
  n_layers=6` on full DR1 (a parallel pair of SLURM scripts will go in
  this folder once the tokenizer is in hand).
- Honest val on a held-out healpix subset, not random rows from the
  same files.

## Sources

- DR1 release: <https://data.desi.lbl.gov/doc/releases/dr1/>
- Perlmutter running jobs: <https://docs.nersc.gov/systems/perlmutter/running-jobs/>
- QOS table: <https://docs.nersc.gov/jobs/policy/>
- NERSC PyTorch module: <https://docs.nersc.gov/machinelearning/pytorch/>
- DDP example we cribbed from: <https://github.com/NERSC/nersc-dl-multigpu>
