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
| `pretrain_tokenizer.py` | Single-GPU AMP loop; trains `SpectrumTokenizer` from `src/tokenizers/spectrum.py` |
| `smoke_tokenizer.slurm` | 10-min shared-QOS smoke job; validates env + data path |
| `pretrain_tokenizer.slurm` | 24h shared-QOS full pretrain |
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
| `$HOME` | the cloned repo | small, durable |
| `$SCRATCH/deepsrch/` | manifests, checkpoints, logs *during* training | high-perf Lustre |
| `$CFS/desi/public/dr1` | the dataset (read-only) | already there, world-readable |
| `checkpoints/nersc/<run>/` (in repo) | best/final checkpoint mirror | survives `$SCRATCH` purge |

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
