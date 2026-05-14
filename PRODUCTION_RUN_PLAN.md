# Production Run Plan — Best DESI Spectrum Foundation Model

**Status:** drafted 2026-05-12 after Phase 10 resolved the A-vs-B thesis test (A wins, B fails). Goal pivot: stop comparing approaches; build the highest-accuracy unimodal foundation model for DESI galaxy spectra.

**Compute posture (updated 2026-05-12):** allocation is **not a constraint**. Wallclock / deadline is. This means: default to 4-GPU DDP everywhere (3× faster for 25% more A100-hours, irrelevant when hours are abundant), run Phases 1/2/3 unconditionally, and treat the "Beyond the baseline plan" section below as the real plan.

**Baseline to beat (Phase 10, `phase10_mask50_a_big`, single-GPU, ~200 healpix, batch=32, mask=0.50, step=4000):**
- `val/spectrum_acc` (TF): not yet measured cleanly
- `val/masked_spec_acc` (TF, honest): ~28% (pre-mask) → climbing post-mask
- `val/redshift_acc` (TF): **73.8%**
- `val_ar/redshift_acc` (AR, honest): **55.0%**

These were obtained on a sliver of DR1 (~200 healpix, a few thousand training spectra). The plan below scales data ~50×, trains ~25× longer, and optionally retrains the tokenizer.

---

## Success criteria (what to claim in the writeup)

Production run is "successful" if, at the end of training:

| metric | minimum target | stretch |
|---|---|---|
| `val/redshift_acc` (TF) | ≥ 85% | ≥ 92% |
| `val_ar/redshift_acc` (AR) | ≥ 75% | ≥ 85% |
| `val/masked_spec_acc` (TF) | ≥ 50% | ≥ 65% |
| `val_ar/spectrum_acc` (AR) | within 5–10% of masked TF | matches masked TF |

Gap between TF and AR is the honesty gauge — large gap = decoder is still leaning on teacher forcing despite masking; small gap = the encoder genuinely learned to encode the spectrum.

**Abort/redirect conditions** during training:
- SM util drops below 90% sustained → stage data to SCRATCH, restart.
- `val/redshift_acc` plateaus before step 20k → likely tokenizer is the ceiling; trigger Tokenizer v2 (Phase 2 below).
- $HOME quota warning during run → kill, free space, re-launch (CFS_OUT default already fixed in commit `37badb0` but verify).

---

## Order of operations

These run sequentially because each depends on the previous. Total wallclock: ~3–4 days if both Tokenizer v2 and long transformer train at full scale.

### Phase 1 — Long transformer run with current tokenizer v1 (~12h on DDP4)

A real foundation-model-scale baseline on Tokenizer v1. Runs as a 4-GPU DDP job in `regular` QOS. Tokenizer v1 (`val_recon ≈ 1.35`) provides the baseline number even if v2 ships later.

### Phase 2 — Tokenizer v2 (~8h on DDP4, runs in parallel with Phase 1)

Launch alongside Phase 1 as a separate 4-GPU DDP job — no reason to serialize when allocation is abundant. Improvements:
- Top-hat 5-px conv preprocessing (Phase 8 backlog) — smooths spectrum noise before the encoder sees it
- 3× more spectra (10k healpix vs ~200 used for v1)
- Same architecture (ConvNeXt-V2 + LFQ); larger codebook only if v1 utilization analysis says codes are saturated

### Phase 3 — Long transformer run with tokenizer v2 (~12h on DDP4)

Runs unconditionally once Phase 2 finishes, regardless of whether v2 hits the 10% `val_recon` improvement bar — the v1-vs-v2 comparison is itself a writeup-worthy ablation.

---

## Phase 1 — Detailed steps

### 1.1 Build the big manifest

```bash
cd ~/redshifty

python nersc/build_dr1_index.py \
    --root /global/cfs/cdirs/desi/public/dr1 \
    --surveys sv3 main --programs bright dark \
    --max-healpix 10000 \
    --out $SCRATCH/deepsrch/manifests/dr1_10k.jsonl


# expect 5000–10000 lines depending on DR1 sv3+main coverage
```

### 1.2 Decide: stage or stream from CFS

Phase 10 ran with the GPU at 97–99% SM util reading from CFS at ~200 healpix scale. Reasonable bet: 10k healpix will still saturate the GPU. **Default: stream from CFS.** Only stage if SM util drops below 90% mid-run.

Optional staging (skip unless needed):

```bash
# ~1-2h on a CPU node; ~250 GB SCRATCH footprint estimated
sbatch nersc/stage_to_scratch.slurm
# (edit the slurm to point at dr1_10k.jsonl as src and dr1_10k_scratch.jsonl as dst)
```

### 1.3 Launch the long transformer run

```bash
TOKENIZER_CKPT=$SCRATCH/deepsrch/checkpoints/tokenizer_v1_52693687/best.pt \
MANIFEST=$SCRATCH/deepsrch/manifests/dr1_10k.jsonl \
APPROACH=a \
MAX_HEALPIX=10000 \
STEPS=100000 \
BATCH_SIZE=32 \
LR=2e-4 \
ENCODER_MASK_RATIO=0.50 \
HEALPIX_HOLDOUT_FRAC=0.05 \
AR_EVAL_BATCHES=8 \
RUN_NAME=fm_v1_10k_a \
sbatch -t 36:00:00 nersc/train_transformer.slurm
```

Knob rationale:
- `STEPS=100000` — Phase 10 was still climbing at step 4k. 100k gives a real convergence picture.
- `BATCH_SIZE=32` — saturated GPU at Phase 10 sizes; keep.
- `ENCODER_MASK_RATIO=0.50` — Phase 10 confirmed this works as the honesty floor. Could try 0.30 (easier) or 0.70 (harder) as ablations later.
- `HEALPIX_HOLDOUT_FRAC=0.05` — 5% of 10k = 500 held-out healpix, ~15k–50k truly unseen spectra. Solid val signal.
- `AR_EVAL_BATCHES=8` — doubled from Phase 10's 4 because the val set is larger; AR eval still <10 min per checkpoint.

### 1.4 Monitor during run

```bash
# SM util — confirm GPU is the bottleneck
ssh <node>; nvidia-smi dmon -s u

# val curve via wandb URL printed in $RUN_NAME.out
# specifically watch:
#   val/redshift_acc          — should climb past 73.8% by step ~10k
#   val/masked_spec_acc       — honest metric, should climb past Phase 10 number
#   val_ar/redshift_acc       — only at best-checkpoint events
```

### 1.5 At end of Phase 1 — decide on Phase 2

Read final metrics. Three outcomes:

- **Hit stretch targets.** Done. Lock results and write up.
- **Hit minimum but not stretch.** Try Phase 2 (Tokenizer v2) to see if tokenizer is the ceiling.
- **Did not hit minimum.** Diagnose: is loss plateau or NaN-ing? Check LR schedule, mask ratio, AR eval reliability. Tokenizer v2 won't fix a training-loop bug.

---

## Phase 2 — Detailed steps (Tokenizer v2)

### 2.1 Sketch changes (NOT YET IMPLEMENTED — needs code work)

In `src/tokenizers/spectrum.py`:
- Add top-hat 5-px conv as the first layer before ConvNeXt-V2 stem. Acts as denoiser; keeps spectrum length unchanged.
- Verify codebook utilization in v1 first — if codes are saturated (high utilization), consider bumping codebook size; if many codes are dead, the architecture is fine and more data is the answer.

Code estimate: ~20–40 LOC + tests. Half a day.

### 2.2 Train Tokenizer v2

```bash
RUN_NAME=tokenizer_v2 \
MANIFEST=$SCRATCH/deepsrch/manifests/dr1_10k.jsonl \
STEPS=200000 \
BATCH_SIZE=32 \
LR=3e-4 \
NUM_WORKERS=8 \
sbatch -t 24:00:00 nersc/pretrain_tokenizer.slurm
```

Knob rationale:
- `STEPS=200000` — v1 trained for 100k. Double it because v2 has more data and a new preprocessing layer to learn.
- Reuse the 10k manifest from Phase 1.1 — same data, no extra staging.

### 2.3 Success criteria for Tokenizer v2

- `val/total_loss` < v1's 1.35 by at least 10% (so < 1.21).
- No catastrophic codebook collapse (utilization > 50% of codebook used).

If v2 doesn't beat v1 by ≥10%, ship v1 and skip Phase 3.

---

## Phase 3 — Detailed steps (transformer on Tokenizer v2)

Identical to Phase 1.3 but with the new tokenizer:

```bash
TOKENIZER_CKPT=$SCRATCH/deepsrch/checkpoints/tokenizer_v2/best.pt \
MANIFEST=$SCRATCH/deepsrch/manifests/dr1_10k.jsonl \
APPROACH=a MAX_HEALPIX=10000 STEPS=100000 \
BATCH_SIZE=32 LR=2e-4 \
ENCODER_MASK_RATIO=0.50 HEALPIX_HOLDOUT_FRAC=0.05 \
RUN_NAME=fm_v2_10k_a \
sbatch -t 36:00:00 nersc/train_transformer.slurm
```

---

## Scaling option — 4-GPU DDP (1 full Perlmutter node)

This is orthogonal to the Phase 1/2/3 progression — any of those phases can run on 1 GPU (shared QOS) or 4 GPUs (regular QOS, full node, DDP). Decide per phase based on wallclock pressure vs allocation budget.

### Cost / benefit

| metric | 1 A100 (current) | 4 A100 (DDP) |
|---|---|---|
| step rate (Phase 10 evidence) | ~5 step/s, GPU saturated | ~16 step/s expected (~3.2× from comm overhead, not 4×) |
| wallclock for 100k steps | ~36h | ~12h |
| A100-hours for 100k steps | 36 | ~45 (25% more cost for 3× speedup) |
| effective batch size at `BATCH_SIZE=32` | 32 | 128 |
| LR adjustment needed | — | linear scale: `LR=2e-4 → 8e-4` (or sqrt: `4e-4`) |

The 25% allocation premium is the price of finishing in 1/3 the wallclock. Worth it if you're running Phase 1 + Phase 3 back-to-back (saves ~2 days), or if you need to iterate on hyperparameters.

### Code work required (~50 LOC × 2 scripts, half a day total)

The same DDP diff applies to **both** training entrypoints:
- `nersc/train_transformer.py` (transformer pretraining)
- `nersc/pretrain_tokenizer.py` (tokenizer pretraining)

Walkthrough below is written against `train_transformer.py`. Apply the identical pattern to `pretrain_tokenizer.py` — its main loop is structurally the same (single-GPU AMP loop, AdamW + GradScaler, DataLoader, periodic eval, best-checkpoint mirror). The docstring already flags this is the intended next step: *"DDP is intentionally not in this file -- start single-GPU, validate the pipeline, then promote to DDP later."*

**Tokenizer-specific gotcha — codebook utilization under DDP.** The LFQ codebook usage stats logged in `pretrain_tokenizer.py` count code activations across the batch. Under DDP each rank sees a different mini-batch, so the rank-0 count is a partial sample (1/4 of global). Two options:
1. **Simplest**: log rank-0's local view, document it's an underestimate of true utilization (typical 4× scale-up to estimate global).
2. **Correct**: `dist.all_reduce(code_counts, op=SUM)` before logging. ~3 LOC.

For the first DDP tokenizer run, option 1 is fine — it doesn't affect training, only metric reporting.

In `nersc/train_transformer.py::main()`:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# at the top of main(), before any CUDA work
is_distributed = "RANK" in os.environ
if is_distributed:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    rank, world_size, local_rank = 0, 1, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# after model is on device
if is_distributed:
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

# train_loader
train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,        # this is PER-GPU batch
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    ...
)

# at the top of each epoch / when iter restarts
if train_sampler is not None:
    train_sampler.set_epoch(epoch)

# gate logging, wandb, checkpointing on rank 0 only
if rank == 0:
    wlog(wandb_run, {...}, step=step)
    if val_better:
        torch.save(ckpt, ...)        # also save model.module.state_dict() not model.state_dict()

# at the very end
if is_distributed:
    dist.destroy_process_group()
```

Three gotchas:
1. **Checkpoint state dict**: under DDP, save `model.module.state_dict()`, not `model.state_dict()` — otherwise checkpoint loads break when you eval on 1 GPU later.
2. **DistributedSampler.set_epoch(step)** must be called per epoch or shuffling becomes deterministic across epochs.
3. **Validation only on rank 0** is simplest, OR shard val across ranks and `all_reduce` the metrics. Pick rank-0-only for first DDP run.

### SLURM diff — new `*_ddp.slurm` files (one per training script)

Copy `nersc/train_transformer.slurm` → `nersc/train_transformer_ddp.slurm` AND `nersc/pretrain_tokenizer.slurm` → `nersc/pretrain_tokenizer_ddp.slurm`. Both get the same SBATCH directive changes:

```bash
#SBATCH -A deepsrch_g
#SBATCH -C gpu
#SBATCH -q regular           # was: shared
#SBATCH --nodes=1
#SBATCH --ntasks=4           # was: 1   — one task per GPU
#SBATCH --cpus-per-task=32   # 128 cores / 4 GPUs
#SBATCH --gpus=4             # was: 1
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00      # was: 24:00:00 — 3× faster

# launcher: srun spawns 4 Python procs, each binds to 1 GPU via LOCAL_RANK
srun python "$REPO/nersc/train_transformer.py" \
    --manifest "$MANIFEST" \
    --tokenizer-ckpt "$TOKENIZER_CKPT" \
    ...
    --amp
```

SLURM populates `RANK`, `WORLD_SIZE`, `LOCAL_RANK` env vars from `srun -n 4` automatically when the job has `--ntasks=4 --gpus-per-task=1` — PyTorch's `init_process_group` reads them.

### Launching a 4-GPU DDP Phase 1

```bash
TOKENIZER_CKPT=$SCRATCH/deepsrch/checkpoints/tokenizer_v1_52693687/best.pt \
MANIFEST=$SCRATCH/deepsrch/manifests/dr1_10k.jsonl \
APPROACH=a MAX_HEALPIX=10000 STEPS=100000 \
BATCH_SIZE=32 \
LR=8e-4 \
ENCODER_MASK_RATIO=0.50 HEALPIX_HOLDOUT_FRAC=0.05 \
AR_EVAL_BATCHES=8 \
RUN_NAME=fm_v1_10k_a_ddp4 \
sbatch nersc/train_transformer_ddp.slurm
```

Note `BATCH_SIZE=32` is the per-GPU batch — effective batch is 128. `LR=8e-4` is the linear-scaled value (4× the single-GPU `2e-4`). If training diverges in the first ~500 steps, drop to `LR=4e-4` (sqrt scaling) — transformers sometimes need sublinear.

### Launching a 4-GPU DDP Tokenizer v2

```bash
RUN_NAME=tokenizer_v2_ddp4 \
MANIFEST=$SCRATCH/deepsrch/manifests/dr1_10k.jsonl \
STEPS=200000 \
BATCH_SIZE=32 \
LR=1.2e-3 \
NUM_WORKERS=8 \
sbatch nersc/pretrain_tokenizer_ddp.slurm
```

`BATCH_SIZE=32` is per-GPU (global 128). `LR=1.2e-3` is the linear-scaled value (4× the single-GPU `3e-4`). Tokenizer pretraining is more stable than the transformer w.r.t. LR, so linear should be fine; only fall back to sqrt (`6e-4`) if you see loss spikes in the first 1k steps.

### Verifying DDP actually works (before a long run)

Run a 10-min smoke at 4 GPUs first to make sure NCCL + DistributedSampler + checkpoint paths all behave:

```bash
STEPS=200 RUN_NAME=ddp_smoke MAX_HEALPIX=20 \
    sbatch -t 0:30:00 -q debug nersc/train_transformer_ddp.slurm
```

Success criteria for the smoke:
- 4 lines of `[setup] device=cuda:{0,1,2,3}` in the log (one per rank).
- Only rank 0 writes to `metrics.jsonl` / wandb.
- `best.pt` saves once (not 4 times).
- No NCCL timeouts or hangs.

### Multi-node DDP (>4 GPUs) — out of scope for this plan

Going beyond 4 GPUs means `--nodes=N` with NCCL over Slingshot fabric — works on Perlmutter but adds bandwidth-aware tuning (`NCCL_NSOCKS_PERTHREAD`, `NCCL_SOCKET_IFNAME=hsn0`, etc.) that's not worth debugging for a final project. Stick to 1 node × 4 GPUs.

---

## Beyond the baseline plan — what abundant compute unlocks

Since allocation is not a constraint, Phases 1/2/3 are the floor, not the ceiling. The following experiments materially raise the chance of "best" rather than "good." Ranked by expected ROI on final accuracy:

### B1 — Scale data past 10k healpix

`build_dr1_index.py` with no `--max-healpix` cap returns all DR1 sv3+main bright/dark healpix. Could be 30k–50k (need to confirm via `wc -l`). At ~30 spectra/healpix avg, that's ~1M training spectra vs ~300k at 10k healpix — a real foundation-model scale.

Implication for Phase 1 / Phase 3: bump `MAX_HEALPIX=10000` to whatever the full DR1 sv3+main yields. Tokenizer v2 (Phase 2) should also retrain on this larger set.

### B2 — Bigger transformer (200M–300M params)

100M is over-parameterized at 10k healpix scale; at full DR1 scale it's likely under-parameterized. Doubling `d_model` from 768 → 1024 and going from 6+6 to 8+8 layers gives ~250M params with reasonable compute scaling on DDP4.

Knobs: `D_MODEL=1024 N_LAYERS=8 N_HEADS=16`. Runs in ~24h on DDP4 at 500k steps.

Decision rule: only attempt B2 after B1 lands, since bigger model on small data overfits.

### B3 — Longer training (500k+ steps)

Phase 10 was still climbing at 4k of a planned 100k. Bigger model + bigger data both want more steps. With DDP4 at ~16 step/s, 500k steps = ~9h wallclock.

### B4 — Mask-ratio sweep

Cheap ablation that produces a writeup figure. Run Phase 1 simultaneously at `ENCODER_MASK_RATIO ∈ {0.15, 0.30, 0.50, 0.70}` as 4 parallel DDP4 jobs. ~12h each, all parallel = 12h wallclock, 4× the allocation. Pick the winner for Phase 3.

### B5 — Multi-survey expansion

Adding sv1 + sv2 roughly doubles available DR1 spectra. Quality variance is higher (sv1/sv2 are commissioning surveys) so this is a hedge, not a default. Try only after B1+B2+B3 land and if results indicate data-hungry behavior.

### Recommended expanded plan (uses ~5× the baseline allocation, ~2 days wallclock)

1. **B1 + Phase 2**: build a full-DR1 manifest, kick off tokenizer v2 on it (parallel)
2. **Phase 1 at full DR1, with B4 mask-ratio sweep (4 parallel jobs)**: pick winning mask ratio
3. **Phase 3 at full DR1, with B2 (bigger model) + B3 (500k steps)** on winning mask + tokenizer v2

Total wallclock: ~30h. Total A100-hours: ~600. The output is a genuinely-best-effort foundation model, with mask-ratio + tokenizer ablations as writeup material.

---

## Time + allocation budget

Two variants — pick per-phase based on wallclock pressure.

### Single-GPU variant (shared QOS)

| phase | wallclock | A100-hours | notes |
|---|---|---|---|
| 1.1 manifest build | ~10 min | 0 | login node OK |
| 1.2 staging (optional) | ~1–2h | 0 | CPU node |
| 1.3 long transformer v1 | 36h | 36 | shared QOS, 1 A100 |
| 2.2 tokenizer v2 | 24h | 24 | shared QOS, 1 A100 |
| 3 long transformer v2 | 36h | 36 | only if v2 ships |
| **total worst case** | **~4 days wall** | **~96** | 1.3 + 2.2 can run in parallel as separate jobs → ~36h |

### 4-GPU DDP variant (regular QOS)

| phase | wallclock | A100-hours | notes |
|---|---|---|---|
| 1.3 long transformer v1 | ~12h | ~45 | regular QOS, full node |
| 2.2 tokenizer v2 | ~8h | ~30 | regular QOS, full node |
| 3 long transformer v2 | ~12h | ~45 | only if v2 ships |
| **total worst case** | **~32h wall** | **~120** | ~25% more A100-hours, ~3× less wallclock |

If allocation is tight: skip Phase 2/3, ship Phase 1 only (~36 A100-hours single-GPU, ~45 DDP).

---

## What goes in the writeup

The A-vs-B finding is already in `RESEARCH_LOG.md` (commit `2944571`). The production-run section should add:

1. **Methodology**: data scale (10k healpix vs 200 in Phase 10), Tokenizer v1 (and v2 if applicable), 100k steps, encoder masking 0.50, healpix-level val split.
2. **Headline table**: TF + AR redshift acc, TF + AR spec acc, val loss curves for Phase 1 (and Phase 3 if run).
3. **Comparison to Phase 10**: same architecture, 50× more data, 25× longer training. Quantify the improvement.
4. **Honest reconstruction examples**: pull 3–5 val spectra, plot true vs AR-generated, show fidelity. (Notebook hook: `notebooks/05_training_visualization.ipynb` already has the plotting code from commit `0514443`.)
5. **Tokenizer ablation** (only if Phase 3 runs): v1-trained transformer vs v2-trained transformer on identical val set.
6. **Limitations**: single-GPU, single-survey-pair (sv3 + main bright/dark), no DDP, no downstream task probes.

---

## Things explicitly NOT in this plan

- **Multi-node DDP (>4 GPUs).** 1 node × 4 GPUs is in scope (see Scaling option section). Beyond that requires NCCL fabric tuning not worth debugging for a final project.
- **Bigger model (>100M params).** Current model is over-parameterized at 10k-healpix scale. Revisit only if Phase 3 with v2 plateaus.
- **Downstream task evaluation** (galaxy type classification, redshift regression head, etc.). Out of scope for the foundation-model accuracy goal; could be a follow-up.
- **More surveys / programs.** sv3 + main bright/dark already covers the high-quality DR1 redshift sample. Adding sv1/sv2 doubles data but introduces quality variance.
- **BERT 80/10/10 mask noise.** Phase 10 confirmed pure MASK works. Marginal complexity not worth it.

---

## Decision points for the user

**Compute is not a constraint, deadline is** — defaults below assume DDP4 + run-everything-in-parallel.

Before launching:
- [ ] Confirm deadline (drives expanded-plan scope: 30h wall for full B1+B2+B3+B4)
- [ ] Implement the ~50 LOC DDP change in `nersc/train_transformer.py` + `nersc/pretrain_tokenizer.py` + new `*_ddp.slurm` files
- [ ] Run 10-min DDP smoke (debug QOS) on each script before launching the long jobs
- [ ] Build the full-DR1 manifest (B1) — confirm size; if >50k healpix, decide whether to cap or stage
- [ ] Implement Tokenizer v2 top-hat preprocessing (~20-40 LOC in `src/tokenizers/spectrum.py`)

Mid-run gates:
- [ ] After mask-ratio sweep (B4) finishes: pick the winning ratio for Phase 3
- [ ] After Tokenizer v2 finishes: confirm it ships before launching Phase 3 against it

After Phase 3 completes:
- [ ] Compare all variants on the headline metric table
- [ ] If results are below stretch targets, consider B5 (multi-survey) or further scale-up
- [ ] Lock results into RESEARCH_LOG.md and start the writeup
