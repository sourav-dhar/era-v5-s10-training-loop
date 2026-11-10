"""Write README.md from the executed run's results/*.json, so every number in the README is the measured one."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
MFU = {p.stem.split("_", 1)[1]: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "results").glob("mfu_*.json"))}
REPO = "https://github.com/sourav-dhar/era-v5-s10-training-loop"
PAGE = "https://sourav-dhar.github.io/era-v5-s10-training-loop/"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


sh, gc, acc, nm, inj = S["shapes"], S["gradcheck"], S["accum"], S["norm"], S["inject"]
t4, a100 = MFU.get("T4"), MFU.get("A100")
fv = acc["final_val"]
nc, cl = inj["poisoned, no clip"], inj["poisoned, clip 1.0"]
best_t4 = max(t4["ladder"], key=lambda r: r["MFU"])
best_a100 = max(a100["ladder"], key=lambda r: r["MFU"]) if a100 else None
ceil_t4 = max(g["TFLOPs"] for g in t4["gemm"][:2])
mm_t4 = t4["gemm"][2:6]
fl = {r["format"]: r for r in S["float"]}
sw, uf = S["swamping"], S["underflow"]
eps = S["eps_sweep"]
MAIN = json.loads((ROOT / "results" / "main_run.json").read_text(encoding="utf-8"))
_g = sorted(MAIN["gnorm"][1150:1250]); NORM_TYP = _g[len(_g) // 2]                     # typical norm around the checkpoint
DROP_X = 1 / S["grad_failures"][0]["ratio"]
_lag = nc["react_step"] - inj["poison_step"]
LAG_TXT = "the same step as the norm" if _lag <= 0 else ("one step after the norm" if _lag == 1 else f"{_lag} steps after the norm")
CLIP_MOVE = ((f"step +{cl['react_step']}" + (", barely" if cl["peak_excess"] < 0.05 else "")) if cl.get("react_step") is not None
             else "never crosses +0.02")
_p99 = S["norm"]["quantiles"]["p99"]
CAP_POS = "right at p99" if abs(_p99 - 1) < 0.1 else ("above p99" if _p99 < 1 else "below p99")

grad_rows = "\n".join(f"| {r['layer']} | `{r['weight']}` | {r['backward']:+.10f} | {r['nudge']:+.10f} | {r['rel_error']:.1e} | **{r['agree_digits']:.1f}** |" for r in gc["rows"])
fail_rows = "\n".join(f"| {r['case']} | {r['backward']:+.8f} | {r['nudge']:+.8f} | **{r['ratio']:.6f}** |" for r in S["grad_failures"])
sweep_rows = "\n".join(f"| {r['short_len']} | {pct(r['token_share'])} | {pct(r['vote'], 0)} | {r['loss_err_pct']:+.2f}% | {r['grad_cos']:.4f} |" for r in acc["sweep"])


def ladder_rows():
    rows = []
    for i, r in enumerate(t4["ladder"]):
        a = a100["ladder"][i] if a100 and i < len(a100["ladder"]) else None
        rows.append(f"| {r['rung']} | {r['tok_s'] / 1e3:,.0f}k | **{pct(r['MFU'])}** | "
                    + (f"{a['tok_s'] / 1e3:,.0f}k | **{pct(a['MFU'])}** |" if a else "– | – |"))
    return "\n".join(rows)


width_rows = "\n".join(
    f"| {r['n_embd']} | {r['params_M']:.1f}M | {pct(r['MFU'])} | "
    + (f"{pct(next((w['MFU'] for w in a100['width'] if w['n_embd'] == r['n_embd']), float('nan')))} |" if a100 else "– |")
    for r in t4["width"])
tot = sum(t4["buckets"].values())
bucket_rows = "\n".join(f"| {k} | {v:.1f} | {pct(v / tot, 0)} |" for k, v in sorted(t4["buckets"].items(), key=lambda kv: -kv[1]))
uf_rows = "\n".join(f"| {r['loss_scale']} | {pct(r['fp16_zeroed'], 3)} | {pct(r['fp16_subnormal'], 2)} | {pct(r['fp16_overflow'], 3)} | {pct(r['bf16_zeroed'], 3)} |" for r in uf)
float_rows = "\n".join(f"| {r['format']} | {r['layout']} | `{r['sign']} {r['exponent']} {r['mantissa']}` | `{r['hex']}` | {r['value']} | {r['rel_error']:+.2e} |" for r in S["float"])
diag = "\n".join(f"- {d}" for d in S.get("mfu_diagnosis", []))

README = f"""<div align="center">

# The Training Loop, Interrogated

**ERA V5 · Session 10** — *"Take a small model and a real loop, and make it tell you the truth about itself."*

A {S['n_params'] / 1e6:.1f}M-parameter GPT written from scratch, trained on TinyShakespeare on a {S['gpu']} (and benchmarked on an A100) via Modal.<br>
Every number below was **measured**, and every claim has a figure behind it.

[**📓 Notebook**]({REPO}/blob/main/training_loop.ipynb) · [**🕹️ Interactive companion**]({PAGE}) · [Results JSON]({REPO}/tree/main/results)

</div>

---

## At a glance

| # | The assignment asked | What the loop told me |
|---|---|---|
| 1 | Print every tensor shape | Every activation, parameter, gradient and optimizer-state shape, with one line per dimension. Activations cost **{sh['act_MiB'] / sh['state_MiB']:.1f}×** the 16 B/weight training state; **{pct(sh['att_share'], 0)}** of them are attention matrices. |
| 2 | Verify one gradient by hand | Verified **one weight in every layer type (10)**: `backward()` and the nudge agree to **≥ {gc['min_digits']:.1f} significant digits** (fp64). Dropout breaks it by **{DROP_X:,.0f}×**, a missing `zero_grad` by exactly **2.000000**, and the check itself caught a silent fp32 downcast in my own loss function. |
| 3 | Break gradient accumulation | Same data, same init: the correct loop equals the big batch to **{acc['big_vs_correct_maxdiff']:.0e}**; the average-of-averages loop finishes at val **{fv['broken']:.4f}** vs **{fv['correct']:.4f}**, and the damage lands on **long-context** tokens. |
| 4 | Find where the grad norm moved first | A poisoned batch: norm **{nc['norm_at_poison'] / NORM_TYP:.0f}×** normal on step t, loss reacts at **t+{nc['react_step'] - inj['poison_step']}**; clipping cuts the damage **{nc['peak_excess'] / max(cl['peak_excess'], 1e-4):.0f}×**. Across {nm['n_spikes']} natural spikes the norm and loss move **together** — and the notebook explains why. |
| 5 | Compute your own MFU | T4: **{pct(t4['ladder'][0]['MFU'])} → {pct(best_t4['MFU'])}** in six steps (A100: **{pct(best_a100['MFU']) if best_a100 else '–'}**). A perfect square matmul on a T4 reaches only {pct(ceil_t4 / t4['peak16'], 0)} of datasheet, and widening the model takes the same code to {pct(t4['width'][-1]['MFU'])}. |
| 6 | 0.1 in fp32 / bf16 / fp8 E4M3 | `0x3DCCCCCD` · `0x3DCD` · `0x1D` — derived by hand, **matched bit-for-bit against hardware**. In bf16, a weight receiving +0.001 a thousand times **never moves**. |

> *Every serious training bug is silent, and the loss curve is not going to be the one that tells you.* This repository found four silent failures: one I planted (accumulation), one the course warned about (the poisoned batch), one I wrote by accident (the loss downcast), and one nobody planted (overfitting, §4).

---

## Setup

| | |
|---|---|
| **Model** | GPT, {S['config']['n_layer']} layers × {S['config']['n_head']} heads × {S['config']['n_embd']} wide, context {S['config']['block_size']}, vocab {S['config']['vocab_size']} (characters) · {S['n_params']:,} parameters · pre-norm, GELU, untied head |
| **Data** | TinyShakespeare, 1.1M characters, 90/10 split |
| **Loop** | AdamW (β = 0.9, 0.99, wd 0.1), warmup 100 → cosine, clip 1.0 from step one, {S['amp_dtype']} autocast + dynamic loss scaling, fused AdamW, PyTorch SDPA |
| **Hardware** | {S['gpu']} (whole notebook) and A100 (MFU section), both on [Modal](https://modal.com) · torch {S['torch']} |
| **Reproduce** | `modal run tools/modal_app.py` · or open the notebook on any CUDA machine · `S10_PROFILE=quick` runs every cell on a CPU in ~3 min |

---

## 1 · Every tensor shape in the step

One real training step (B = 32, T = 256) with the forward pass narrating itself. The full table — every activation in all 6 blocks, every parameter, its gradient and both AdamW buffers — is in the notebook. The dimensions:

| symbol | one line on what it means |
|---|---|
| **B** = 32 | batch: independent sequences processed side by side |
| **T** = 256 | time: positions in a sequence — how many characters of context |
| **C** = 384 | channels: the width of every token's vector in the residual stream |
| **nh** = 6 | attention heads; the C channels split into nh independent groups |
| **hs** = 64 | head size = C / nh |
| **3C** = 1152 | q, k, v produced together by one matmul, then split |
| **4C** = 1536 | the MLP's hidden width (4× expansion) |
| **V** = 65 | vocabulary: one logit per possible next character |
| **( )** | a scalar — the loss, the grad norm |

```
idx                (32, 256)            B T        character ids
tok_emb            (32, 256, 384)       B T C      each id looked up as a C-wide vector
pos_emb            (256, 384)           T C        one vector per position, broadcast over B
h0.attn.qkv        (32, 256, 1152)      B T 3C     q, k, v from one fused matmul
h0.attn.q          (32, 6, 256, 64)     B nh T hs  split into heads
h0.attn.weights    (32, 6, 256, 256)    B nh T T   row t: how much position t attends to each earlier one
h0.attn.out        (32, 256, 384)       B T C      heads merged, projected back
h0.mlp.hidden      (32, 256, 1536)      B T 4C     expanded 4x, after GELU
logits             (32, 256, 65)        B T V      a score for every next character, at every position
logits.view(-1,V)  (8192, 65)           (B·T) V    every position is its own classification problem
loss               ()                   ()         one number
c_fc.weight        (1536, 384)          4C C       ← nn.Linear stores (out, in): it computes x @ W.T
grad / exp_avg / exp_avg_sq                        always exactly the weight's shape
```

**What the shapes cost.** Training state (fp32 weight 4 + grad 4 + Adam 8 bytes) is **{sh['state_MiB']:.0f} MiB**, the lesson's 16 bytes/weight. The activations this one step keeps for backward are **{sh['act_MiB']:,.0f} MiB — {sh['act_MiB'] / sh['state_MiB']:.1f}× the state**, and **{pct(sh['att_share'], 0)}** of that is the `B·nh·T·T` attention matrix, which grows with T². That is the memory the fused attention kernel in §5 never materialises.

---

## 2 · Verifying gradients by hand

Warm-up on the lesson's chain (`w1=3, w2=4, x=2, t=20`): `backward()` gives **64.0000000000**, the nudge gives 64.0000000081.

Then the real model, in **float64** with dropout off, central difference `(L(w+h) − L(w−h)) / 2h`, h = 1e-5, nudging in each tensor the element with the *largest* gradient (a zero gradient agrees trivially and proves nothing):

| layer | weight | `backward()` | nudge | rel. error | digits |
|---|---|--:|--:|--:|--:|
{grad_rows}

<p align="center"><img src="figures/02_eps_sweep.png" width="80%"></p>

**How small should the nudge be?** Every precision has a V: too large and curvature pollutes the slope (∝ h²), too small and `L(w+h) − L(w−h)` cancels catastrophically (∝ ε/h). fp64 bottoms out at **{eps['fp64']['best_digits']:.1f} digits** (h ≈ {eps['fp64']['best_h']:.0e}), fp32 at **{eps['fp32']['best_digits']:.1f}**, bf16 at **{eps['bf16']['best_digits']:.1f}** — bf16 can't even measure its own gradients.

**"If they do not agree, you have found something worth understanding."**

| planted failure | `backward()` | nudge | ratio |
|---|--:|--:|--:|
{fail_rows}

- **Dropout** draws a new mask every forward, so `L(w+h)` and `L(w−h)` come from different networks. Re-seed and agreement returns exactly. This is the reason the lesson gives for V4 using **dropout = 0 in its reversible models**: a reversible backward recomputes the forward and needs it repeatable.
- **No `zero_grad`**: gradients accumulate, so `.grad` is exactly 2× the truth. It doesn't crash, and the loss still falls.
- **The one I didn't plant.** My first gradient check reported only ~6 agreeing digits in "float64". The cause: `token_loss` called `logits.float()`, silently downcasting the float64 model to fp32 inside the loss. The gradient check found a precision bug in the code doing the checking. Fixed: upcast-only.

---

## 3 · Breaking gradient accumulation on purpose

Each optimizer step gathers 4 micro-batches whose sequences have **different lengths** (16 – 256 tokens). Three loops see byte-identical data, initial weights and RNG, all in fp32:

- **big batch**: everything padded into one batch, with `ignore_index` masking — the ground truth;
- **correct**: each micro-batch's `sum(token losses) / tokens in the whole step`;
- **broken**: `mean(token losses) / K` — the average of the averages, the bug that lived in major frameworks until 2024.

<p align="center"><img src="figures/03_accumulation_curves.png" width="100%"></p>

- **Correct = big batch.** Max gradient difference **{acc['big_vs_correct_maxdiff']:.1e}**, cosine 1.00000000, identical validation curves (both end at **{fv['correct']:.4f}**).
- **Broken is a different model.** Its gradient points elsewhere (cosine **{acc['broken_cos']:.3f}** on a typical step) and it ends at **{fv['broken']:.4f}**, worse by **{fv['broken'] - fv['correct']:+.3f}** nats on the same data and compute.
- **Who pays.** A short micro-batch holds a few % of the tokens but gets 25% of the vote, so the optimizer over-serves *short-context* prediction. Per position, the broken model is worse by {acc['per_pos_delta_early']:+.3f} on the first 16 positions but by **{acc['per_pos_delta_late']:+.3f}** on the second half of the context: long-range modelling is what it gave up.
- **Its own curve looked fine.** The broken loop's logged loss is off its own true token-weighted loss by {acc['report_gap']:+.3f}. Nothing in the curve says anything is wrong.

<p align="center"><img src="figures/03_how_it_hid.png" width="85%"></p>

**How it hid.** Three micro-batches at 256 tokens and one at length ℓ, from trained weights:

| short micro-batch ℓ | its token share | its vote | logged-loss error | grad cosine to correct |
|--:|--:|--:|--:|--:|
{sweep_rows}

At ℓ = 256 the error is **exactly zero** and the cosine exactly 1. Testing on equal-length batches, which is how most code gets tested, can never reveal the bug.

---

## 4 · The grad norm, at every step

The main run logs, every step: training loss, **pre-clip** grad norm (measured *after* un-scaling — an fp16 scaled gradient reports the loss scale, not the gradient), the norm of every layer group, the loss scale, tokens/s, and a **probe loss**: the current weights scored on one fixed validation batch *before* each update.

<p align="center"><img src="figures/04_gradnorm_vs_loss.png" width="100%"></p>

**A silent failure nobody planted.** The training loss falls all the way to {nm['final_train']:.2f}. The probe bottoms out at **{nm['best_probe']:.3f} around step {nm['best_step']}** and climbs back to **{nm['final_probe']:.3f}**: the model starts memorising Shakespeare instead of learning to write it, and the grad norm rises with it. A dashboard showing only the training loss would ship the worst checkpoint of the run.

### One step where the norm moved first

<p align="center"><img src="figures/04_poisoned_batch.png" width="90%"></p>

From a checkpoint at step 1200, three continuations see the same clean data; in two, steps +{inj['poison_step']} and +{inj['poison_step'] + 1} are replaced by noise.

| run | grad norm at the poisoned step | probe loss first moves | peak extra loss |
|---|--:|--:|--:|
| poisoned, **no clip** | **{nc['norm_at_poison']:.2f}** (typical ≈ {NORM_TYP:.2f}) | step +{nc['react_step']} — **{LAG_TXT}** | **{nc['peak_excess']:+.3f}** |
| poisoned, **clip 1.0** | {cl['norm_at_poison']:.2f} → clipped to 1.0 | {CLIP_MOVE} | {cl['peak_excess']:+.3f} |

The norm reports the bad data on the step it arrives, because it is computed from that batch *before* the update. The loss can only move once the update has landed.

### …and when it doesn't

<p align="center"><img src="figures/04_spike_triggered.png" width="75%"></p>

Averaging the probe loss around all **{nm['n_spikes']} natural grad-norm spikes** in the main run (against nearby ordinary steps) shows a real loss bump — but it is already there **at k = {nm['sta_peak_k']:+d}** ({nm['sta_peak']:+.4f} ± {nm['sta_peak_sem']:.4f}), *with* the norm, not after it. At step k the probe and the norm are measured on the **same weights**, so when the cause (a slightly bad previous update) is already in the weights, both instruments see it at once. The precise version of the lesson's claim:

> **The grad norm moves *before* the loss when trouble arrives with the data, and *with* the loss when it is already in the weights. It is never later, and it is much louder.**

<p align="center"><img src="figures/04_layer_heatmap_and_clip.png" width="100%"></p>

**Choosing the clip from data, not habit** (one of the lesson's open questions for V5). After warmup the norm's p50 / p90 / p99 are **{nm['quantiles']['p50']:.2f} / {nm['quantiles']['p90']:.2f} / {nm['quantiles']['p99']:.2f}**. A cap of 1.0 sits {CAP_POS} — it clipped {pct(nm['clipped_frac'])} of steps, mostly in warmup and the overfitting tail. A cap at 1.5 × p99 ≈ **{1.5 * nm['quantiles']['p99']:.2f}** would leave normal training untouched while still sitting far below a poisoned step's {nc['norm_at_poison']:.1f}. The heatmap shows where the gradient lives: the embeddings, layer-0 attention/MLP and the output head carry the most, and LayerNorm gains the least.

---

## 5 · MFU, measured honestly

`MFU = 6 · N · tokens/s ÷ dense tensor-core peak`. N counts the matmul weights; I also report the PaLM form `6N + 12·L·T·C`, because at T = 256 attention is a real share of the work that `6N` ignores. Peak is the datasheet's dense fp16/bf16 tensor-core number: the machine we're paying for.

<p align="center"><img src="figures/05_mfu_T4.png" width="100%"></p>

| rung (each adds one change) | T4 tok/s | T4 MFU | A100 tok/s | A100 MFU |
|---|--:|--:|--:|--:|
{ladder_rows()}

<p align="center"><img src="figures/05_mfu_cross_gpu.png" width="85%"></p>

### What is costing the distance to 40%

{diag}

**Where the milliseconds go** (T4, best eager rung; wall time measured *without* the profiler attached, which would otherwise slow the Python side ~4× and fake a large idle share — a mistake I made on the first run):

| bucket | ms / step | share |
|---|--:|--:|
{bucket_rows}

**Model size is the biggest lever** (same code, 6 layers, eager):

| n_embd | params | T4 MFU | A100 MFU |
|--:|--:|--:|--:|
{width_rows}

**Honest bottom line.** {pct(best_t4['MFU'])} on the T4 is about **{pct(best_t4['tflops_6N'] / ceil_t4, 0)} of what this card sustains on a perfect matmul**. The remaining gap is, in order: a model too narrow to fill the tensor cores (C = 384 matmuls run at {min(g['TFLOPs'] for g in mm_t4):.0f}–{max(g['TFLOPs'] for g in mm_t4):.0f} TFLOP/s), memory-bound non-matmul kernels (LayerNorm, GELU, softmax, autocast's dtype casts, the optimizer) taking most of the step, and a GPU generation without bf16 or FlashAttention-2. Launch overhead is *not* it on the T4 (the GPU is busy {pct(1 - t4['idle_frac'], 0)} of the time), but it is on the much faster A100 ({pct(a100['idle_frac'], 0) if a100 else '–'} idle), which is exactly where `torch.compile` and CUDA graphs pay off.

---

## 6 · The number 0.1, by hand

**Step 1 — binary.** Doubling the fraction and taking the integer part each time: `0.1 → 0.2 → 0.4 → 0.8 → 1.6 → 1.2 → 0.4 → …` gives `0.000110011001100…₂`, so

$$0.1 = 1.\\underbrace{{1001\\,1001\\,1001\\ldots}}_{{\\text{{mantissa, never ends}}}} \\times 2^{{-4}}$$

Every format stores **sign 0**, **true exponent −4**, and as much of `1001 1001…` as fits, **rounded to nearest-even**. The digit after every cut is `1` followed by non-zero bits, so every format rounds **up**.

| | fp32 (1·8·23, bias 127) | bf16 (1·8·7, bias 127) | fp8 E4M3 (1·4·3, bias 7) |
|---|---|---|---|
| exponent field | −4 + 127 = 123 = `01111011` | −4 + 127 = 123 = `01111011` | −4 + 7 = 3 = `0011` |
| mantissa, cut | `10011001100110011001100│1100…` | `1001100│1100…` | `100│1100…` |
| rounded | `10011001100110011001101` | `1001101` (= 77) | `101` (= 5) |
| **bits** | **`0 01111011 10011001100110011001101`** | **`0 01111011 1001101`** | **`0 0011 101`** |
| hex | `0x3DCCCCCD` | `0x3DCD` | `0x1D` |
| value | 13421773 / 2²⁷ = **0.100000001490116…** | 205 / 2048 = **0.10009765625** | 13 / 128 = **0.1015625** |
| error | +1.5 × 10⁻⁸ relative | +9.8 × 10⁻⁴ (0.098%) | +1.6 × 10⁻² (**1.56%**) |

<p align="center"><img src="figures/06_bits_of_0.1.png" width="90%"></p>

**Checked against hardware** (`struct.pack('>f')`, `torch.bfloat16`, `torch.float8_e4m3fn`): all three **match bit-for-bit**. The same exact-rational encoder covers fp16 and E5M2 too:

| format | layout | bits | hex | stored value | rel. error |
|---|---|---|---|---|--:|
{float_rows}

<p align="center"><img src="figures/06_number_line.png" width="90%"></p>

Near 0.1, E4M3 can say **0.09375, 0.1015625, 0.109375** and nothing in between.

### Why it matters: update swamping, and the fp32 master copy

<p align="center"><img src="figures/06_swamping.png" width="100%"></p>

- A weight at 1.0 receives +0.001 a thousand times. fp32 → **{sw['fp32']:.4f}**. **bf16 → {sw['bf16 (round-to-nearest)']:.4f}**: the gap above 1.0 is 2⁻⁷ ≈ 0.0078, so each update is under half a gap and rounds straight back. Every single one. This is why the lesson's 16 bytes/weight include a **4-byte fp32 master copy**.
- **Stochastic rounding** fixes it *on average*: 20 runs give **{sw['sr_mean_20']:.3f} ± {sw['sr_std_20']:.3f}**. That's how low-precision recipes get by without a master copy.
- Summing 0.1 two thousand times: bf16 **stalls at {sw['sum_bf16']:g}**, E4M3 **at {sw['sum_fp8 E4M3']:g}**, and fp16 **overshoots to {sw['sum_fp16']:.2f}**, because between 64 and 128 every `+0.1` rounds *up* to `+0.125`. Rounding error is not noise: it can stall or bias.

### fp16 underflow, on this model's real gradients

<p align="center"><img src="figures/06_fp16_underflow.png" width="90%"></p>

| loss scale | fp16: becomes 0 | fp16: subnormal (losing digits) | fp16: overflow | bf16: becomes 0 |
|---|--:|--:|--:|--:|
{uf_rows}

At scale 1, **{pct(uf[0]['fp16_subnormal'], 1)} of real gradient elements** land in fp16's subnormal range and {pct(uf[0]['fp16_zeroed'], 3)} vanish; ×1024 rescues nearly all of them, and bf16 loses none at any scale.

### Which one would I train in?

**bf16 for compute, fp32 for master weights and optimizer state, and fp8 E4M3 for the big matmuls only on hardware that does it natively, with per-block scales.**

- **Not fp32 compute.** Its 7.2 digits are drowned by gradient noise anyway, at 2× the memory traffic and a fraction of tensor-core speed (§5: fp32 is the bottom rung of the ladder).
- **bf16** keeps fp32's exponent, so the histogram above has nothing to fear and there's no loss scale to tune. Its 2.4 digits are fine for activations and gradients, but **not for accumulating updates into weights**, which is what the swamping plot shows. Hence the fp32 master copy.
- **fp8 E4M3** misses 0.1 by 1.56% and has nothing between 0.094 and 0.109. That's acceptable for matmul *inputs* when every block carries its own scale (the 2026 production recipe), and unacceptable for anything that accumulates: weights, optimizer moments, softmax, the loss, LayerNorm reductions.
- **On this T4 specifically:** fp16 + dynamic loss scaling, because Turing has no bf16 tensor cores. The underflow table and the GradScaler trace (notebook) show what that costs.

---

## What this run says about the lesson's open questions for V5

| question | evidence from this repo |
|---|---|
| **What precision?** | bf16 compute + fp32 master (§6 swamping, underflow). fp8 only for matmul inputs with block scales. |
| **What clip threshold?** | Pick it from the norm distribution after warmup: here p99 ≈ {nm['quantiles']['p99']:.2f}, so a cap of ~1.5 × p99 leaves normal steps alone and still catches a poisoned batch. Log the norm from step one. |
| **What MFU to accept?** | Decide it against the card's *practical* ceiling, not the datasheet: on a T4 a perfect matmul only reaches {pct(ceil_t4 / t4['peak16'], 0)}. Set the number before the run. |
| **Normalise the loss how?** | By token, never by micro-batch (§3): the alternative silently trades long-context quality for short. |

---

## Repository

```
training_loop.ipynb        the executed notebook (T4 run): every section above, with outputs
figures/                   every figure in this README
results/                   summary.json, main_run.json (per-step logs), mfu_T4.json, mfu_A100.json
docs/index.html            the interactive companion page (GitHub Pages)
tools/notebook_cells.md    notebook source (cells as markdown/code blocks; tags pick sections)
tools/build_notebook.py    builds training_loop.ipynb from the source
tools/run_notebook.py      headless execution (profiles: quick / full / mfu)
tools/modal_app.py         runs it on Modal T4 + A100 and pulls the results back
tools/make_readme.py       writes this README from results/*.json
```
"""

(ROOT / "README.md").write_text(README, encoding="utf-8")
print("README.md written,", len(README.splitlines()), "lines")
