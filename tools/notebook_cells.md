%%md
# The Training Loop, Interrogated

**ERA V5 · Session 10** — *"Take a small model and a real loop, and make it tell you the truth about itself."*

A ~10.7M-parameter character-level GPT, written from scratch below, trained on TinyShakespeare. Every claim in this notebook is **printed or plotted from a measurement**, not asserted.

| # | The assignment asks | Where | What this notebook adds |
|---|---|---|---|
| 1 | Print every tensor shape in the step, one line per dimension | [§1](#1) | activations, params (and the `nn.Linear` transpose trap), grads, optimizer state, and the bytes each one costs |
| 2 | Verify one gradient by hand | [§2](#2) | one weight in **every kind of layer**, an ε-sweep showing where fp64 / fp32 / bf16 stop agreeing, and three planted failures |
| 3 | Break gradient accumulation on purpose | [§3](#3) | three loops on identical data (big batch / correct / broken), *how the bug hid*, and **which tokens it actually hurt** |
| 4 | Log the grad norm, find where it moved first | [§4](#4) | automatic lead-detection, per-layer norm heatmap, a poisoned-batch experiment with and without clipping, a clip threshold chosen from data |
| 5 | Compute your own MFU, honestly | [§5](#5) | an optimisation ladder, a GEMM ceiling, a profiler breakdown, MFU vs model width, and a T4-vs-A100 comparison |
| 6 | 0.1 in fp32, bf16, fp8 E4M3 by hand | [§6](#6) | derivation checked bit-for-bit against hardware, the number line around 0.1, update swamping, stochastic rounding, fp16 underflow on *real* gradients |

> *Every serious training bug is silent, and the loss curve is not going to be the one that tells you.* — the lesson, §17

%%md
## 0 · Setup

`S10_PROFILE` picks the scale: `full` (the GPU run reported here), `quick` (a CPU smoke test of every cell), `mfu` (only the MFU section, used for the A100 comparison).

%%code [core]
import os, math, time, json, copy, struct, random, platform, urllib.request, re
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib as mpl, matplotlib.pyplot as plt
from IPython.display import display, Markdown

PROFILE = os.environ.get('S10_PROFILE', 'full')
QUICK = PROFILE == 'quick'
FIG, RES, DATA = Path('figures'), Path('results'), Path('data')
for d in (FIG, RES, DATA): d.mkdir(exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False      # fp32 means fp32 in this notebook
torch.backends.cudnn.allow_tf32 = False

def seed_all(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def sync():
    if device == 'cuda': torch.cuda.synchronize()

GPU_NAME = torch.cuda.get_device_name(0) if device == 'cuda' else (platform.processor() or 'CPU')
CC = torch.cuda.get_device_capability(0) if device == 'cuda' else (0, 0)
# Mixed precision: bf16 needs Ampere (sm_80)+. A T4 is Turing (sm_75), so it gets fp16 + loss scaling.
AMP_DTYPE = torch.bfloat16 if CC[0] >= 8 else torch.float16
USE_AMP = device == 'cuda'

def amp_ctx(enabled=True):
    return torch.autocast(device_type='cuda', dtype=AMP_DTYPE) if (enabled and USE_AMP) else nullcontext()

def make_scaler(enabled=True):
    return torch.amp.GradScaler('cuda', enabled=bool(enabled and USE_AMP and AMP_DTYPE == torch.float16))

RESULTS = {'profile': PROFILE, 'gpu': GPU_NAME, 'torch': torch.__version__,
           'amp_dtype': str(AMP_DTYPE).replace('torch.', '') if USE_AMP else 'none'}
print(f'profile={PROFILE} · device={device} · {GPU_NAME} · sm_{CC[0]}{CC[1]} · torch {torch.__version__}')
print(f'mixed precision dtype: {RESULTS["amp_dtype"]}')

%%code [core]
# ── One visual language for every figure (validated categorical palette, recessive chrome) ──
PAL = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100',
           magenta='#e87ba4', green='#008300', violet='#6250d6', red='#e34948')
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e7e6e2', '#fcfcfb'
mpl.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'axes.labelcolor': INK2, 'axes.labelsize': 9.5, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'text.color': INK, 'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.titlepad': 24, 'legend.frameon': False, 'legend.fontsize': 8.5,
    'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round', 'figure.dpi': 110,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'axes.prop_cycle': mpl.cycler(color=[PAL[k] for k in ['blue', 'orange', 'aqua', 'yellow', 'magenta', 'green', 'violet', 'red']]),
})

def title(ax, head, sub=None):
    ax.set_title(head)
    if sub: ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.8, color=INK2, va='bottom')

def save(fig, name):
    fig.savefig(FIG / f'{name}.png'); plt.show()

def ema(x, a=0.9):
    out, m = [], None
    for v in x:
        m = v if m is None else a * m + (1 - a) * v
        out.append(m)
    return np.array(out)

%%code [core]
# ── Scale of every experiment, per profile ──
PROFILES = {
    'quick': dict(model=dict(n_layer=2, n_head=2, n_embd=64, block_size=64), B=16, main_steps=150, warmup=20,
                  acc=dict(K=4, Bm=4, steps=80, lens=[4, 8, 16, 32, 64], eval_every=20),
                  inj=dict(start=100, steps=40, at=12, n=2),
                  mfu=dict(B=8, steps=3, warm=1, widths=[64, 128])),
    'full':  dict(model=dict(n_layer=6, n_head=6, n_embd=384, block_size=256), B=32, main_steps=2500, warmup=100,
                  acc=dict(K=4, Bm=8, steps=600, lens=[16, 32, 64, 128, 256], eval_every=25),
                  inj=dict(start=1200, steps=150, at=40, n=2),
                  mfu=dict(B=32, steps=25, warm=8, widths=[128, 256, 384, 512, 768, 1024])),
}
PROFILES['mfu'] = PROFILES['full']
P = PROFILES[PROFILE]
T = P['model']['block_size']
print(json.dumps(P, indent=1))

%%md
### Data: TinyShakespeare, one token per character

%%code [core]
path = DATA / 'tinyshakespeare.txt'
if not path.exists():
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', path)
text = path.read_text(encoding='utf-8')
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_split = int(0.9 * len(data)); train_data, val_data = data[:n_split], data[n_split:]

def get_batch(split, B, T, gen=None):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - T - 1, (B,), generator=gen)
    x = torch.stack([d[i:i + T] for i in ix]); y = torch.stack([d[i + 1:i + 1 + T] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

print(f'{len(text):,} characters · vocab V = {V} · train {len(train_data):,} / val {len(val_data):,} tokens')
print('sample:', repr(text[:80]))

%%md
### The model: a GPT in ~70 lines

Pre-norm transformer blocks (`x + attn(ln(x))`, `x + mlp(ln(x))`), learned positions, untied output head, GELU MLP. Two attention paths — `naive` materialises the full `T×T` attention matrix (so we can *see* it); `sdpa` calls PyTorch's fused kernel (FlashAttention / memory-efficient), which never stores it.

Every `rec(...)` call is a no-op during training; when we pass a list as `tr`, the forward pass narrates itself.

%%code [core]
@dataclass
class GPTConfig:
    vocab_size: int = 65; block_size: int = 256; n_layer: int = 6; n_head: int = 6; n_embd: int = 384
    dropout: float = 0.0; attn: str = 'sdpa'

def rec(tr, name, t, dims, what):
    if tr is not None:
        tr.append(dict(tensor=name, shape=tuple(t.shape), dims=dims, dtype=str(t.dtype).replace('torch.', ''),
                       MiB=t.numel() * t.element_size() / 2**20, what=what))

class Attention(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c_attn, s.c_proj = nn.Linear(c.n_embd, 3 * c.n_embd), nn.Linear(c.n_embd, c.n_embd)
        s.nh, s.impl, s.p = c.n_head, c.attn, c.dropout
        s.drop = nn.Dropout(c.dropout)
        s.register_buffer('mask', torch.tril(torch.ones(c.block_size, c.block_size, dtype=torch.bool)), persistent=False)

    def forward(s, x, tr=None, L=''):
        B, T, C = x.shape; nh, hs = s.nh, C // s.nh
        qkv = s.c_attn(x); rec(tr, f'{L}.attn.qkv', qkv, 'B T 3C', 'query, key, value for every position, from one fused matmul')
        q, k, v = (t.view(B, T, nh, hs).transpose(1, 2) for t in qkv.split(C, dim=2))
        rec(tr, f'{L}.attn.q (k, v same)', q, 'B nh T hs', 'split into heads: each head sees a hs-wide slice of the channels')
        if s.impl == 'naive':
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            att = att.masked_fill(~s.mask[:T, :T], float('-inf')).softmax(dim=-1)
            rec(tr, f'{L}.attn.weights', att, 'B nh T T', 'row t: how much position t attends to each earlier position')
            y = F.dropout(att, s.p, s.training) @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=s.p if s.training else 0.0)
        rec(tr, f'{L}.attn.y (per head)', y, 'B nh T hs', 'value vectors mixed by the attention weights')
        y = s.drop(s.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))
        rec(tr, f'{L}.attn.out', y, 'B T C', 'heads merged and projected back to the residual width')
        return y

class MLP(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c_fc, s.c_proj, s.drop = nn.Linear(c.n_embd, 4 * c.n_embd), nn.Linear(4 * c.n_embd, c.n_embd), nn.Dropout(c.dropout)

    def forward(s, x, tr=None, L=''):
        h = F.gelu(s.c_fc(x)); rec(tr, f'{L}.mlp.hidden', h, 'B T 4C', 'expanded 4x wide, after GELU')
        y = s.drop(s.c_proj(h)); rec(tr, f'{L}.mlp.out', y, 'B T C', 'squeezed back to the residual width')
        return y

class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.ln_1, s.attn, s.ln_2, s.mlp = nn.LayerNorm(c.n_embd), Attention(c), nn.LayerNorm(c.n_embd), MLP(c)

    def forward(s, x, tr=None, L=''):
        x = x + s.attn(s.ln_1(x), tr, L)
        x = x + s.mlp(s.ln_2(x), tr, L); rec(tr, f'{L}.residual', x, 'B T C', 'the residual stream leaving this block')
        return x

class GPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.cfg = c
        s.wte, s.wpe = nn.Embedding(c.vocab_size, c.n_embd), nn.Embedding(c.block_size, c.n_embd)
        s.drop = nn.Dropout(c.dropout)
        s.h = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        s.ln_f, s.lm_head = nn.LayerNorm(c.n_embd), nn.Linear(c.n_embd, c.vocab_size, bias=False)
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
        for n, p in s.named_parameters():          # GPT-2: scale residual projections by 1/sqrt(2L)
            if n.endswith('c_proj.weight'): nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    def forward(s, idx, tr=None):
        B, T = idx.shape
        rec(tr, 'idx (input)', idx, 'B T', 'character ids: B sequences, T positions each')
        tok = s.wte(idx); rec(tr, 'tok_emb = wte(idx)', tok, 'B T C', 'each id looked up as a C-wide vector')
        pos = s.wpe(torch.arange(T, device=idx.device)); rec(tr, 'pos_emb = wpe(0..T-1)', pos, 'T C', 'one vector per position, shared by the whole batch')
        x = s.drop(tok + pos); rec(tr, 'x = tok + pos', x, 'B T C', 'broadcast add: (B,T,C) + (T,C) -> the residual stream')
        for i, b in enumerate(s.h): x = b(x, tr, f'h{i}')
        x = s.ln_f(x); rec(tr, 'ln_f(x)', x, 'B T C', 'final LayerNorm')
        logits = s.lm_head(x); rec(tr, 'logits', logits, 'B T V', 'a score for every possible next character, at every position')
        return logits

def token_loss(logits, y, reduction='mean'):
    """Cross-entropy over valid tokens (target -100 = padding, ignored). Upcasts fp16/bf16 logits to fp32, never
    downcasts: an earlier `.float()` here silently turned the float64 gradient check into an fp32 one."""
    logits = logits if logits.dtype == torch.float64 else logits.float()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction=reduction)

def make_opt(model, lr=1e-3, wd=0.1, fused=False):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    kw = dict(fused=True) if (fused and device == 'cuda') else {}
    return torch.optim.AdamW([{'params': decay, 'weight_decay': wd}, {'params': rest, 'weight_decay': 0.0}],
                             lr=lr, betas=(0.9, 0.99), **kw)

def lr_at(step, total, peak, warm, floor=0.1):
    if step < warm: return peak * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))

CFG = GPTConfig(vocab_size=V, **P['model'])
_m = GPT(CFG); N_PARAMS = sum(p.numel() for p in _m.parameters()); del _m
print(f'{CFG}\nN = {N_PARAMS:,} parameters ({N_PARAMS / 1e6:.2f}M)')
RESULTS['n_params'] = N_PARAMS; RESULTS['config'] = P['model'] | {'vocab_size': V}

%%md
<a id="1"></a>
## 1 · Every tensor shape in one training step

One real step at training size — forward, loss, backward, clip, optimizer update — with the forward pass recording every intermediate tensor. Attention uses the `naive` path here so the `B nh T T` attention matrix is visible (the fused path never builds it — §5 is where that pays off).

**The dimension glossary** (one line per dimension):

| symbol | meaning |
|---|---|
| **B** | batch: how many independent sequences are processed side by side in this micro-batch |
| **T** | time: positions in a sequence, i.e. how many characters of context (≤ `block_size`) |
| **C** | channels: width of the residual stream, the size of every token's vector (`n_embd`) |
| **nh** | number of attention heads; the C channels are split into nh independent groups |
| **hs** | head size = C / nh, the width each head works in |
| **3C** | q, k and v produced together by one matmul, then split |
| **4C** | the MLP's hidden width, 4× expansion |
| **V** | vocabulary: one logit per possible next character |
| **()** | a scalar — the loss and the grad norm are single numbers |

%%code [core,shapes]
seed_all()
model = GPT(replace(CFG, attn='naive')).to(device)
opt = make_opt(model)
x, y = get_batch('train', P['B'], T)

trace = []
logits = model(x, trace)
flat = logits.view(-1, V); rec(trace, 'logits.view(-1, V)', flat, '(B·T) V', 'every position becomes its own classification problem')
rec(trace, 'targets.view(-1)', y.view(-1), '(B·T)', 'the true next character for each of those positions')
loss = token_loss(logits, y); rec(trace, 'loss', loss, '()', 'mean cross-entropy over all B·T predictions: one number')
loss.backward()
gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); rec(trace, 'grad norm', gnorm, '()', 'length of all gradients stacked into one vector')
opt.step()

act = pd.DataFrame(trace)
act['shape'] = act['shape'].map(lambda s: '(' + ', '.join(map(str, s)) + ')')
pd.set_option('display.max_rows', 200, 'display.max_colwidth', 90)
display(act.style.format({'MiB': '{:.2f}'}).hide(axis='index'))

%%md
**Parameters, their gradients, and the optimizer's state.** A gradient always has exactly its weight's shape — *a gradient belongs to one weight*. AdamW then keeps two more tensors of the same shape (`exp_avg`, `exp_avg_sq`). Note the `nn.Linear` trap: weights are stored **(out, in)**, so `c_fc` is `(4C, C)` even though it maps C → 4C (it computes `x @ W.T`).

%%code [core,shapes]
def param_dims(n):
    rules = [('wte.weight', 'V C', 'token embedding table: one row per character'),
             ('wpe.weight', 'T_max C', 'position table: one row per position up to block_size'),
             ('c_attn.weight', '3C C', '(out, in): C in, 3C out for q, k, v'), ('c_attn.bias', '3C', 'bias for q, k, v'),
             ('attn.c_proj.weight', 'C C', '(out, in): merges heads back'), ('attn.c_proj.bias', 'C', ''),
             ('c_fc.weight', '4C C', '(out, in): expands C -> 4C'), ('c_fc.bias', '4C', ''),
             ('mlp.c_proj.weight', 'C 4C', '(out, in): squeezes 4C -> C'), ('mlp.c_proj.bias', 'C', ''),
             ('ln_1.weight', 'C', 'LayerNorm gain'), ('ln_1.bias', 'C', 'LayerNorm shift'),
             ('ln_2.weight', 'C', 'LayerNorm gain'), ('ln_2.bias', 'C', 'LayerNorm shift'),
             ('ln_f.weight', 'C', 'final LayerNorm gain'), ('ln_f.bias', 'C', 'final LayerNorm shift'),
             ('lm_head.weight', 'V C', '(out, in): C -> V scores, no bias')]
    return next(((d, w) for k, d, w in rules if n.endswith(k)), ('?', ''))

rows = []
for n, p in model.named_parameters():
    st = opt.state[p]
    d, w = param_dims(n)
    rows.append(dict(parameter=n, shape=tuple(p.shape), dims=d, grad=tuple(p.grad.shape),
                     exp_avg=tuple(st['exp_avg'].shape), exp_avg_sq=tuple(st['exp_avg_sq'].shape), params=p.numel(), what=w))
ptab = pd.DataFrame(rows)
show = ptab[~ptab.parameter.str.match(r'h\.[1-9]')]          # blocks 1..L-1 repeat block 0 exactly
display(show.style.hide(axis='index'))
print(f'… blocks h.1 – h.{CFG.n_layer - 1} repeat h.0 exactly.  Total: {ptab.params.sum():,} parameters in {len(ptab)} tensors.')
print(f"optimizer step counter: {opt.state[next(model.parameters())]['step']}  (shape {tuple(torch.as_tensor(opt.state[next(model.parameters())]['step']).shape)}, a scalar)")

%%md
**What one step costs in memory.** The lesson's rule is 16 bytes per weight of training state. Our fp32 loop gets there differently (4 weight + 4 grad + 8 Adam, no master copy needed) but lands on the same number. The *activations* the backward pass must keep are a separate bill — and for a small model with long context, they dwarf the weights.

%%code [core,shapes]
state_MiB = 16 * N_PARAMS / 2**20
act_MiB = act[~act.tensor.isin(['loss', 'grad norm'])].MiB.sum()
att_MiB = act[act.tensor.str.contains('attn.weights')].MiB.sum()
mem = pd.DataFrame([
    dict(item='weights (fp32)', MiB=4 * N_PARAMS / 2**20), dict(item='gradients (fp32)', MiB=4 * N_PARAMS / 2**20),
    dict(item='AdamW exp_avg + exp_avg_sq', MiB=8 * N_PARAMS / 2**20),
    dict(item='activations recorded above', MiB=act_MiB), dict(item='  of which: attention matrices (B·nh·T·T)', MiB=att_MiB)])
display(mem.style.format({'MiB': '{:,.1f}'}).hide(axis='index'))
print(f'training state = 16 B x {N_PARAMS:,} = {state_MiB:.0f} MiB;  activations ≈ {act_MiB:.0f} MiB '
      f'({act_MiB / state_MiB:.1f}x the state), {att_MiB / act_MiB:.0%} of it attention matrices.')
RESULTS['shapes'] = dict(state_MiB=state_MiB, act_MiB=float(act_MiB), att_share=float(att_MiB / act_MiB))
del model, opt, logits, loss, trace

%%md
<a id="2"></a>
## 2 · Verify a gradient by hand

**Warm-up — the lesson's chain.** `h = w1·x`, `y = w2·h`, `loss = (y − t)²` with `w1=3, w2=4, x=2, t=20`. The chain rule gives ∂L/∂w1 = 2(y−t)·w2·x = 64.

%%code [core,grad]
w1 = torch.tensor(3.0, dtype=torch.float64, requires_grad=True); w2 = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
f = lambda a, b: ((b * (a * 2.0)) - 20.0) ** 2
L0 = f(w1, w2); L0.backward()
h_ = 1e-6
num = (f(w1.detach() + h_, w2.detach()) - f(w1.detach() - h_, w2.detach())) / (2 * h_)
print(f'loss = {L0.item():.1f}   backward(): dL/dw1 = {w1.grad.item():.10f}   nudge: {num.item():.10f}   '
      f'dL/dw2 = {w2.grad.item():.1f} (= 8 x h = 48)')

%%md
**Now the real model.** Central difference, `(L(w+h) − L(w−h)) / 2h`, against `backward()`. Done in **float64 on CPU** (so rounding cannot hide anything) with dropout off, nudge h = 1e-5 (the sweep below shows why). In each parameter tensor I nudge the element with the *largest* gradient — a weight with zero gradient would "agree" trivially and prove nothing.

%%code [core,grad]
GC_B, GC_T = 4, min(64, T)
seed_all(3)
gc_cfg = replace(CFG, attn='sdpa')
m64 = GPT(gc_cfg).double().eval()
gx, gy = (t.cpu() for t in get_batch('val', GC_B, GC_T))
m64.zero_grad(); L = token_loss(m64(gx), gy); L.backward()
params64 = dict(m64.named_parameters())

@torch.no_grad()
def loss_of(m, x=None, y=None):
    return token_loss(m(gx if x is None else x), gy if y is None else y).item()

def nudge(m, p, idx, h, loss_fn=None):
    loss_fn = loss_fn or (lambda: loss_of(m))
    orig = p.data[idx].item()
    p.data[idx] = orig + h; lp = loss_fn()
    p.data[idx] = orig - h; lm = loss_fn()
    p.data[idx] = orig
    return (lp - lm) / (2 * h)

last = CFG.n_layer - 1
picks = [('wte.weight', 'token embedding'), ('wpe.weight', 'position embedding'),
         ('h.0.ln_1.weight', 'LayerNorm gain'), ('h.0.attn.c_attn.weight', 'q/k/v projection'),
         ('h.0.attn.c_attn.bias', 'q/k/v bias'), (f'h.{last}.attn.c_proj.weight', 'attention output'),
         ('h.0.mlp.c_fc.weight', 'MLP expand'), (f'h.{last}.mlp.c_proj.weight', 'MLP squeeze'),
         ('ln_f.bias', 'final LayerNorm shift'), ('lm_head.weight', 'output head')]
rows = []
for name, kind in picks:
    p = params64[name]; idx = np.unravel_index(int(p.grad.abs().argmax()), p.shape); idx = tuple(int(i) for i in idx)
    ana = p.grad[idx].item(); numg = nudge(m64, p, idx, 1e-5)
    rel = abs(numg - ana) / max(abs(numg), abs(ana), 1e-300)
    rows.append(dict(layer=kind, weight=f'{name}{list(idx)}', backward=ana, nudge=numg, rel_error=rel, agree_digits=-math.log10(max(rel, 1e-17))))
gc = pd.DataFrame(rows)
display(gc.style.format({'backward': '{:+.10f}', 'nudge': '{:+.10f}', 'rel_error': '{:.1e}', 'agree_digits': '{:.1f}'}).hide(axis='index'))
print(f'loss = {L.item():.6f}.  Worst agreement across all {len(gc)} layer types: {gc.agree_digits.min():.1f} significant digits.')
RESULTS['gradcheck'] = dict(min_digits=float(gc.agree_digits.min()), median_digits=float(gc.agree_digits.median()),
                            rows=gc.to_dict('records'))

%%md
**How small should the nudge be?** Too big and the curvature of the loss pollutes the slope (truncation error ∝ h²). Too small and `L(w+h) − L(w−h)` is the difference of two nearly equal numbers, so rounding error ∝ ε/h takes over. Every precision has a "V" with its own best h. In bf16 the bottom of the V is shallow and far to the right: the loss itself only carries ~2–3 digits, so the nudge has to be enormous before the change it causes is even representable.

%%code [core,grad]
name = 'h.0.mlp.c_fc.weight'
p64 = params64[name]; idx = tuple(int(i) for i in np.unravel_index(int(p64.grad.abs().argmax()), p64.shape))
truth = p64.grad[idx].item()
hs_ = np.logspace(-12, 0, 37)
sweep = {}
for label, dt in [('fp64', torch.float64), ('fp32', torch.float32), ('bf16', torch.bfloat16)]:
    m = copy.deepcopy(m64).to(dt); pp = dict(m.named_parameters())[name]
    errs = []
    for h in hs_:
        hq = float(torch.tensor(h, dtype=dt))                  # the nudge must itself be representable
        if hq == 0: errs.append(np.nan); continue
        orig = pp.data[idx].item()
        pp.data[idx] = orig + hq; lp = loss_of(m, gx, gy); hp = pp.data[idx].item() - orig
        pp.data[idx] = orig - hq; lm = loss_of(m, gx, gy); hm = orig - pp.data[idx].item()
        pp.data[idx] = orig
        g = (lp - lm) / (hp + hm) if (hp + hm) != 0 else np.nan   # divide by the nudge actually stored
        errs.append(abs(g - truth) / abs(truth))
    sweep[label] = np.array(errs)

fig, ax = plt.subplots(figsize=(8.6, 4.4))
for (label, errs), c in zip(sweep.items(), [PAL['blue'], PAL['aqua'], PAL['orange']]):
    ax.loglog(hs_, np.clip(errs, 1e-14, 1e3), color=c, label=label, marker='o', ms=3.5)
    j = np.nanargmin(errs); ax.annotate(f'{label}: best h ≈ {hs_[j]:.0e}\n{-np.log10(errs[j]):.1f} digits', (hs_[j], errs[j]),
                                       xytext=(8, -26 if label != 'bf16' else 12), textcoords='offset points', fontsize=8, color=INK2)
ax.set_xlabel('nudge size h'); ax.set_ylabel('relative error vs float64 backward()')
ax.legend(loc='upper right'); ax.invert_xaxis()
title(ax, 'Each precision has its own best nudge', f'{name}{list(idx)}: central difference vs backward(). Right of the minimum: rounding wins. Left: curvature wins.')
save(fig, '02_eps_sweep')
RESULTS['eps_sweep'] = {k: dict(best_h=float(hs_[np.nanargmin(v)]), best_digits=float(-np.log10(np.nanmin(v)))) for k, v in sweep.items()}

%%md
**"If they do not agree, you have found something worth understanding."** Three planted failures, each of which a real loop can commit silently:

1. **Dropout on.** Every forward draws a new random mask, so `L(w+h)` and `L(w−h)` are measured on *different networks*. Reseed before each forward and the agreement returns — this is exactly why V4 set dropout to zero in its reversible models: a reversible backward pass recomputes the forward and needs it to be repeatable.
2. **Forgot `zero_grad()`.** A second `backward()` adds onto the first. The nudge says *g*, `.grad` says *2g* — exactly.
3. **bf16 for the check.** Covered by the sweep above: the measurement instrument is coarser than the thing measured.

%%code [core,grad]
seed_all(3)
md = GPT(replace(gc_cfg, dropout=0.1)).double().train()
md.load_state_dict(m64.state_dict())
pd_ = dict(md.named_parameters())['h.0.mlp.c_fc.weight']
torch.manual_seed(0); md.zero_grad(); token_loss(md(gx), gy).backward(); ana_d = pd_.grad[idx].item()
num_fresh = nudge(md, pd_, idx, 1e-5)                              # new mask every forward
def seeded():
    torch.manual_seed(0); return loss_of(md)
num_seeded = nudge(md, pd_, idx, 1e-5, seeded)                     # same mask as the backward pass

m64.zero_grad(); token_loss(m64(gx), gy).backward(); token_loss(m64(gx), gy).backward()   # forgot zero_grad
twice = params64['h.0.mlp.c_fc.weight'].grad[idx].item()

fail = pd.DataFrame([
    dict(case='dropout 0.1, fresh mask per forward', backward=ana_d, nudge=num_fresh, ratio=ana_d / num_fresh),
    dict(case='dropout 0.1, mask re-seeded (repeatable)', backward=ana_d, nudge=num_seeded, ratio=ana_d / num_seeded),
    dict(case='no zero_grad between two backward()', backward=twice, nudge=truth, ratio=twice / truth)])
display(fail.style.format({'backward': '{:+.8f}', 'nudge': '{:+.8f}', 'ratio': '{:.6f}'}).hide(axis='index'))
RESULTS['grad_failures'] = fail.to_dict('records')
del md, m64

%%md
<a id="3"></a>
## 3 · Breaking gradient accumulation on purpose

Each optimizer step gathers **K micro-batches whose sequences have different lengths** — some micro-batches are long documents, some are the short tail end of one. Three loops see *byte-identical* data, initial weights and RNG:

| loop | per-micro-batch loss passed to `backward()` | what it computes |
|---|---|---|
| **big batch** | — (all K padded into one batch, `ignore_index` masks the padding) | the ground truth |
| **correct** | `sum(token losses) / tokens in the whole step` | every token gets one equal vote |
| **broken** | `mean(token losses) / K` | *the average of the averages* — the pre-2024 framework bug |

Everything runs in fp32 so any difference is the bug, not rounding.

%%code [core,accum]
A = P['acc']; K, Bm = A['K'], A['Bm']

def make_mbs(step, equal=False, lens=None):
    g = torch.Generator().manual_seed(50_000 + step)
    Ts = lens or ([T] * K if equal else [A['lens'][int(i)] for i in torch.randint(len(A['lens']), (K,), generator=g)])
    return [get_batch('train', Bm, t, g) for t in Ts]

def accum_backward(m, mbs, mode):
    n_tok = sum(y.numel() for _, y in mbs); reported = true_sum = 0.0
    if mode == 'big':                                     # pad every micro-batch to T, mask with -100
        xs = torch.cat([F.pad(x, (0, T - x.size(1)), value=0) for x, _ in mbs])
        ys = torch.cat([F.pad(y, (0, T - y.size(1)), value=-100) for _, y in mbs])
        loss = token_loss(m(xs), ys); loss.backward(); return loss.item(), loss.item()
    for x, y in mbs:
        logits = m(x)
        if mode == 'correct': loss = token_loss(logits, y, 'sum') / n_tok
        else:                 loss = token_loss(logits, y, 'mean') / len(mbs)      # the bug
        loss.backward(); reported += loss.item()
        true_sum += token_loss(logits.detach(), y, 'sum').item()
    return reported, true_sum / n_tok

VAL = [get_batch('val', 16, T, torch.Generator().manual_seed(900 + i)) for i in range(4 if QUICK else 8)]
@torch.no_grad()
def eval_val(m, per_pos=False):
    m.eval(); tot = torch.zeros(T, device=device)
    for x, y in VAL:
        tot += F.cross_entropy(m(x).float().transpose(1, 2), y, reduction='none').mean(0)
    m.train(); pp = (tot / len(VAL)).cpu().numpy()
    return pp if per_pos else float(pp.mean())

def accum_train(mode, steps):
    seed_all(11); m = GPT(replace(CFG, attn='sdpa')).to(device); o = make_opt(m)
    log = dict(reported=[], true=[], val_step=[], val=[])
    for s in range(steps):
        for gp in o.param_groups: gp['lr'] = lr_at(s, steps, 1e-3, 50)
        rep, tru = accum_backward(m, make_mbs(s), mode)
        o.step(); o.zero_grad(set_to_none=True)
        log['reported'].append(rep); log['true'].append(tru)
        if s % A['eval_every'] == 0 or s == steps - 1: log['val_step'].append(s); log['val'].append(eval_val(m))
    log['val_per_pos'] = eval_val(m, per_pos=True).tolist()
    return m, log

t0 = time.time(); acc_models, acc_logs = {}, {}
for mode in ['big', 'correct', 'broken']:
    acc_models[mode], acc_logs[mode] = accum_train(mode, A['steps'])
    print(f'{mode:>8}: final val loss {acc_logs[mode]["val"][-1]:.4f}   ({time.time() - t0:.0f}s)')

%%md
**First, prove the correct loop *is* the big batch** — gradient by gradient, at one step, from the same weights. Then sweep how unequal the micro-batches are, to watch the bug appear and vanish.

%%code [core,accum]
def grads_for(mode, mbs, src):
    m = copy.deepcopy(src); m.zero_grad(set_to_none=True)
    rep, tru = accum_backward(m, mbs, mode)
    return torch.cat([p.grad.flatten() for p in m.parameters()]), rep, tru

ref = acc_models['correct']
mbs = make_mbs(10_001)
g_big, _, _ = grads_for('big', mbs, ref); g_cor, rep_c, tru_c = grads_for('correct', mbs, ref); g_brk, rep_b, tru_b = grads_for('broken', mbs, ref)
cos = lambda a, b: F.cosine_similarity(a.double(), b.double(), dim=0).item()
print(f'micro-batch lengths this step: {[x.size(1) for x, _ in mbs]}')
print(f'big vs correct : max |Δgrad| = {(g_big - g_cor).abs().max().item():.2e}   cosine = {cos(g_big, g_cor):.8f}')
print(f'broken vs correct: cosine = {cos(g_brk, g_cor):.6f}   norm ratio = {g_brk.norm().item() / g_cor.norm().item():.4f}')
print(f'loss the loop reports — correct {rep_c:.4f}   broken {rep_b:.4f}   (true token mean {tru_c:.4f}: broken is {100 * (rep_b / tru_c - 1):+.2f}%)')

short = [l for l in [1, 2, 4, 8, 16, 32, 64, 128, 256] if l <= T]
rows = []
for l in short:
    mbs = make_mbs(20_000 + l, lens=[T] * (K - 1) + [l])
    gc_, rc, tc = grads_for('correct', mbs, ref); gb_, rb, _ = grads_for('broken', mbs, ref)
    rows.append(dict(short_len=l, token_share=l / ((K - 1) * T + l), vote=1 / K, loss_err_pct=100 * (rb / tc - 1), grad_cos=cos(gb_, gc_), grad_norm_ratio=(gb_.norm() / gc_.norm()).item()))
hid = pd.DataFrame(rows)
display(hid.style.format({'token_share': '{:.1%}', 'vote': '{:.0%}', 'loss_err_pct': '{:+.2f}%', 'grad_cos': '{:.5f}', 'grad_norm_ratio': '{:.3f}'}).hide(axis='index'))
RESULTS['accum'] = dict(big_vs_correct_maxdiff=(g_big - g_cor).abs().max().item(), broken_cos=cos(g_brk, g_cor),
                        sweep=hid.to_dict('records'),
                        final_val={k: v['val'][-1] for k, v in acc_logs.items()})

%%code [core,accum]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
ax.plot(hid.short_len, hid.loss_err_pct, color=PAL['red'], marker='o', ms=6, label='reported loss error')
ax.axhline(0, color=MUTED, lw=0.8); ax.set_xscale('log', base=2)
ax.set_xlabel(f'length of the one short micro-batch (others are {T})'); ax.set_ylabel('error in the logged loss (%)')
j = hid.index[-1]; ax.annotate('lengths equal:\nerror exactly 0', (hid.short_len[j], hid.loss_err_pct[j]), xytext=(-90, 25), textcoords='offset points', fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED))
title(ax, 'How the bug hid', 'The error only exists when token counts differ.')
ax = axes[1]
ax.plot(hid.short_len, 1 - hid.grad_cos, color=PAL['red'], marker='o', ms=6)
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlabel(f'length of the one short micro-batch'); ax.set_ylabel('1 − cosine(broken grad, correct grad)')
title(ax, 'The gradient points somewhere else', 'Not just mis-scaled: the direction itself changes.')
fig.tight_layout(); save(fig, '03_how_it_hid')

%%code [core,accum]
st = np.arange(A['steps'])
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axes[0]
ax.plot(st, ema(acc_logs['correct']['reported']), color=PAL['blue'], label='correct loop: what it logs')
ax.plot(st, ema(acc_logs['broken']['reported']), color=PAL['red'], label='broken loop: what it logs')
ax.plot(st, ema(acc_logs['broken']['true']), color=PAL['red'], ls=(0, (2, 2)), lw=1.6, label='broken loop: true token-weighted loss')
ax.set_xlabel('optimizer step'); ax.set_ylabel('training loss (EMA)'); ax.legend(loc='upper right')
gap = np.mean(np.array(acc_logs['broken']['reported'][-100:]) - np.array(acc_logs['broken']['true'][-100:]))
title(ax, 'Both curves together', f'broken loop logs {gap:+.3f} vs its own true loss — and still looks fine')
ax = axes[1]
for mode, c, ls in [('big', PAL['aqua'], '-'), ('correct', PAL['blue'], (0, (3, 2))), ('broken', PAL['red'], '-')]:
    ax.plot(acc_logs[mode]['val_step'], acc_logs[mode]['val'], color=c, ls=ls, label=mode, lw=2.2 if mode != 'correct' else 1.8)
ax.set_xlabel('optimizer step'); ax.set_ylabel('validation loss, full context'); ax.legend(loc='upper right')
fv = RESULTS['accum']['final_val']
title(ax, 'Same data, different model', f'final val: big {fv["big"]:.4f} · correct {fv["correct"]:.4f} · broken {fv["broken"]:.4f}')
ax = axes[2]
d = np.array(acc_logs['broken']['val_per_pos']) - np.array(acc_logs['correct']['val_per_pos'])
w = max(1, T // 32); dpos = d[:len(d) // w * w].reshape(-1, w).mean(1); xpos = np.arange(len(dpos)) * w + w / 2
ax.bar(xpos, dpos, width=w * 0.85, color=[PAL['blue'] if v < 0 else PAL['red'] for v in dpos])
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel('position in the sequence (context length)'); ax.set_ylabel('val loss: broken − correct')
title(ax, 'Who the bug favoured', 'val loss gap by position; blue = broken is better there')
fig.tight_layout(); save(fig, '03_accumulation_curves')
RESULTS['accum']['report_gap'] = float(gap)
RESULTS['accum']['per_pos_delta_early'] = float(d[:max(1, T // 16)].mean()); RESULTS['accum']['per_pos_delta_late'] = float(d[T // 2:].mean())
del acc_models

%%md
<a id="4"></a>
## 4 · The grad norm, at every step

The real training run: AdamW, warmup + cosine, clipping at 1.0 from step one, mixed precision. Every step logs the training loss, the **pre-clip** grad norm (after un-scaling — measuring a scaled fp16 gradient would report the loss scale, not the gradient), the norm of every layer group, the loss scale, tokens/s, and a **probe loss**: the loss of the *current* weights on one fixed validation batch, measured *before* the update. The probe removes batch-to-batch noise, so when it moves it is the model that moved.

%%code [core,norm]
def grad_groups(m):
    g = {}
    for n, p in m.named_parameters():
        if n.startswith('wte'): k = 'tok emb'
        elif n.startswith('wpe'): k = 'pos emb'
        elif n.startswith('h.'):
            i, part = n.split('.')[1:3]; k = f'L{i} ' + {'attn': 'attn', 'mlp': 'mlp'}.get(part, 'ln')
        elif n.startswith('ln_f'): k = 'ln_f'
        else: k = 'lm_head'
        g.setdefault(k, []).append(p)
    return g

def train_step(m, o, scaler, x, y, clip, groups=None):
    with amp_ctx(): loss = token_loss(m(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(o)                                       # true gradients from here on
    gn = [torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in ps])) for ps in groups.values()] if groups else []
    total = torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
    scaler.step(o); scaler.update(); o.zero_grad(set_to_none=True)
    return loss.detach(), total, gn

@torch.no_grad()
def probe_loss(m, px, py):
    m.eval()
    with amp_ctx(): l = token_loss(m(px), py)
    m.train(); return l

S, B, CLIP, PEAK_LR = P['main_steps'], P['B'], 1.0, 1e-3
seed_all(42)
model = GPT(replace(CFG, attn='sdpa')).to(device)
opt = make_opt(model, lr=PEAK_LR, fused=True); scaler = make_scaler()
groups = grad_groups(model); gnames = list(groups)
px, py = get_batch('val', 16, T, torch.Generator().manual_seed(4242))
gen = torch.Generator().manual_seed(2024)
log = {k: [] for k in ['loss', 'probe', 'gnorm', 'lr', 'scale', 'dt', 'tok_s']}; glog = []
ckpt = None; t_start = time.time()
for s in range(S):
    lr = lr_at(s, S, PEAK_LR, P['warmup'])
    for gp in opt.param_groups: gp['lr'] = lr
    pl = probe_loss(model, px, py)
    x, y = get_batch('train', B, T, gen)
    sync(); t0 = time.perf_counter()
    loss, total, gn = train_step(model, opt, scaler, x, y, CLIP, groups)
    vals = torch.stack([loss.float(), pl.float(), total.float(), *[g.float() for g in gn]]).cpu().numpy()
    dt = time.perf_counter() - t0
    log['loss'].append(vals[0]); log['probe'].append(vals[1]); log['gnorm'].append(vals[2]); glog.append(vals[3:])
    log['lr'].append(lr); log['scale'].append(scaler.get_scale() if scaler.is_enabled() else 1.0)
    log['dt'].append(dt); log['tok_s'].append(B * T / dt)
    if s == P['inj']['start']:
        ckpt = dict(model=copy.deepcopy(model.state_dict()), opt=copy.deepcopy(opt.state_dict()), scaler=scaler.state_dict(), lr=lr)
    if s % max(1, S // 10) == 0 or s == S - 1:
        print(f'step {s:5d} | loss {vals[0]:.4f} | probe {vals[1]:.4f} | grad norm {vals[2]:7.3f} | lr {lr:.2e} | {B * T / dt:,.0f} tok/s')
G = np.array(glog); L_ = {k: np.array(v) for k, v in log.items()}
print(f'done in {time.time() - t_start:.0f}s · final val loss {eval_val(model):.4f}')
json.dump({k: [float(v) for v in vs] for k, vs in log.items()} | {'groups': gnames, 'group_norms': G.round(5).tolist()}, open(RES / 'main_run.json', 'w'))

%%md
**A step where the norm moved first.** Rather than eyeballing, a detector: compute a robust z-score of each series against its own trailing 50-step window (median and MAD, so one spike cannot hide itself). An *event* is a step where the grad norm is anomalous (z > 4) while the probe loss is still quiet, followed within 25 steps by the probe loss turning anomalous (z > 3).

%%code [core,norm]
def robust_z(x, w=50):
    x = np.asarray(x, float); z = np.full_like(x, np.nan)
    for i in range(w, len(x)):
        win = x[i - w:i]; med = np.median(win); mad = 1.4826 * np.median(np.abs(win - med)) + 1e-12
        z[i] = (x[i] - med) / mad
    return z

def lead_events(gn, loss, w=50, zg=4.0, zl=3.0, horizon=25):
    Zg, Zl = robust_z(np.log(gn), w), robust_z(loss, w); ev, i = [], w
    while i < len(gn):
        if Zg[i] > zg and not Zl[i] > zl:
            j = next((k for k in range(i + 1, min(len(gn), i + horizon)) if Zl[k] > zl), None)
            if j is not None:
                ev.append(dict(norm_step=i, loss_step=j, lead=j - i, norm_z=Zg[i], loss_z=Zl[j])); i = j + 1; continue
        i += 1
    return pd.DataFrame(ev), Zg, Zl

ev, Zg, Zl = lead_events(L_['gnorm'], L_['probe'])
if len(ev):
    display(ev.style.format({'norm_z': '{:.1f}', 'loss_z': '{:.1f}'}).hide(axis='index'))
    best = ev.sort_values('norm_z', ascending=False).iloc[0]
    print(f'Strongest natural event: grad norm jumped at step {int(best.norm_step)} (z={best.norm_z:.1f}); '
          f'the probe loss only reacted at step {int(best.loss_step)}, {int(best.lead)} step(s) later.')
else:
    best = None; print('No natural lead event in this run — the controlled experiment below supplies one.')

# Statistically, over the whole run: a spike-triggered average. Take every natural grad-norm spike (z > 3, first step of
# each burst), look at how the probe loss moves from 5 steps before to 15 after, and subtract how it moves around an
# ordinary step at the same point in training (so the downward trend cancels out).
ks = np.arange(-5, 16); lo_s, hi_s = 2 * P['warmup'], S - ks[-1] - 1
spikes = [i for i in range(lo_s, hi_s) if Zg[i] > 3 and not Zg[i - 1] > 3]
win = lambda i: L_['probe'][i + ks] - L_['probe'][i]
base_w = np.array([win(i) for i in range(lo_s, hi_s)]).mean(0)
sta = np.array([win(i) for i in spikes]).mean(0) - base_w if spikes else None
if sta is not None:
    kpk = int(ks[np.argmax(sta)])
    print(f'{len(spikes)} natural grad-norm spikes. Averaged over all of them, the extra probe loss peaks at k = {kpk:+d} '
          f'step(s) after the spike (+{sta.max():.4f}); before the spike (k<0) it averages {sta[ks < 0].mean():+.4f}.')
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axvline(0, color=PAL['violet'], lw=1.2); ax.text(0, sta.max(), ' grad-norm spike', color=PAL['violet'], fontsize=8.5, va='top')
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(ks, sta, color=PAL['blue'], marker='o', ms=6)
    ax.set_xlabel('steps relative to the grad-norm spike'); ax.set_ylabel('extra probe loss vs an ordinary step')
    title(ax, 'What the loss does after a grad-norm spike', f'average over {len(spikes)} natural spikes in the main run; probe measured before each update')
    save(fig, '04_spike_triggered')
else:
    kpk = None; print('No natural grad-norm spikes (z > 3) after warmup.')
RESULTS['norm'] = dict(events=ev.to_dict('records'), n_spikes=len(spikes), sta_peak_k=kpk, sta_peak=float(sta.max()) if sta is not None else None,
                       sta=dict(k=ks.tolist(), excess=sta.tolist()) if sta is not None else None,
                       final_val=float(eval_val(model)), clipped_frac=float(np.mean(L_['gnorm'] > CLIP)),
                       median_tok_s=float(np.median(L_['tok_s'][10:])))

%%code [core,norm]
st = np.arange(S)
fig, axes = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1], hspace=0.12))
ax = axes[0]
ax.plot(st, L_['gnorm'], color=PAL['violet'], lw=0.9, alpha=0.45)
ax.plot(st, ema(L_['gnorm'], 0.95), color=PAL['violet'], lw=2, label='grad norm (pre-clip)')
ax.axhline(CLIP, color=PAL['green'], lw=1.2); ax.text(S, CLIP, ' clip 1.0', color=PAL['green'], fontsize=8.5, va='center')
ax.set_yscale('log'); ax.set_ylabel('grad norm (log)')
title(ax, 'Grad norm and probe loss, every step', f'{S:,} steps · {np.mean(L_["gnorm"] > CLIP):.1%} of steps clipped · probe = fixed validation batch, measured before each update')
ax = axes[1]
ax.plot(st, L_['loss'], color=PAL['orange'], lw=0.8, alpha=0.35, label='training loss (this batch)')
ax.plot(st, L_['probe'], color=PAL['blue'], lw=1.8, label='probe loss')
ax.set_ylabel('loss'); ax.set_xlabel('step'); ax.legend(loc='upper right')
if best is not None:
    for a in axes:
        a.axvline(best.norm_step, color=PAL['violet'], lw=1); a.axvline(best.loss_step, color=PAL['blue'], lw=1)
    axes[0].annotate(f'norm moves (step {int(best.norm_step)})', (best.norm_step, L_['gnorm'][int(best.norm_step)]), xytext=(10, 10), textcoords='offset points', fontsize=8.5, color=INK2)
    axes[1].annotate(f'loss reacts (step {int(best.loss_step)})', (best.loss_step, L_['probe'][int(best.loss_step)]), xytext=(10, 18), textcoords='offset points', fontsize=8.5, color=INK2)
save(fig, '04_gradnorm_vs_loss')

%%code [core,norm]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axes[0]
nb = 100 if S >= 1000 else 10
Gb = np.log10(G[:len(G) // nb * nb].reshape(-1, nb, G.shape[1]).mean(1).T + 1e-12)
im = ax.imshow(Gb, aspect='auto', cmap=mpl.colors.LinearSegmentedColormap.from_list('b', ['#cde2fb', '#5598e7', '#1c5cab', '#0d366b']),
               extent=[0, S, len(gnames) - 0.5, -0.5], interpolation='nearest')
ax.set_yticks(range(len(gnames))); ax.set_yticklabels(gnames, fontsize=7.5); ax.grid(False); ax.set_xlabel('step')
cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label('log10 grad norm', color=INK2); cb.outline.set_visible(False)
title(ax, 'Which layers carry the gradient', f'mean norm per layer group, {nb}-step buckets')
ax = axes[1]
post = L_['gnorm'][P['warmup']:]
qs = {q: np.quantile(post, q) for q in [0.5, 0.9, 0.99]}
ax.hist(np.log10(post), bins=60, color=PAL['violet'], alpha=0.85, edgecolor=SURF, linewidth=0.6)
for (q, v), c in zip(qs.items(), [MUTED, PAL['yellow'], PAL['red']]):
    ax.axvline(np.log10(v), color=c, lw=1.4); ax.text(np.log10(v), ax.get_ylim()[1] * 0.92, f' p{int(q * 100)}={v:.2f}', color=INK2, fontsize=8)
ax.axvline(0, color=PAL['green'], lw=1.4); ax.set_xlabel('log10 grad norm (after warmup)'); ax.set_ylabel('steps')
title(ax, 'Choosing the clip from data', 'the threshold should sit just above normal life (≈p99), not at habit')
fig.tight_layout(); save(fig, '04_layer_heatmap_and_clip')
RESULTS['norm']['quantiles'] = {f'p{int(q * 100)}': float(v) for q, v in qs.items()}
print('grad-norm quantiles after warmup:', {f'p{int(q * 100)}': round(float(v), 3) for q, v in qs.items()},
      f'→ data-driven cap ≈ {1.5 * qs[0.99]:.2f} (1.5 × p99)')

%%md
### The controlled experiment: one poisoned batch

From a checkpoint at step `inj.start`, three continuations see the same clean data stream at the same learning rate. In two of them, `inj.n` consecutive batches are replaced by **noise** (random characters predicting random characters — a corrupted shard, a bad tokenizer, a padding bug). One of those runs has clipping switched off.

%%code [core,norm]
I = P['inj']
def poisoned(B, T, g):
    return (torch.randint(V, (B, T), generator=g).to(device), torch.randint(V, (B, T), generator=g).to(device))

def continue_run(poison, clip):
    seed_all(7); m = GPT(replace(CFG, attn='sdpa')).to(device); m.load_state_dict(ckpt['model'])
    o = make_opt(m, lr=ckpt['lr'], fused=True); o.load_state_dict(ckpt['opt']); sc = make_scaler(); sc.load_state_dict(ckpt['scaler'])
    for gp in o.param_groups: gp['lr'] = ckpt['lr']
    g, gpz = torch.Generator().manual_seed(77), torch.Generator().manual_seed(78); out = dict(gnorm=[], probe=[])
    for s in range(I['steps']):
        pl = probe_loss(m, px, py)
        x, y = get_batch('train', B, T, g)
        if poison and I['at'] <= s < I['at'] + I['n']: x, y = poisoned(B, T, gpz)
        _, total, _ = train_step(m, o, sc, x, y, clip)
        out['gnorm'].append(total.item()); out['probe'].append(pl.item())
    return {k: np.array(v) for k, v in out.items()}

inj = {'clean': continue_run(False, CLIP), 'poisoned, no clip': continue_run(True, float('inf')), 'poisoned, clip 1.0': continue_run(True, CLIP)}
a = I['at']; base = inj['clean']['probe']
react = {}
for k in ['poisoned, no clip', 'poisoned, clip 1.0']:
    excess = inj[k]['probe'] - base
    j = next((t for t in range(a, I['steps']) if excess[t] > 0.02), None); react[k] = j
    print(f'{k:>18}: grad norm at poison step {inj[k]["gnorm"][a]:8.2f} (clean: {inj["clean"]["gnorm"][a]:.2f}) · '
          f'probe loss first >0.02 above clean at step {j} ({(j - a) if j is not None else "-"} step(s) after the norm) · '
          f'peak excess {excess.max():.3f} · excess at end {excess[-10:].mean():+.4f}')
RESULTS['inject'] = {k: dict(norm_at_poison=float(v['gnorm'][a]), peak_excess=float((v['probe'] - base).max()), end_excess=float((v['probe'] - base)[-10:].mean()), react_step=react.get(k)) for k, v in inj.items() if k != 'clean'}
RESULTS['inject']['poison_step'] = a

%%code [core,norm]
st = np.arange(I['steps'])
fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, gridspec_kw=dict(hspace=0.12))
cols = {'clean': MUTED, 'poisoned, no clip': PAL['red'], 'poisoned, clip 1.0': PAL['green']}
for k, v in inj.items():
    axes[0].plot(st, v['gnorm'], color=cols[k], lw=1.8 if k != 'clean' else 1.4, label=k)
    axes[1].plot(st, v['probe'], color=cols[k], lw=1.8 if k != 'clean' else 1.4, label=k)
axes[0].set_yscale('log'); axes[0].axhline(CLIP, color=PAL['green'], lw=0.8); axes[0].set_ylabel('grad norm (log)')
axes[1].set_ylabel('probe loss'); axes[1].set_xlabel(f'steps after checkpoint (step {I["start"]})'); axes[1].legend(loc='upper right')
for ax in axes: ax.axvspan(a - 0.5, a + I['n'] - 0.5, color=PAL['red'], alpha=0.07, lw=0)
r = react['poisoned, no clip']
if r is not None:
    axes[1].annotate(f'loss reacts at +{r - a}', (r, inj['poisoned, no clip']['probe'][r]), xytext=(14, 10), textcoords='offset points', fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED))
title(axes[0], 'A poisoned batch: the norm screams first', f'{I["n"]} noise batches at step +{a} (shaded). The norm jumps on the step itself; the loss can only respond after the update lands.')
save(fig, '04_poisoned_batch')

%%md
<a id="5"></a>
## 5 · MFU, measured honestly

$$\text{MFU} = \frac{6N \times \text{tokens/s}}{\text{peak FLOP/s}}$$

Two honesty choices, stated up front:
* **N** counts the weights that do matmuls (embedding *lookups* are not FLOPs; for this model that only removes 0.1 M). I also report the PaLM variant that adds the attention matmuls, `6N + 12·L·T·C` per token, because at T=256 and C=384 attention is a real share of the work that `6N` ignores.
* **Peak** is the datasheet's dense **tensor-core** number for the GPU — the machine we're paying for. For fp32 rungs I also show MFU against the fp32 CUDA-core peak, which flatters them.

%%code [core,mfu]
PEAKS = [('H100', (67.0, 989.0)), ('A100', (19.5, 312.0)), ('L40S', (91.6, 362.0)), ('L4', (30.3, 121.0)),
         ('A10', (31.2, 125.0)), ('T4', (8.1, 65.0))]   # (fp32 CUDA-core, fp16/bf16 dense tensor-core) TFLOP/s
GPU_TAG, (PEAK32, PEAK16) = next(((k, v) for k, v in PEAKS if k in GPU_NAME.upper()), ('CPU', (None, None)))
if GPU_TAG == 'H100' and 'PCIE' in GPU_NAME.upper(): PEAK32, PEAK16 = 51.0, 756.0
print(f'{GPU_NAME}: peak fp32 {PEAK32} TFLOP/s · peak {RESULTS["amp_dtype"]} tensor {PEAK16} TFLOP/s')

def flops_per_token(cfg, n_mm):
    return 6 * n_mm, 6 * n_mm + 12 * cfg.n_layer * cfg.block_size * cfg.n_embd

def bench(cfg, B, amp, attn, fused, compiled, steps, warm):
    seed_all(0); m = GPT(replace(cfg, attn=attn)).to(device)
    o = make_opt(m, fused=fused); sc = make_scaler(amp)
    fn = torch.compile(m) if compiled else m
    x, y = get_batch('train', B, cfg.block_size)
    def step():
        with amp_ctx(amp): loss = token_loss(fn(x), y)
        sc.scale(loss).backward(); sc.unscale_(o); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        sc.step(o); sc.update(); o.zero_grad(set_to_none=True)
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    for _ in range(warm): step()
    sync(); t0 = time.perf_counter()
    for _ in range(steps): step()
    sync(); dt = (time.perf_counter() - t0) / steps
    n_mm = sum(p.numel() for n, p in m.named_parameters() if not n.startswith(('wte', 'wpe')))
    f6, fa = flops_per_token(cfg, n_mm); tps = B * cfg.block_size / dt
    out = dict(B=B, step_ms=dt * 1e3, tok_s=tps, tflops_6N=f6 * tps / 1e12, tflops_attn=fa * tps / 1e12,
               mem_GiB=torch.cuda.max_memory_allocated() / 2**30 if device == 'cuda' else float('nan'), n_mm=n_mm)
    if PEAK16:
        out['MFU'] = out['tflops_6N'] / PEAK16; out['MFU_attn'] = out['tflops_attn'] / PEAK16
        out['MFU_vs_fp32_peak'] = out['tflops_6N'] / PEAK32 if not amp else float('nan')
    del m, o, fn
    if device == 'cuda': torch.cuda.empty_cache()
    return out

M_ = P['mfu']; B0 = M_['B']; can_compile = device == 'cuda'
amp_name = RESULTS['amp_dtype']
RUNGS = [('fp32 · naive attention · eager', dict(amp=False, attn='naive', fused=False, compiled=False, B=B0)),
         (f'+ {amp_name} autocast', dict(amp=True, attn='naive', fused=False, compiled=False, B=B0)),
         ('+ fused attention (SDPA)', dict(amp=True, attn='sdpa', fused=False, compiled=False, B=B0)),
         ('+ fused AdamW', dict(amp=True, attn='sdpa', fused=True, compiled=False, B=B0)),
         ('+ torch.compile', dict(amp=True, attn='sdpa', fused=True, compiled=can_compile, B=B0)),
         ('+ 2x micro-batch', dict(amp=True, attn='sdpa', fused=True, compiled=can_compile, B=2 * B0))]
ladder = []
for label, kw in RUNGS:
    try:
        r = bench(CFG, kw['B'], kw['amp'], kw['attn'], kw['fused'], kw['compiled'], M_['steps'], M_['warm'] + (5 if kw['compiled'] else 0))
    except Exception as e:                                    # OOM or compile failure: report it, don't hide it
        print(f'{label}: FAILED ({type(e).__name__}: {str(e)[:120]})'); torch.cuda.empty_cache() if device == 'cuda' else None; continue
    ladder.append(dict(rung=label, **r)); print(f"{label:<34} {r['tok_s']:>10,.0f} tok/s  {r['tflops_6N']:6.2f} TFLOP/s  MFU {100 * r.get('MFU', float('nan')):5.2f}%")
lad = pd.DataFrame(ladder)
display(lad.drop(columns=['n_mm']).style.format({'step_ms': '{:.1f}', 'tok_s': '{:,.0f}', 'tflops_6N': '{:.2f}', 'tflops_attn': '{:.2f}', 'mem_GiB': '{:.2f}',
                                                 'MFU': '{:.2%}', 'MFU_attn': '{:.2%}', 'MFU_vs_fp32_peak': '{:.2%}'}, na_rep='—').hide(axis='index'))

%%md
**What is the most this GPU will ever give?** A large square matmul is the friendliest workload a GPU can get. Its throughput is the *practical* ceiling — and the same benchmark at **our model's actual matmul shapes** shows how much a small `C` leaves on the table before any of our own code is involved.

%%code [core,mfu]
def gemm_tflops(M, N, K_, dt, iters=None):
    iters = iters or (3 if device == 'cpu' else 50)
    a, b = torch.randn(M, K_, device=device, dtype=dt), torch.randn(K_, N, device=device, dtype=dt)
    for _ in range(3): a @ b
    sync(); t0 = time.perf_counter()
    for _ in range(iters): a @ b
    sync(); return 2 * M * N * K_ * iters / (time.perf_counter() - t0) / 1e12

gdt = AMP_DTYPE if device == 'cuda' else torch.float32
C_, BT = CFG.n_embd, B0 * T
shapes = [('square 4096³ (ideal)', 4096, 4096, 4096), ('square 8192³ (ideal)', 8192, 8192, 8192) if not QUICK else ('square 1024³', 1024, 1024, 1024),
          (f'qkv: ({BT}×{C_}) @ ({C_}×{3 * C_})', BT, 3 * C_, C_), (f'attn proj: ({BT}×{C_}) @ ({C_}×{C_})', BT, C_, C_),
          (f'mlp up: ({BT}×{C_}) @ ({C_}×{4 * C_})', BT, 4 * C_, C_), (f'mlp down: ({BT}×{4 * C_}) @ ({4 * C_}×{C_})', BT, C_, 4 * C_),
          (f'lm head: ({BT}×{C_}) @ ({C_}×{V})', BT, V, C_)]
gemm = pd.DataFrame([dict(matmul=n, TFLOPs=gemm_tflops(M, N_, K_, gdt)) for n, M, N_, K_ in shapes])
if PEAK16: gemm['of_peak'] = gemm.TFLOPs / PEAK16
display(gemm.style.format({'TFLOPs': '{:.1f}', 'of_peak': '{:.0%}'}).hide(axis='index'))

%%md
**Where does a step's time actually go?** The PyTorch profiler on the best *eager* rung (compiled graphs fuse kernels and hide the breakdown). Every GPU kernel is sorted into a bucket; whatever is left of the wall-clock step is time the GPU sat idle waiting for Python to launch the next kernel.

%%code [core,mfu]
from torch.profiler import profile, ProfilerActivity
BUCKETS = [('matmul', r'gemm|cutlass|xmma|cublas|sm\d+_|ampere_|turing_|volta_|hopper|wgmma|gemv|addmm|mm_'),
           ('attention', r'flash|fmha|attention|efficient'), ('optimizer', r'adam|multi_tensor|foreach|fused_'),
           ('layernorm · gelu · softmax', r'layer_norm|layernorm|gelu|softmax|norm'),
           ('loss', r'cross_entropy|nll|log_softmax'), ('memcpy', r'memcpy|memset|copy'),
           ('elementwise · reductions', r'elementwise|vectorized|reduce|unrolled|index|embedding|cat|fill|where|mul|add')]
def bucket(name):
    return next((b for b, rx in BUCKETS if re.search(rx, name, re.I)), 'other kernels')

prof_rows, idle_frac = [], float('nan')
if device == 'cuda':
    m = GPT(replace(CFG, attn='sdpa')).to(device); o = make_opt(m, fused=True); sc = make_scaler(); x, y = get_batch('train', B0, T)
    for _ in range(5): train_step(m, o, sc, x, y, 1.0)
    n_prof = 5; sync(); t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_prof): train_step(m, o, sc, x, y, 1.0)
        sync()
    wall_ms = (time.perf_counter() - t0) * 1e3 / n_prof
    for e in prof.key_averages():
        t_us = getattr(e, 'self_device_time_total', None)
        t_us = t_us if t_us is not None else getattr(e, 'self_cuda_time_total', 0)
        if t_us and t_us > 0 and not e.key.startswith(('aten::', 'autograd', 'Optimizer', 'ProfilerStep', 'cuda')):
            prof_rows.append(dict(kernel=e.key[:90], bucket=bucket(e.key), ms_per_step=t_us / 1e3 / n_prof))
    kern = pd.DataFrame(prof_rows)
    by = kern.groupby('bucket').ms_per_step.sum().sort_values(ascending=False)
    busy = by.sum(); idle_frac = max(0.0, 1 - busy / wall_ms)
    by_all = pd.concat([by, pd.Series({'GPU idle (launch overhead)': max(0.0, wall_ms - busy)})])
    display(pd.DataFrame({'ms / step': by_all, 'share': by_all / by_all.sum()}).style.format({'ms / step': '{:.2f}', 'share': '{:.1%}'}))
    print('top 8 kernels:'); display(kern.sort_values('ms_per_step', ascending=False).head(8).style.format({'ms_per_step': '{:.3f}'}).hide(axis='index'))
    print(f'wall time {wall_ms:.1f} ms/step · GPU busy {busy:.1f} ms · idle {idle_frac:.0%}')
    del m, o; torch.cuda.empty_cache()
else:
    by_all = pd.Series(dtype=float); print('profiler breakdown needs a GPU (skipped on CPU)')

%%md
**Is it us, or is it the model's size?** Same loop (best eager settings), same depth, wider and wider models.

%%code [core,mfu]
width = []
for w_ in M_['widths']:
    cfg_w = replace(CFG, n_embd=w_, n_head=max(1, w_ // 64))
    try:
        r = bench(cfg_w, B0, True, 'sdpa', True, False, max(3, M_['steps'] // 2), M_['warm'])
        width.append(dict(n_embd=w_, params_M=r['n_mm'] / 1e6, **{k: r[k] for k in ['tok_s', 'tflops_6N', 'tflops_attn', 'mem_GiB']}, MFU=r.get('MFU', float('nan')), MFU_attn=r.get('MFU_attn', float('nan'))))
        print(f'n_embd {w_:5d}  {r["n_mm"] / 1e6:6.1f}M params  {r["tflops_6N"]:6.2f} TFLOP/s  MFU {100 * r.get("MFU", float("nan")):.1f}%')
    except Exception as e:
        print(f'n_embd {w_}: FAILED ({type(e).__name__})'); torch.cuda.empty_cache() if device == 'cuda' else None
wid = pd.DataFrame(width)
mfu_out = dict(gpu=GPU_NAME, tag=GPU_TAG, amp=amp_name, peak16=PEAK16, peak32=PEAK32, ladder=lad.to_dict('records'),
               gemm=gemm.to_dict('records'), width=wid.to_dict('records'), buckets=by_all.to_dict(), idle_frac=idle_frac)
json.dump(mfu_out, open(RES / f'mfu_{GPU_TAG}.json', 'w'), default=float)
RESULTS['mfu'] = mfu_out

%%code [core,mfu]
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
ax = axes[0]
if PEAK16 and len(lad):
    yv = np.arange(len(lad))[::-1]
    ax.barh(yv, 100 * lad.MFU, color=PAL['blue'], height=0.62, label='MFU (6N)')
    ax.barh(yv, 100 * (lad.MFU_attn - lad.MFU), left=100 * lad.MFU, color='#9ec5f4', height=0.62, label='+ attention FLOPs')
    for yy, (_, r) in zip(yv, lad.iterrows()):
        ax.text(100 * r.MFU_attn + 0.4, yy, f'{100 * r.MFU:.1f}%  ·  {r.tok_s / 1e3:,.0f}k tok/s', va='center', fontsize=8.3, color=INK2)
    ax.set_yticks(yv); ax.set_yticklabels(lad.rung, fontsize=8.5); ax.grid(axis='y', visible=False)
    ax.axvline(40, color=PAL['green'], lw=1.2); ax.text(40, yv.max() + 0.55, ' 40% target', color=PAL['green'], fontsize=8.5)
    ax.set_xlim(0, max(45, 100 * lad.MFU_attn.max() * 1.45)); ax.set_xlabel(f'% of {GPU_TAG} dense {amp_name} tensor peak ({PEAK16:.0f} TFLOP/s)')
    ax.legend(loc='lower right')
title(ax, f'The MFU ladder on a {GPU_TAG}', 'each rung adds one change to the one above')
ax = axes[1]
if len(by_all):
    order = list(by_all.index); colors = [PAL['blue'], PAL['aqua'], PAL['orange'], PAL['violet'], PAL['yellow'], PAL['magenta'], PAL['green'], PAL['red']]
    cmap = {b: (MUTED if 'idle' in b else colors[i % len(colors)]) for i, b in enumerate(order)}
    left = 0
    for b in order:
        v = by_all[b]; ax.barh(0, v, left=left, color=cmap[b], height=0.5, edgecolor=SURF, linewidth=2); left += v
    ax.set_yticks([]); ax.set_xlabel('milliseconds per step'); ax.grid(axis='y', visible=False)
    handles = [mpl.patches.Patch(color=cmap[b], label=f'{b}  {by_all[b] / by_all.sum():.0%}') for b in order]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8)
title(ax, 'Where one step goes', 'profiler, best eager rung')
ax = axes[2]
if len(wid) and PEAK16:
    ax.plot(wid.n_embd, 100 * wid.MFU, color=PAL['blue'], marker='o', ms=7, label='MFU (6N)')
    ax.plot(wid.n_embd, 100 * wid.MFU_attn, color='#9ec5f4', marker='o', ms=6, label='MFU (6N + attention)')
    for _, r in wid.iterrows(): ax.text(r.n_embd, 100 * r.MFU - 2.2, f'{r.params_M:.0f}M', ha='center', fontsize=7.8, color=INK2)
    ax.axhline(40, color=PAL['green'], lw=1.2); ax.set_xscale('log', base=2); ax.set_xticks(wid.n_embd); ax.set_xticklabels(wid.n_embd)
    ax.set_xlabel('model width n_embd (6 layers, eager)'); ax.set_ylabel('MFU %'); ax.legend(loc='upper left'); ax.set_ylim(bottom=0)
title(ax, 'Wider models fill the machine', 'same code, bigger matmuls')
fig.tight_layout(); save(fig, f'05_mfu_{GPU_TAG}')

%%md
**The diagnosis, from the numbers above** (generated, not hand-typed):

%%code [core,mfu]
diag = []
if PEAK16 and len(lad):
    best_r = lad.loc[lad.MFU.idxmax()]; first = lad.iloc[0]
    ceil_ = gemm.TFLOPs.iloc[:2].max()
    diag.append(f'**Headline:** best rung `{best_r.rung}` reaches **{best_r.MFU:.1%} MFU** ({best_r.tflops_6N:.1f} TFLOP/s of {PEAK16:.0f}); '
                f'counting attention FLOPs, {best_r.MFU_attn:.1%}. The fp32 starting point was {first.MFU:.1%} — a {best_r.tok_s / first.tok_s:.1f}× speed-up with the same maths.')
    diag.append(f'**The ceiling is not the datasheet.** A large square {amp_name} matmul — the best case — sustains {ceil_:.1f} TFLOP/s = **{ceil_ / PEAK16:.0%} of peak**. '
                f'{"A T4 is a 70 W card: under sustained load it drops well below its boost clock, so a real ~40% target means ~" + format(0.4 * PEAK16 / ceil_, ".0%") + " of what this card can actually sustain." if GPU_TAG == "T4" else "That gap is clock and power behaviour, not our code."}')
    mshape = gemm.iloc[2:6]
    diag.append(f'**Our matmuls are too small to fill it.** At this model\'s shapes (C={CFG.n_embd}) the same matmuls run at '
                f'{mshape.TFLOPs.min():.1f}–{mshape.TFLOPs.max():.1f} TFLOP/s ({mshape.of_peak.min():.0%}–{mshape.of_peak.max():.0%} of peak): too few columns per tile to keep the tensor cores fed.')
    if len(by_all):
        mm = by_all.get('matmul', 0) / by_all.sum()
        diag.append(f'**Most of a step is not matmul.** The profiler puts matmuls at **{mm:.0%}** of the step; LayerNorm, GELU, softmax, the loss, the optimizer and plain elementwise kernels are memory-bound — they move bytes, not FLOPs — '
                    f'and the GPU idles {idle_frac:.0%} of the wall clock waiting on Python to launch ~hundreds of small kernels.')
    if len(wid) > 1:
        diag.append(f'**Size fixes a lot of it.** With identical code, MFU goes from {wid.MFU.iloc[0]:.1%} at n_embd={wid.n_embd.iloc[0]} to {wid.MFU.iloc[-1]:.1%} at n_embd={wid.n_embd.iloc[-1]}. '
                    f'The distance to 40% is mostly the price of a 10M-parameter model, then the GPU class, then launch overhead.')
    if GPU_TAG == 'T4':
        diag.append('**The GPU generation.** Turing has no bf16 tensor cores (so we pay for fp16 + loss scaling), and FlashAttention-2 needs Ampere; SDPA falls back to the memory-efficient kernel.')
    diag.append('**Not counted in 6N at all:** the optimizer step, clipping, the probe/eval passes, and recomputation. MFU is deliberately strict — "useful model FLOPs" only.')
display(Markdown('\n'.join(f'- {d}' for d in diag) if diag else '_MFU needs a GPU; run with S10_PROFILE=full on CUDA._'))
RESULTS['mfu_diagnosis'] = diag

%%code [mfu-compare]
others = {}
for fpath in sorted(RES.glob('mfu_*.json')):
    d = json.load(open(fpath))
    if d.get('peak16'): others[d['tag']] = d
if len(others) >= 2:
    fig, ax = plt.subplots(figsize=(11, 4.4))
    tags = list(others); rungs = [r['rung'].replace('bfloat16', 'mixed').replace('float16', 'mixed') for r in others[tags[0]]['ladder']]
    w = 0.8 / len(tags)
    for i, (tg, c) in enumerate(zip(tags, [PAL['blue'], PAL['orange'], PAL['aqua']])):
        vals = [100 * r['MFU'] for r in others[tg]['ladder']]
        xs = np.arange(len(vals)) + (i - (len(tags) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.9, color=c, label=f"{tg} ({others[tg]['amp']}, peak {others[tg]['peak16']:.0f} TFLOP/s)")
        for xx, v in zip(xs, vals): ax.text(xx, v + 0.4, f'{v:.1f}', ha='center', fontsize=7.8, color=INK2)
    ax.set_xticks(np.arange(len(rungs))); ax.set_xticklabels([r.replace(' · ', '\n') for r in rungs], fontsize=8)
    ax.axhline(40, color=PAL['green'], lw=1.2); ax.set_ylabel('MFU %'); ax.legend(loc='upper left'); ax.grid(axis='x', visible=False)
    title(ax, 'Same code, two machines', 'MFU against each GPU\'s own dense tensor-core peak')
    save(fig, '05_mfu_cross_gpu')
    RESULTS['mfu_cross'] = {tg: [dict(rung=r['rung'], MFU=r['MFU'], tok_s=r['tok_s']) for r in d['ladder']] for tg, d in others.items()}
else:
    print('Only one GPU measured so far — the cross-GPU chart appears once results/mfu_<GPU>.json exists for a second card.')

%%md
<a id="6"></a>
## 6 · The number 0.1, by hand

### Step 1 — 0.1 in binary
Multiply the fraction by 2 again and again; each time, the integer part is the next binary digit.

| ×2 | 0.1→0.2 | 0.2→0.4 | 0.4→0.8 | 0.8→**1**.6 | 0.6→**1**.2 | 0.2→0.4 | 0.4→0.8 | 0.8→**1**.6 | 0.6→**1**.2 | … |
|---|---|---|---|---|---|---|---|---|---|---|
| bit | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 | repeats `0011` forever |

$$0.1_{10} = 0.0\,0011\,0011\,0011\ldots_2 = 1.\underbrace{1001\,1001\,1001\,1001\ldots}_{\text{mantissa, never ends}} \times 2^{-4}$$

So every format stores **sign = 0**, **true exponent = −4**, and as much of `1001 1001 1001…` as it has room for, **rounded to nearest, ties to even**. The digit after the cut is always `1` followed by non-zero bits, so every format below **rounds up**.

### Step 2 — fp32 (1 · 8 · 23, bias 127)
* exponent field = −4 + 127 = **123** = `01111011`
* mantissa: first 23 bits `10011001100110011001100`, next bits `1100…` > half → round up → `10011001100110011001101`
* **bits: `0 01111011 10011001100110011001101`** = `0x3DCCCCCD`
* value = (1 + 5033165/2²³) × 2⁻⁴ = 13421773 / 2²⁷ = **0.100000001490116119384765625** (error +1.5 × 10⁻⁹, 0.0000015%)

### Step 3 — bf16 (1 · 8 · 7, bias 127)
* same exponent as fp32 — that is the whole point of bf16 — `01111011`
* mantissa: `1001100`, next bits `1100…` → round up → `1001101` (= 77)
* **bits: `0 01111011 1001101`** = `0x3DCD` (literally the top half of fp32's `0x3DCCCCCD`, rounded)
* value = (1 + 77/128) × 2⁻⁴ = 205/2048 = **0.10009765625** (error +9.8 × 10⁻⁵, **0.098%**)

### Step 4 — fp8 E4M3 (1 · 4 · 3, bias 7)
* exponent field = −4 + 7 = **3** = `0011`
* mantissa: `100`, next bits `1100…` → round up → `101` (= 5)
* **bits: `0 0011 101`** = `0x1D`
* value = (1 + 5/8) × 2⁻⁴ = 13/128 = **0.1015625** (error +1.6 × 10⁻³, **1.56%**)
* its neighbours are 0.09375 and 0.109375 — nothing in between exists.

Now let the machine check every bit.

%%code [core,float]
def rne(fr):
    f = math.floor(fr); d = fr - f
    return f + 1 if d > Fraction(1, 2) else f if d < Fraction(1, 2) else (f if f % 2 == 0 else f + 1)

FORMATS = {'fp32': (8, 23, True), 'fp16': (5, 10, True), 'bf16': (8, 7, True), 'fp8 E4M3': (4, 3, False), 'fp8 E5M2': (5, 2, True)}

def encode(x, eb, mb, ieee=True):
    """Exact round-to-nearest-even encoding of a rational x. E4M3 (ieee=False) has no infinities: max 448."""
    x = Fraction(x); bias = 2 ** (eb - 1) - 1; emin = 1 - bias
    s = 1 if x < 0 else 0; a = abs(x)
    if a == 0: return s, 0, 0
    e = math.frexp(float(a))[1] - 1
    while a < Fraction(2) ** e: e -= 1
    while a >= Fraction(2) ** (e + 1): e += 1
    if e < emin:                                             # subnormal
        q = rne(a / Fraction(2) ** emin * 2 ** mb)
        return (s, 1, 0) if q == 2 ** mb else (s, 0, q)
    M = rne((a / Fraction(2) ** e - 1) * 2 ** mb)
    if M == 2 ** mb: M, e = 0, e + 1
    E = e + bias; top = 2 ** eb - 1
    if ieee and E >= top: return s, top, 0                   # overflow -> inf
    if not ieee and (E > top or (E == top and M == 2 ** mb - 1)): return s, top, 2 ** mb - 2   # saturate to 448
    return s, E, M

def decode(s, E, M, eb, mb, ieee=True):
    bias = 2 ** (eb - 1) - 1; sign = -1 if s else 1
    if ieee and E == 2 ** eb - 1: return float('nan') if M else sign * float('inf')
    if E == 0: return sign * Fraction(M, 2 ** mb) * Fraction(2) ** (1 - bias)
    return sign * (1 + Fraction(M, 2 ** mb)) * Fraction(2) ** (E - bias)

def q(x, fmt):
    eb, mb, ieee = FORMATS[fmt]; return decode(*encode(x, eb, mb, ieee), eb, mb, ieee)

rows = []
for fmt in FORMATS:
    eb, mb, ieee = FORMATS[fmt]; s, E, M = encode(Fraction(1, 10), eb, mb, ieee); val = decode(s, E, M, eb, mb, ieee)
    word = (s << (eb + mb)) | (E << mb) | M
    rows.append(dict(format=fmt, layout=f'1·{eb}·{mb}', sign=s, exponent=f'{E:0{eb}b}', mantissa=f'{M:0{mb}b}', hex=f'0x{word:0{(1 + eb + mb + 3) // 4}X}',
                     value=f'{float(val):.12g}', exact=str(val), rel_error=float((val - Fraction(1, 10)) / Fraction(1, 10)),
                     ulp_at_0_1=float(Fraction(2) ** (-4 - mb))))
ftab = pd.DataFrame(rows)
display(ftab.style.format({'rel_error': '{:+.3e}', 'ulp_at_0_1': '{:.3e}'}).hide(axis='index'))

hw_fp32 = struct.unpack('>I', struct.pack('>f', 0.1))[0]
hw_bf16 = torch.tensor([0.1], dtype=torch.float32).to(torch.bfloat16).view(torch.int16).item() & 0xFFFF
hw_e4m3 = torch.tensor([0.1], dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8).item()
mine = {r['format']: int(r['hex'], 16) for r in rows}
for fmt, hw in [('fp32', hw_fp32), ('bf16', hw_bf16), ('fp8 E4M3', hw_e4m3)]:
    print(f'{fmt:>9}: by hand 0x{mine[fmt]:X} · hardware 0x{hw:X} · {"MATCH ✓" if mine[fmt] == hw else "MISMATCH ✗"}')
    assert mine[fmt] == hw
RESULTS['float'] = ftab.to_dict('records')

%%code [core,float]
show_f = ['fp32', 'bf16', 'fp8 E4M3']
fig, axes = plt.subplots(len(show_f), 1, figsize=(13, 1.25 * len(show_f) + 0.9))
fcol = dict(sign=PAL['violet'], exp=PAL['blue'], man=PAL['orange'])
for ax, fmt in zip(axes, show_f):
    r = ftab.set_index('format').loc[fmt]; bits = [('sign', str(r.sign))] + [('exp', b) for b in r.exponent] + [('man', b) for b in r.mantissa]
    for i, (k, b) in enumerate(bits):
        ax.add_patch(mpl.patches.FancyBboxPatch((i * 1.0 + 0.06, 0.08), 0.88, 0.84, boxstyle='round,pad=0,rounding_size=0.12',
                                                facecolor=fcol[k] if b == '1' else SURF, edgecolor=fcol[k], linewidth=1.4))
        ax.text(i + 0.5, 0.5, b, ha='center', va='center', fontsize=10, fontweight='bold', color=SURF if b == '1' else fcol[k], family='monospace')
    ax.text(-0.6, 0.5, fmt, ha='right', va='center', fontsize=11, fontweight='bold')
    ax.text(33.2, 0.5, f'{r.hex}   =  {r.value}\nrel. error {r.rel_error:+.2e}', ha='left', va='center', fontsize=9, color=INK2, family='monospace')
    ax.set_xlim(-6.5, 46); ax.set_ylim(0, 1); ax.axis('off')
handles = [mpl.patches.Patch(color=fcol[k], label=l) for k, l in [('sign', 'sign'), ('exp', 'exponent (range)'), ('man', 'mantissa (detail)')]]
fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.suptitle('0.1, bit by bit — exponent bits buy range, mantissa bits buy detail', x=0.06, ha='left', fontweight='bold', fontsize=12)
save(fig, '06_bits_of_0.1')

%%code [core,float]
fig, ax = plt.subplots(figsize=(13, 3.3))
lo, hi = 0.0925, 0.1105
for yrow, (fmt, c) in enumerate([('fp8 E4M3', PAL['orange']), ('bf16', PAL['blue']), ('fp32', PAL['aqua'])]):
    eb, mb, ieee = FORMATS[fmt]; vals = []
    for e in (-5, -4):
        for M in range(2 ** mb if mb < 12 else 0):
            v = float((1 + Fraction(M, 2 ** mb)) * Fraction(2) ** e)
            if lo <= v <= hi: vals.append(v)
    if vals: ax.vlines(vals, yrow - 0.28, yrow + 0.28, color=c, lw=1.6 if fmt != 'bf16' else 0.9)
    else: ax.add_patch(mpl.patches.Rectangle((lo, yrow - 0.28), hi - lo, 0.56, color=c, alpha=0.35, lw=0))
    v01 = float(q(Fraction(1, 10), fmt)); ax.plot([v01], [yrow], 'o', ms=9, color=c, markeredgecolor=SURF, markeredgewidth=2)
    spacing = float(Fraction(2) ** (-4 - mb))
    ax.text(hi + 0.0003, yrow, f'{fmt}: spacing {spacing:.1e}' + ('  (≈8.4M values per octave)' if fmt == 'fp32' else ''), va='center', fontsize=8.8, color=INK2)
ax.axvline(0.1, color=INK, lw=1); ax.text(0.1, 2.55, 'true 0.1', ha='center', fontsize=9)
ax.set_yticks([]); ax.set_ylim(-0.6, 2.8); ax.set_xlim(lo, hi)
for t_ in ax.texts: t_.set_clip_on(False); ax.grid(axis='y', visible=False)
ax.set_xlabel('the number line around 0.1 (every tick is a value the format can hold; dot = where 0.1 lands)')
title(ax, 'What each format can say near 0.1')
save(fig, '06_number_line')

%%md
### Why this matters for training: update swamping

A weight sits at 1.0 and the optimizer adds `+0.001` to it, a thousand times. The right answer is 2.0. In bf16 the gap between 1.0 and the next number up is 2⁻⁷ ≈ 0.0078, so a 0.001 step is *less than half a gap* and rounds straight back to 1.0 — every single time. **That is why the lesson's 16 bytes per weight include a 4-byte fp32 master copy**: the update is applied in fp32 and only then rounded to bf16 for the next forward pass. Stochastic rounding (round up with probability proportional to the distance) is the other way out — correct *on average* — and is what lets fp8 / NVFP4 recipes survive.

%%code [core,float]
rng = np.random.default_rng(0)
def q_sr(x, fmt):
    """stochastic rounding: pick the neighbour above with probability = how close x is to it"""
    eb, mb, _ = FORMATS[fmt]; x = Fraction(x); e = math.floor(math.log2(float(abs(x)))); ulp = Fraction(2) ** (e - mb)
    lo_ = (x // ulp) * ulp; p = float((x - lo_) / ulp)
    return lo_ + ulp if rng.random() < p else lo_

n_upd, du = 1000, 0.001
traj = {k: [1.0] for k in ['fp32', 'bf16 (round-to-nearest)', 'bf16 + stochastic rounding', 'fp8 E4M3 (round-to-nearest)']}
w = {k: Fraction(1) for k in traj}
for _ in range(n_upd):
    w['fp32'] = q(w['fp32'] + Fraction(du), 'fp32'); w['bf16 (round-to-nearest)'] = q(w['bf16 (round-to-nearest)'] + Fraction(du), 'bf16')
    w['bf16 + stochastic rounding'] = q_sr(w['bf16 + stochastic rounding'] + Fraction(du), 'bf16'); w['fp8 E4M3 (round-to-nearest)'] = q(w['fp8 E4M3 (round-to-nearest)'] + Fraction(du), 'fp8 E4M3')
    for k in traj: traj[k].append(float(w[k]))

sums = {f: [0.0] for f in ['fp32', 'fp16', 'bf16', 'fp8 E4M3']}; acc = {f: Fraction(0) for f in sums}
for _ in range(2000):
    for f in sums: acc[f] = q(acc[f] + Fraction(1, 10), f); sums[f].append(float(acc[f]))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
ax = axes[0]
for (k, v), c in zip(traj.items(), [PAL['aqua'], PAL['red'], PAL['blue'], PAL['orange']]):
    ax.plot(v, color=c, label=f'{k} → {v[-1]:.4f}', lw=2 if 'stochastic' not in k else 1.4)
ax.axhline(2.0, color=MUTED, lw=0.8); ax.set_xlabel('update number'); ax.set_ylabel('weight'); ax.legend(loc='upper left')
title(ax, 'A weight that should reach 2.0', '1,000 updates of +0.001 each')
ax = axes[1]
for (f, v), c in zip(sums.items(), [PAL['aqua'], PAL['violet'], PAL['blue'], PAL['orange']]):
    ax.plot(v, color=c, label=f'{f}: stalls at {v[-1]:g}' if v[-1] < 199 else f'{f}: {v[-1]:.2f}')
ax.plot([0, 2000], [0, 200], color=MUTED, lw=0.8); ax.set_xlabel('number of 0.1s added'); ax.set_ylabel('running sum'); ax.legend(loc='upper left')
title(ax, 'Adding 0.1, two thousand times', 'each format stops growing once 0.1 is under half its spacing')
fig.tight_layout(); save(fig, '06_swamping')
RESULTS['swamping'] = {k: v[-1] for k, v in traj.items()} | {f'sum_{k}': v[-1] for k, v in sums.items()}

%%md
### fp16 underflow and loss scaling — on this model's real gradients

The lesson's claim: fp16 cannot hold anything below 2⁻²⁴ ≈ 5.96 × 10⁻⁸, and below 2⁻¹⁴ ≈ 6.1 × 10⁻⁵ it only holds a few digits (subnormals). Instead of a made-up gradient, take **every gradient element of the trained model** (fp32, one batch), cast it to fp16 and bf16, and count the casualties. Then multiply by a loss scale first, as `GradScaler` does.

%%code [core,float]
gm = copy.deepcopy(model).float() if 'model' in globals() else GPT(CFG).to(device)
gm.zero_grad(); xg, yg = get_batch('val', P['B'], T, torch.Generator().manual_seed(5)); token_loss(gm(xg), yg).backward()
g = torch.cat([p.grad.flatten() for p in gm.parameters()]).float(); g = g[g != 0]
rows = []
for S_ in [1, 2**10, 2**16]:
    gs = g * S_; h16 = gs.half(); b16 = g.bfloat16()
    rows.append(dict(loss_scale=f'{S_:,}', fp16_zeroed=(h16 == 0).float().mean().item(), fp16_subnormal=((h16 != 0) & (gs.abs() < 2**-14)).float().mean().item(),
                     fp16_overflow=torch.isinf(h16).float().mean().item(), bf16_zeroed=(b16 == 0).float().mean().item()))
uf = pd.DataFrame(rows)
display(uf.style.format({c: '{:.4%}' for c in uf.columns if c != 'loss_scale'}).hide(axis='index'))

fig, ax = plt.subplots(figsize=(12, 4))
lg = torch.log10(g.abs()).cpu().numpy()
ax.hist(lg, bins=120, color=PAL['blue'], alpha=0.85, label='|gradient|, loss scale 1')
ax.hist(lg + np.log10(2**10), bins=120, color=PAL['aqua'], alpha=0.45, label='× 1024 loss scale')
for v, lab, c in [(np.log10(2**-24), 'fp16: becomes exactly 0', PAL['red']), (np.log10(2**-14), 'fp16: subnormal, losing digits', PAL['yellow']), (np.log10(65504), 'fp16 max 65504', PAL['red'])]:
    ax.axvline(v, color=c, lw=1.4); ax.text(v, ax.get_ylim()[1] * 0.95, ' ' + lab, color=INK2, fontsize=8, rotation=90, va='top')
ax.set_xlabel('log10 |gradient element|  (bf16 floor is 10⁻³⁸, far off the left edge)'); ax.set_ylabel('count'); ax.legend(loc='upper left')
title(ax, f'{g.numel() / 1e6:.1f}M real gradients against the fp16 floor', f'trained model, step {P["main_steps"]}: {uf.fp16_zeroed[0]:.3%} vanish and {uf.fp16_subnormal[0]:.2%} lose precision in fp16 at scale 1 · bf16 loses {uf.bf16_zeroed[0]:.3%}')
save(fig, '06_fp16_underflow')
RESULTS['underflow'] = uf.to_dict('records')

%%code [core,float]
if 'L_' in globals() and np.ptp(L_['scale']) > 0:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(L_['scale'], color=PAL['violet']); ax.set_yscale('log', base=2); ax.set_xlabel('step'); ax.set_ylabel('loss scale')
    drops = int(np.sum(np.diff(L_['scale']) < 0))
    title(ax, 'GradScaler at work during the main run', f'doubles after 2,000 clean steps, halves on any inf/NaN gradient ({drops} back-offs, each a skipped step)')
    save(fig, '06_loss_scale')
    RESULTS['loss_scale'] = dict(final=float(L_['scale'][-1]), backoffs=drops)
else:
    print('No dynamic loss scale in this run (bf16 or CPU) — nothing to plot; bf16 never needs it.')

%%md
### Which one would I train in?

**Compute in bf16, keep master weights and optimizer state in fp32 — and move the big matmuls to fp8 E4M3 only on hardware that does it natively, with per-block scales.**

* **Not fp32** for compute: it is 7.2 digits of precision that gradient noise drowns anyway, at 2× the memory traffic and a fraction of the tensor-core speed (§5: the fp32 rung).
* **bf16** keeps fp32's 8 exponent bits, so the gradient histogram above has nothing to fear from underflow and no loss scale to tune. Its 2.4 digits are enough for *activations and gradients* — but not for *accumulating updates into weights*, which is what the swamping plot shows. Hence the fp32 master copy: those 4 extra bytes per weight buy the 0.001-step that bf16 alone would delete.
* **fp8 E4M3** stores 0.1 with 1.56% error and cannot represent anything between 0.094 and 0.109. That is acceptable for matmul *inputs* when every block of values carries its own scale (the 2026 production recipe), and unacceptable for anything that accumulates: weights, optimizer moments, softmax, the loss, and the reductions inside LayerNorm.
* **On this T4** specifically, the honest answer is fp16 + dynamic loss scaling — Turing has no bf16 tensor cores. The underflow table and the GradScaler trace above show what that costs.

%%md
## Summary

%%code [summary]
s = RESULTS
lines = [f"**Model** {s['n_params'] / 1e6:.2f}M params · {s['gpu']} · {s['amp_dtype']} · torch {s['torch']}"]
if 'shapes' in s: lines.append(f"**1 · Shapes** training state {s['shapes']['state_MiB']:.0f} MiB (16 B/weight) vs activations {s['shapes']['act_MiB']:.0f} MiB — {s['shapes']['att_share']:.0%} of them attention matrices")
if 'gradcheck' in s: lines.append(f"**2 · Gradients** backward() and the nudge agree to ≥ {s['gradcheck']['min_digits']:.1f} significant digits in all 10 layer types (fp64)")
if 'accum' in s: lines.append(f"**3 · Accumulation** big batch vs correct loop: max |Δgrad| {s['accum']['big_vs_correct_maxdiff']:.1e}; broken loop logs a loss {s['accum']['report_gap']:+.3f} off the truth; final val {s['accum']['final_val']['correct']:.4f} (correct) vs {s['accum']['final_val']['broken']:.4f} (broken)")
if 'inject' in s: lines.append(f"**4 · Grad norm** poisoned batch: norm {s['inject']['poisoned, no clip']['norm_at_poison']:.1f} on the step itself, loss reacts after the update; averaged over {s['norm']['n_spikes']} natural spikes, the loss excess peaks {s['norm']['sta_peak_k']} step(s) after the norm")
if 'mfu' in s and s['mfu'].get('ladder'):
    b = max(s['mfu']['ladder'], key=lambda r: r.get('MFU', 0)); lines.append(f"**5 · MFU** best {b.get('MFU', float('nan')):.1%} on {s['mfu']['tag']} ({b['rung']})")
if 'float' in s: lines.append('**6 · 0.1** fp32 `0x3DCCCCCD` · bf16 `0x3DCD` (0.098% off) · fp8 E4M3 `0x1D` (1.56% off) — all verified against hardware')
display(Markdown('\n\n'.join(lines)))
json.dump(RESULTS, open(RES / 'summary.json', 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print('saved results/summary.json and', len(list(FIG.glob('*.png'))), 'figures')
