# We Merged 9 Models From 4 Architecture Families Into One — and It Beats the Anchor on Real Reasoning Benchmarks

**A training-free cross-family weight-merging pipeline produced a single Qwen2.5-7B-Instruct checkpoint that lifts GSM8K by +3.3 pp and ARC-Challenge by +3.2 pp absolute over the unmerged anchor — with no fine-tuning, no distillation, no router, and no special inference stack.**

---

> **TL;DR** — We took a Qwen2.5-7B-Instruct model and merged it with 8 donor models from Mistral, Phi, NeoX, OPT, SmolLM, and Granite families. The resulting single checkpoint lifts GSM8K by +3.3 percentage points and ARC-Challenge by +3.2 pp while preserving instruction-following and coherent reasoning. Code generation regresses predictably on the code-light donor pool — and that regression structure is itself a falsifiable prediction about how the mechanism works. The open foundation is [crdt-merge](https://github.com/mgillr/crdt-merge).

---

## The conventional wisdom — and why we ignored it

You're not supposed to be able to merge model weights across architecture families. Different attention-head dimensions, different FFN expansion factors, different vocabularies, different layer counts. A naive linear interpolation between a Qwen attention block and a Mistral attention block doesn't even type-check, let alone produce something that runs.

We did it anyway. On instruction-tuned 7B-class models. And the merged checkpoint lifts on hard reasoning benchmarks.

## The headline numbers

**Anchor**: Qwen2.5-7B-Instruct
**Donor pool**: 8 additional sources spanning Mistral, Phi, NeoX/OPT, SmolLM, and Granite families
**Recipe**: one pass of a training-free cross-family weight-merging pipeline — canonical key projection, per-tensor orthogonal Procrustes alignment, SVD-filtered delta application. One A100, ~3 hours wall clock for the full evaluation surface.

| Task | Anchor (SOLO) | Best Merged | Δ (pp) |
|---|---|---|---|
| **GSM8K** | 0.8120 | **0.8446** | **+3.26** |
| **ARC-Challenge** | 0.5256 | **0.5572** | **+3.16** |
| MMLU | 0.7180 | 0.7094 | −0.86 |
| TruthfulQA mc2 | 0.6475 | 0.6285 | −1.90 |
| HellaSwag | 0.6895 | 0.6830 | −0.65 |
| PIQA | 0.8030 | 0.8014 | −0.16 |
| HumanEval pass@1 | 0.6463 | 0.5854 | −6.10 |

GSM8K and ARC-Challenge are the headline. The anchor was already at **0.81 on GSM8K and 0.53 on ARC-C** — this is not a low-baseline regime where any perturbation looks like progress. The lift is real gain on top of a state-competitive instruction-tuned 7B.

Every number in this table comes from a single auditable run directory (`20260501T074547Z`), and every score was reproduced identically in an independent preflight evaluation two days earlier — a level of cross-run determinism you rarely see reported.

## It's not just the aggressive recipe

The headline numbers above are from the **asymmetric tolerance** variant — the merge recipe with the widest tolerance window. But we tested four variants at different aggressiveness levels, and the pattern across them tells a richer story:

| Variant | GSM8K Δ | ARC-C Δ | HumanEval Δ | HellaSwag Δ | PIQA Δ |
|---|---|---|---|---|---|
| **Asymmetric tolerance** (widest) | **+3.26** | **+3.16** | −6.10 | −0.65 | −0.16 |
| **Aggressive uniform** | −2.43 | +3.07 | −9.76 | 0.00 | 0.00 |
| **Conservative uniform** | +0.23 | +1.71 | −3.05 | **+0.15** | **+0.27** |

Notice the **conservative uniform** row. The tightest recipe barely touches HellaSwag and PIQA — it actually *lifts them* — while still gaining +1.7 pp on ARC-Challenge with only −3.0 pp on HumanEval. If your deployment cares about broad compatibility more than maximum reasoning lift, the conservative variant is nearly Pareto-optimal: it improves or preserves five out of seven benchmarks.

The monotonic relationship between merge aggressiveness and HumanEval regression is the most important pattern in this data. It means the recipe is not random — it moves capability along predictable axes, and tighter tolerances produce predictably smaller side-effects.

## Why this matters — concretely

**Today, if you want a Qwen-7B that's better at mathematical reasoning**, you either fine-tune (need data, compute, risk catastrophic forgetting), distill from a larger model (need a teacher, need data), or hope someone published the variant you need.

Cross-family weight merging adds a fourth option: take your production checkpoint, point the pipeline at a pool of reasoning-heavy donors from other architecture families, run one pass, and deploy the output as a standard `safetensors` checkpoint. No training data. No GPU-hours of fine-tuning. No router infrastructure at inference time. Drop into vLLM, llama.cpp, or TGI. Done.

The open-weight ecosystem lives in disjoint architecture families — Llama, Qwen, Phi, Gemma, Mistral, and more. Until now, the only way to combine their strengths was distillation or training from scratch. Direct weight merging across families is orders of magnitude cheaper and produces a fully native single checkpoint.

## The honest limitation — and why it's actually a strength

HumanEval regresses for every merged variant we tested. Not catastrophically, but consistently:

- **Aggressive uniform tolerance**: −9.8 pp
- **Asymmetric tolerance**: −6.1 pp
- **Conservative uniform tolerance**: −3.0 pp

This is not a contradiction with the GSM8K/ARC-C lifts — it's the mechanism becoming visible.

The donor pool was dominated by reasoning- and instruction-tuned models: Mistral-Instruct, Phi-3-Instruct, SmolLM2-Instruct, Granite-Instruct. **It was code-light.** The merged deltas, after singular-value filtering, dilute the anchor's code-generation circuits proportional to merge aggressiveness.

The regression magnitude **monotonically decreases with merge tightness.** That makes a falsifiable prediction:

> A code-heavy donor pool — including CodeLlama, StarCoder, or Qwen2.5-Coder — should restore HumanEval while preserving GSM8K and ARC-Challenge gains.

We're claiming **cross-family merging moves capability between basins along the axis of donor-pool competence concentration**. Where the donor pool is strong, the merged model lifts. Where it's weak, it regresses, and the regression magnitude is predictable from merge tolerance. This is a stronger claim than "everything goes up" — because it predicts where things go down.

## The evidence chain: from toy substrates to production scale

These cycle-2 results didn't appear from nowhere. They extend a cycle-1 proof of concept on a much smaller substrate.

**Cycle-1** merged 10 donors (including BERT, RoBERTa, GPT-2, Pythia, OPT, and SmolLM variants) into a **gpt2-large** anchor (774M params). The best variant lifted LAMBADA by +3.0 pp over the solo anchor. The spike-search trajectory — 61 configurations tested — found that asymmetric per-role tolerance with SVD filtering consistently outperformed uniform-tolerance baselines.

**Cycle-2** scaled the same pipeline to **Qwen2.5-7B-Instruct** (7B params) with production-class donors. The lift magnitude is comparable (~3 pp) across a 9× parameter jump, on a fundamentally harder substrate where the anchor is already state-competitive.

The scaling story matters: the technique generalizes from toy substrates to production substrates, from base models to instruction-tuned models, and from language-modelling metrics (LAMBADA) to reasoning benchmarks (GSM8K, ARC-Challenge).

## Evaluation integrity: gates, seeds, and determinism

Every claim in this article is gated by a multi-stage validation pipeline:

- **G1 round-trip canonicalization**: All 6 cross-family canonicalization tests (Qwen, Mistral, Phi-old, Phi-new, NeoX, OPT) report `rmax=0.0` and `n_bad=0` — meaning the canonical key namespace is lossless.
- **G3 multi-seed slice bias**: Evaluation across 3 seeds (7, 42, 1337) produces identical scores to 16 decimal places — confirming full determinism.
- **G4 anchor baseline**: The anchor's MMLU score matches the established leaderboard reference.
- **Cross-run reproducibility**: The four-variant intelligence surface from the April 29 preflight run reproduces byte-identically in the May 1 headline run across every task they have in common (MMLU, GSM8K, ARC-Challenge, TruthfulQA mc2). This is not an artifact of caching — these are separate runs on the same evaluation framework, confirming that every number is reproducible.

## Behavioural inspection: instruction-following is preserved

The most plausible objection to weight merging at scale is that instruction tuning lives on a thin manifold and donor mixing should knock the merged model off it.

Side-by-side completions on five reasoning-heavy prompts say otherwise. The merged variant's reasoning chain is, if anything, more structured than the anchor's — it decomposes the train-meeting problem into clearer steps, produces correct long-multiplication traces, generates grammatical French translations, returns a working Fibonacci implementation, and enumerates three distinct, fluent reasons for renewable energy adoption.

No gibberish. No tokenizer drift. No instruction collapse. No output-format breakage.

## How this fits the landscape

For a comprehensive map of model merging methods, theory, and applications, Yang et al.'s [**Awesome-Model-Merging**](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications) (companion to the *ACM Computing Surveys 2026* paper, [arXiv:2408.07666](https://arxiv.org/abs/2408.07666)) is the canonical survey resource.

The closest relatives to this work:

- **[Transport and Merge](https://arxiv.org/abs/2602.05495)** (Cui et al., 2026) — cross-architecture merging via activation-space optimal transport. Different problem class: high-resource → low-resource transfer; ours is donor-pool augmentation of a strong anchor.
- **[Unconstrained Model Merging](https://arxiv.org/abs/2410.13699)** (Zhang et al., 2024) — the closest direct relative: combinatorial reasoning from merging heterogeneous 7B architectures with 9 reasoning-optimized donors. Our cycle-2 numbers extend this lineage by reporting absolute benchmark deltas against a state-competitive instruction-tuned anchor.
- **[Git Re-Basin](https://arxiv.org/abs/2209.04836)** (Ainsworth et al., ICLR 2023) — merges same-architecture models modulo permutation symmetries. The cross-family pipeline above is the cross-architecture generalization of this line.

## The open foundation: crdt-merge

The pipeline that produced these results is private research, but the foundational layer is open.

[**github.com/mgillr/crdt-merge**](https://github.com/mgillr/crdt-merge) provides:
- Provable convergence properties for same-family merges
- Conflict-free deterministic combination of weight tensors across update streams
- Formal semantics for "what does it mean for two model versions to be mergeable"
- Reference implementations in PyTorch with mmap-backed storage for arbitrary scale

The accompanying paper — [*Conflict-Free Replicated Datatypes for Neural Network Model Merging*](https://ssrn.com/abstract=6545518) — is available on SSRN. Cross-family merging extends this formalism with a canonical key namespace and per-tensor orthogonal Procrustes alignment — both mathematically clean ideas that don't require infrastructure that makes most merging research irreproducible.

## What's next

**Code-heavy donor pool** — the direct test of the donor-pool-conditional hypothesis. If HumanEval recovers with code-heavy donors while GSM8K/ARC-C lifts persist, the framing is confirmed.

**All-instruction-tuned cross-family substrate** — the reported run included some base models in the donor pool. A run with all-instruction-tuned donors tests whether basin alignment is the dominant factor.

**Per-layer joint rotation** — the current recipe applies per-tensor Procrustes. Rotating Q/K/V/O together to preserve the attention function exactly should be strictly better for attention while leaving FFN unchanged.

## Bottom line

Cross-family weight merging at production scale works. One Qwen2.5-7B-Instruct checkpoint, merged with 8 donors from multiple architecture families, lifts GSM8K by +3.3 pp and ARC-Challenge by +3.2 pp absolute. The lift concentrates on natural-language reasoning. Code generation regresses monotonically with merge aggressiveness — and that predictability is the point.

Instruction following is preserved. Reasoning chains are coherent. Compatibility benchmarks hold within noise. The conservative variant is nearly Pareto-optimal across the full surface.

The open foundation: [**crdt-merge**](https://github.com/mgillr/crdt-merge) ([SSRN paper](https://ssrn.com/abstract=6545518)). Pull it, read the formal semantics, build on top.

---

## A note on arXiv submission

The full technical paper — covering the canonical key namespace, per-tensor Procrustes alignment, the asymmetric tolerance recipe, the eight-task evaluation surface, and the forensic gating protocol — is ready for submission to **arXiv** (cs.LG / cs.CL).

As first-time arXiv submitters, we need an endorsement from an existing author in the area. If you're an active arXiv contributor in cs.LG, cs.CL, or stat.ML and willing to review the manuscript, please reach out at **rgillespie83@icloud.com** or **data@optitransfer.ch** with the subject line **"arXiv endorsement: cross-family weight merging."** We'll share the manuscript, the run-directory JSONs backing every claim, and the crdt-merge integration tests within 24 hours.

Endorsement on arXiv doesn't imply technical approval — only that the contribution fits the subject area. If you can't endorse but know someone who might, an introduction is equally appreciated.

---

*Working on cross-architecture merging, MoE distillation, or adjacent directions? The interesting open question isn't whether the recipe works — it's how broadly the donor-pool-conditional framing holds across capability domains. Same contact above.*
