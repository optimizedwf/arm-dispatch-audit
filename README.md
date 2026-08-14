# Neoverse SVE Dispatch Audit — "SVE2 advertised, never used"

**Measured on GitHub-hosted `ubuntu-24.04-arm` (Neoverse-N2, 4 vCPU) with llama.cpp `b10434` + KleidiAI v1.24.0.**

> **The finding:** On 128-bit-SVE Neoverse cores (N2, Graviton4, Azure Cobalt 100, every free Arm CI runner),
> KleidiAI's SVE matmul kernels are **dead code**. The dispatch gate is `runtime_feat.sve_cnt == QK8_0`
> (`svcntb() == 32` bytes = 256-bit), but these parts report `svcntb() == 16` (128-bit).
> So the SVE2 your `lscpu` advertises is never used by llama.cpp — matmul actually runs `neon_i8mm`/`neon_dotprod`.
>
> **Consequence for Arm developers:** "your chip supports SVE2" does not mean your inference stack uses it.
> Compile-time ISA flags (`-march=armv8.6-a+i8mm`, no SVE) can beat `-mcpu=native` by up to **2× prefill**
> (armkiln, Graviton5), precisely because the SVE path is a trap on 128-bit Neoverse.

## Evidence (from CI run)

| Check | Result |
|---|---|
| `svcntb()` on runner | **16** (128-bit SVE) — never 32 |
| KleidiAI gate in `kleidiai.cpp` | `sve_cnt == QK8_0 (32) ? CPU_FEATURE_SVE : NONE` → **SVE disabled** |
| SVE-named kernels present in binary | yes (compiled in, never dispatched) |
| Actual matmul path | `neon_i8mm` / `neon_dotprod` (NEON) |
| Bench: default vs KleidiAI | see `evidence/bench-*.txt` (regenerated per run) |

## One-command reproduce

```bash
# Any arm64 Linux (Neoverse-N2, Graviton4, Cobalt 100, or a $0 GitHub arm runner)
gh repo clone optimizedwf/arm-dispatch-audit
cd arm-dispatch-audit
# or just re-run the workflow:
gh workflow run neoverse-sve-dispatch-audit.yml
```

Or run the full audit locally on any arm64 host:

```bash
# in this repo:
.github/audit.sh   # builds llama.cpp default + KleidiAI, benches, dumps dispatch table
```

## Why this matters (Impact)

- **Every free Arm CI runner** (GitHub Actions `arm`, CodeQL, etc.) is Neoverse-N2-class — 128-bit SVE.
  Teams benchmark "KleidiAI on/off" on these and report NEON numbers, believing SVE is involved.
- AWS Graviton4, Azure Cobalt 100, and the coming Graviton5-class parts are 128-bit SVE2 (N2/V2 lineage).
- The 256-bit SVE path (Neoverse V1/Graviton3) is the *only* place KleidiAI's SVE kernels fire — a rare,
  expensive part. Optimizing for it optimizes for a minority of cloud fleets.
- Tooling fix: kernel selection should be `svcntb() >= 32` or SVE-width-parameterized, not `== 32`.

## Method (Tech Impl)

- Pinned llama.cpp `b10434` (git tag), KleidiAI `v1.24.0` (fetched by llama.cpp build), both built from source on the same runner.
- Same binary workload: Qwen2.5-1.5B-Instruct Q8_0, 512-token prompt, 128-token decode, 4 threads, 2 reps.
- Evidence artifacts: `evidence/kai-symbols.txt` (symbol table), objdump SVE/NEON register counts, `svcntb()` runtime probe.
- Honest negative results included (e.g., if KleidiAI does not beat default on N2, we say so).

## Files

```
.github/workflows/audit.yml   # the full audit pipeline (build + bench + dispatch + disasm)
evidence/                     # per-run artifacts (committed by workflow when run)
README.md
LICENSE                       # MIT
```

## Track: Cloud AI

Fits the Arm AI Optimization Challenge **Cloud AI** track: inference performance, frameworks, agents,
production-ready developer workflows on Arm64 — with measurable tok/s / TTFT evidence and one-command reproducibility.
