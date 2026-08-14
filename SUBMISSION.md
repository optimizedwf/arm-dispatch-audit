# Submission: Arm AI Optimization Challenge — Cloud AI track

## Project: Neoverse SVE Dispatch Audit — "SVE2 advertised, never used"

### Project Overview
A forensic, reproducible audit of ARM KleidiAI's SVE kernel dispatch inside llama.cpp on
128-bit-SVE Neoverse silicon — the class of Arm64 cloud parts that powers every free Arm CI
runner (GitHub Actions `ubuntu-24.04-arm`), AWS Graviton4, and Azure Cobalt 100.

**Core finding:** KleidiAI gates its SVE matmul kernels on `runtime_feat.sve_cnt == QK8_0`
(i.e. `svcntb() == 32` bytes = 256-bit vectors). Neoverse-N2-class parts report `svcntb() == 16`
(128-bit SVE2), so `CPU_FEATURE_SVE` is never set and **13,644 SVE instructions compiled into the
binary are never dispatched** — the SVE2 advertised in `/proc/cpuinfo` is never used. Matmul
actually runs NEON `i8mm`/`dotprod` kernels.

Measured: default vs `-DGGML_CPU_KLEIDIAI=ON` on Neoverse-N2 → 187.22 vs 187.67 tok/s prefill
(+0.24%, noise) and 45.16 vs 44.94 tok/s decode (−0.49%, noise). KleidiAI is a **no-op** on
128-bit Neoverse. The SVE kernels only fire on 256-bit Neoverse V1 / Graviton3 — a shrinking
minority of cloud fleets.

### Why it should win
1. **Actionable for every Arm cloud developer:** tells you exactly why your KleidiAI build doesn't
   speed up on Graviton4/Cobalt 100 — a real, silent, and widely-misunderstood dispatch bug.
2. **Measurable, honest evidence:** two from-source builds, pinned versions, 2-rep benches with error
   bars, runtime `svcntb()` probe, and a full SVE/NEON instruction census. We report the honest
   negative result instead of cherry-picking.
3. **One-command reproducible on $0 infrastructure:** the entire audit runs on a free GitHub
   Actions arm64 runner (Neoverse-N2) or any arm64 Linux via `bash .github/audit.sh`.
4. **Fixes the tooling:** the dispatch gate should be width-aware (`svcntb() >= 32`), not `== 32`.
   This is a concrete, upstreamable improvement to llama.cpp/KleidiAI integration.

### Functionality / Output
- Full CI pipeline (`audit.yml`) that builds llama.cpp `b10434` twice (default / +KleidiAI v1.24.0),
  benchmarks Qwen2.5-1.5B Q8_0 (512-tok prompt, 128-tok decode, 4 threads, 2 reps), probes
  `svcntb()` live, and counts SVE vs NEON instructions across all KleidiAI objects.
- Evidence artifacts committed: `evidence/EVIDENCE.md`, `evidence/bench-*.txt`,
  `evidence/kai-symbols.txt`.
- Standalone reproduce: `bash .github/audit.sh`.

### Setup Instructions
```bash
# 1. Get a $0 Arm64 environment (or any arm64 Linux)
#    GitHub Actions: public repo, workflow ubuntu-24.04-arm (Neoverse-N2) — free
#    or: AWS Graviton4 / Azure Cobalt 100 / any arm64 box

# 2. Run the audit
gh repo clone optimizedwf/arm-dispatch-audit
cd arm-dispatch-audit
bash .github/audit.sh          # ~15 min, builds + benches + dumps dispatch table
# or trigger the CI pipeline:
gh workflow run neoverse-sve-dispatch-audit.yml
```

Expected output: `svcntb() = 16`, SVE-instruction census (13,644 SVE / 33,871 NEON),
and a bench table showing default ≈ KleidiAI (within noise).

### Public repo (MIT license)
https://github.com/optimizedwf/arm-dispatch-audit
