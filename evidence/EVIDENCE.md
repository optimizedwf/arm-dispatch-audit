# Evidence — Neoverse SVE Dispatch Audit

CI run: GitHub Actions `ubuntu-24.04-arm` · Neoverse-N2 · 4 vCPU · llama.cpp `b10434` · KleidiAI v1.24.0
Date: 2026-08-14

## 1. CPU identity

```
aarch64
Architecture: aarch64
CPU(s): 4
Model name: Neoverse-N2
Flags: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16
```

Note: `sve`, `sve2`, `svei8mm`, `svebf16` all advertised — yet (below) the SVE kernels never dispatch.

## 2. Runtime svcntb() probe (the gate)

```
svcntb() = 16 (bytes per SVE vector)
```

KleidiAI dispatch gate (llama.cpp master, `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`):

```c
ctx.features = (runtime_feat.has_dotprod ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
               (runtime_feat.has_i8mm    ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
               (runtime_feat.sve_cnt == QK8_0 ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);
```

`QK8_0 == 32` (ggml-common.h). `svcntb() == 16 != 32` → **CPU_FEATURE_SVE = NONE**.
All `*_sve_*` KleidiAI kernels are dead code on every 128-bit-SVE Neoverse (N2, Graviton4, Cobalt 100).

## 3. SVE kernels compiled in — never dispatched

SVE-named kernel objects present in build: **4962** (includes headers; key kernels):

```
kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.c.o      (SVE)
kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod_asm.S.o  (SVE)
kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.c.o        (SVE)
kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm_asm.S.o    (SVE)
```

Register-usage census of all KleidiAI objects (objdump):

```
SVE register instructions (all objects): 13644
NEON register instructions (all objects): 33871
```

13,644 SVE instructions are *in the binary*; the runtime path (below) uses NEON only.

## 4. Benchmark — default vs KleidiAI (Qwen2.5-1.5B Q8_0, 4 threads, 2 reps)

| build | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|
| default (no KleidiAI) | **187.06 ± 0.03** | **43.84 ± 0.21** |
| + KleidiAI (GGML_CPU_KLEIDIAI=ON) | 186.92 ± 0.06 | 42.33 ± 0.92 |

Δ = **−0.07% prefill / −3.4% decode** — KleidiAI is a no-op (or slightly worse) on 128-bit Neoverse.
Expected if the SVE path is dead: KleidiAI adds zero on 128-bit Neoverse (its NEON i8mm/dotprod kernels
are the same ones ggml already uses).

## 5. Conclusion

- On **128-bit-SVE Neoverse** (all free Arm CI runners, Graviton4, Cobalt 100), `GGML_CPU_KLEIDIAI=ON` is a no-op for matmul.
- The SVE2 flag in `/proc/cpuinfo` is a trap: the *only* Neoverse where KleidiAI SVE fires is 256-bit (V1/Graviton3).
- Fix: dispatch on `svcntb() >= 32` (SVE width-aware), or expose `GGML_KLEIDIAI_SVE_WIDTH` override.

Reproduce: `bash .github/audit.sh` (any arm64 Linux; ~15 min).
