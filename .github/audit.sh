#!/usr/bin/env bash
# Neoverse SVE Dispatch Audit — one-command reproduce on any arm64 Linux
# Usage: bash .github/audit.sh [model_repo model_file]
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-b10434}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen2.5-1.5B-Instruct-Q8_0.gguf}"
NPROC=$(nproc)

echo "== CPU identity =="
uname -m
lscpu | grep -E 'Model name|Architecture|Flags' | head -4

echo "== svcntb() probe =="
cat > /tmp/svctest.c <<'EOF'
#include <stdio.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
int main(){ printf("svcntb() = %d bytes/vector\n", svcntb()); return 0; }
#else
int main(){ printf("no SVE compile support\n"); return 0; }
#endif
EOF
gcc -marrmv8.6-a+sve -o /tmp/svctest /tmp/svctest.c 2>/dev/null && /tmp/svctest || echo "svcntb probe unavailable"

echo "== build llama.cpp ${LLAMA_TAG} =="
sudo apt-get update -qq
sudo apt-get install -y -qq cmake ninja-build build-essential
git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build-default -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_NATIVE=OFF
cmake --build build-default -j"${NPROC}" --target llama-bench llama-cli
cmake -B build-kleidiai -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_NATIVE=OFF -DGGML_CPU_KLEIDIAI=ON
cmake --build build-kleidiai -j"${NPROC}" --target llama-bench llama-cli
cd ..

echo "== model =="
python3 -c "
from huggingface_hub import hf_hub_download
for f in ('${MODEL_FILE}', 'qwen2.5-1.5b-instruct-q4_k_m.gguf'):
    print(hf_hub_download('${MODEL_REPO}', f, local_dir='models'))
"

echo "== bench: default =="
./llama.cpp/build-default/bin/llama-bench -m models/${MODEL_FILE} -p 512 -n 128 -t ${NPROC} -r 2 | tee bench-default.txt

echo "== bench: kleidiai =="
./llama.cpp/build-kleidiai/bin/llama-bench -m models/${MODEL_FILE} -p 512 -n 128 -t ${NPROC} -r 2 | tee bench-kleidiai.txt

echo "== bench: default (Q4_K_M) =="
./llama.cpp/build-default/bin/llama-bench -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p 512 -n 128 -t ${NPROC} -r 2 | tee bench-default-q4.txt

echo "== bench: kleidiai (Q4_K_M) =="
./llama.cpp/build-kleidiai/bin/llama-bench -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p 512 -n 128 -t ${NPROC} -r 2 | tee bench-kleidiai-q4.txt

echo "== dispatch table =="
echo "SVE-named kernels in binary: $(nm ./llama.cpp/build-kleidiai/bin/llama-cli | grep -cE 'kai_.*sve')"
svc_obj=$(find ./llama.cpp/build-kleidiai -name '*sve_i8mm_asm.S.o' | head -1)
neon_obj=$(find ./llama.cpp/build-kleidiai -name '*16x4_neon_i8mm.c.o' | head -1)
[ -n "$svc_obj" ] && echo "SVE objdump sample:" && objdump -d "$svc_obj" | grep -E '\bz[0-9]+\b' | head -6
[ -n "$neon_obj" ] && echo "NEON objdump sample:" && objdump -d "$neon_obj" | grep -E '\bv[0-9]+\b' | head -6
echo "KleidiAI matmul symbols:"
nm ./llama.cpp/build-kleidiai/bin/llama-cli | grep kai_matmul | sed 's/.* T //' | sort -u | head -30
echo ""
echo "== verdict =="
echo "svcntb()==16 (128-bit) => CPU_FEATURE_SVE disabled (gate: svcntb()==32)"
echo "=> SVE kernels dead code; matmul runs neon_i8mm/neon_dotprod"
echo "Compare bench-default.txt vs bench-kleidiai.txt above."
