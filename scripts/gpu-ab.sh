#!/data/data/com.termux/files/usr/bin/bash
# Build llama-server/cli from build-max (armv8.6+fp16, no SVE, no KleidiAI), verify, then short GPU tests old vs new.
# Author: Alen Pozun (@termuxSolutions)
S=~/llama.cpp-src; L=~/llb; M=~/models/smollm2-360m-instruct-q8_0.gguf
mkdir -p $L ~/bench $PREFIX/tmp; cd $S || exit 1
grep -E "GGML_CPU_ARM_ARCH|GGML_CPU_KLEIDIAI" build-max/CMakeCache.txt
cmake --build build-max -j 4 --target llama-server llama-cli >$PREFIX/tmp/b.log 2>&1 || { grep -m3 -E "error|FAILED" $PREFIX/tmp/b.log; exit 1; }
cp build-max/bin/llama-server $L/v86fp16-nokleidi-server; cp build-max/bin/llama-cli $L/v86fp16-nokleidi-cli
timeout 10 $L/v86fp16-nokleidi-server -m $M --port 8099 -ngl 0 -lv 4 2>&1 | grep -m1 NEON
O=$(command -v llama-bench); N=$L/v86fp16-nokleidi
for run in "old $O 6-8 3" "old $O 6-9 4" "new $N 6-8 3" "new $N 6-9 4"; do
  set -- $run; n=gpu-$1-${3/-/}
  sleep 30
  taskset -c $3 $2 -m $M -t $4 -ngl 40 -fa 1 -ub 1024 -p 256 -n 64 -r 1 2>/dev/null | grep -E "pp256|tg64" | awk -F'|' -v n=$n '{print n, $(NF-2), $(NF-1)}' | tee ~/bench/$n.txt
done
