#!/data/data/com.termux/files/usr/bin/bash
# CPU A/B: SVE2 vs armv8.6+fp16, KleidiAI on/off. Short bursts, each test saved in ~/bench/<name>.txt
# Author: Alen Pozun (@termuxSolutions)
S=~/llama.cpp-src; L=~/llb; M=~/models/smollm2-360m-instruct-q8_0.gguf
mkdir -p $L ~/bench $PREFIX/tmp; cd $S || exit 1
cp -n build/bin/llama-bench $L/new-v86
cp -n build-max/bin/llama-bench $L/max-sve2-kleidi
for v in "v86fp16-kleidi ON" "v86fp16-nokleidi OFF"; do
  set -- $v
  [ -x $L/$1 ] && continue
  echo ">> building $1"
  cmake -B build-max -DGGML_CPU_ARM_ARCH=armv8.6-a+fp16 -DGGML_CPU_KLEIDIAI=$2 >/dev/null 2>&1
  cmake --build build-max -j 4 --target llama-bench >$PREFIX/tmp/b.log 2>&1 || { grep -m3 -E "error|FAILED" $PREFIX/tmp/b.log; continue; }
  cp build-max/bin/llama-bench $L/$1
done
for c in "6-8 3" "6-9 4"; do
  set -- $c
  for b in new-v86 max-sve2-kleidi v86fp16-kleidi v86fp16-nokleidi; do
    [ -x $L/$b ] || continue
    n=cpu-$b-${1/-/}
    sleep 30
    taskset -c $1 $L/$b -m $M -t $2 -ngl 0 -nopo 1 -fa 1 -p 256 -n 64 -r 1 2>/dev/null | grep -E "pp256|tg64" | awk -F'|' -v n=$n '{print n, $(NF-2), $(NF-1)}' | tee ~/bench/$n.txt
  done
done
