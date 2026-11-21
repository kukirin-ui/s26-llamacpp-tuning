#!/data/data/com.termux/files/usr/bin/bash
# Max-feature llama.cpp build (SVE2/i8mm/bf16 + KleidiAI + Vulkan int dot, Release),
# benchmark vs old Termux package and current build, remove new builds that are slower.
# Author: Alen Pozun (@termuxSolutions). Runs in Termux on Galaxy S26.
SRC=~/llama.cpp-src
M=~/models/smollm2-360m-instruct-q8_0.gguf
OUT=~/bench-results.txt
T=$PREFIX/tmp; mkdir -p $T
cd "$SRC" || exit 1
pkg install -y git cmake clang ninja shaderc vulkan-headers spirv-headers >/dev/null 2>&1

COMMON="-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DVulkan_LIBRARY=/system/lib64/libvulkan.so -DVulkan_GLSLC_EXECUTABLE=$PREFIX/bin/glslc"
TRIES=("armv9.2-a+sve2+i8mm+bf16+dotprod+fp16 ON" "armv9.2-a+sve2+i8mm+bf16+dotprod+fp16 OFF" "armv8.6-a+fp16 ON")
OK=""
for t in "${TRIES[@]}"; do
  set -- $t
  echo ">> trying arch=$1 kleidiai=$2"
  rm -rf build-max
  cmake -B build-max $COMMON -DGGML_CPU_ARM_ARCH=$1 -DGGML_CPU_KLEIDIAI=$2 >$T/cfg.log 2>&1 || { tail -n 5 $T/cfg.log; continue; }
  if cmake --build build-max -j 4 --target llama-bench llama-server llama-cli >$T/build.log 2>&1; then OK="$1 kleidiai=$2"; break; fi
  grep -E -m3 "error|FAILED" $T/build.log
done
[ -z "$OK" ] && { echo "All build configs failed, see $T/build.log"; exit 1; }
echo ">> built: $OK"

bench() {
  sleep 60
  taskset -c 6-8 $1 -m $M -t 3 -fa 1 -ub 1024 -ngl 0 -nopo 1 -p 512 -n 128 -r 3 -o jsonl 2>/dev/null | sed "s/^/$2 cpu /"
  sleep 60
  taskset -c 6-8 $1 -m $M -t 3 -fa 1 -ub 1024 -ngl 40 -p 512 -n 128 -d 0,8064 -r 3 -o jsonl 2>/dev/null | sed "s/^/$2 gpu /"
}
: > $T/raw.txt
for x in "old:$(command -v llama-bench)" "build:$SRC/build/bin/llama-bench" "max:$SRC/build-max/bin/llama-bench"; do
  l=${x%%:*}; b=${x#*:}
  [ -x "$b" ] || continue
  echo ">> benchmarking $l (cooldown 60s per pass)"
  bench "$b" "$l" >> $T/raw.txt
done

awk -v sc="$T/score.txt" 'function f(k){ if(match($0,"\"" k "\": *[0-9.]+")){s=substr($0,RSTART,RLENGTH); sub(/.*: */,"",s); return s} return 0 }
{ l=$1; m=$2; p=f("n_prompt"); g=f("n_gen"); d=f("n_depth"); ts=f("avg_ts")
  k=m " " (g>0 ? "tg" g : "pp" p) " @d" d; v[l","k]=ts; K[k]=1; L[l]=1 }
END{ for(k in K){ line=k; for(l in L) if((l","k) in v) line=line "  " l "=" sprintf("%.1f",v[l","k]); print line }
     for(l in L) if(l!="old"){ s=0;n=0; for(k in K) if((l","k) in v && ("old,"k) in v){ s+=log(v[l","k]/v["old,"k]); n++ }
       if(n) printf "%s %.3f\n", l, exp(s/n) > sc } }' $T/raw.txt | sort | tee $OUT

echo ">> score vs old (1.000 = same):"
while read l s; do
  dir=$([ "$l" = max ] && echo build-max || echo build)
  if awk "BEGIN{exit !($s < 1.0)}"; then echo "$l x$s slower -> removing $SRC/$dir"; rm -rf "$SRC/$dir"
  else echo "$l x$s keep"; fi
done < $T/score.txt | tee -a $OUT
echo ">> built with: $OK" | tee -a $OUT
