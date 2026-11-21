#!/data/data/com.termux/files/usr/bin/bash
# FA prefill sweep on Xclipse 960 (Galaxy S26, Exynos 2600)
# Alen Pozun / @termuxSolutions - Termux, no PC
#
# Question from X: does a smaller ubatch fix flash-attention prefill at depth,
# or is FA prefill on this GPU simply broken?
# Earlier result (stock build, ub 1024, 4K depth): FA on 78 tok/s prefill, FA off 734 tok/s.
#
# What this script tests (each run saved by name in ~/bench/fa-sweep/):
#   A. ubatch sweep, FA on, 4K depth: 64 / 128 / 256 / 512 / 1024
#   B. FA off reference at a small ubatch (256)
#   C. depth curve FA on vs off: 0 / 1K / 2K  -> does FA prefill collapse with depth?
#   D. newer llama.cpp Vulkan code (my custom build), FA on vs off at 4K
#   E. KV cache on CPU (-nkvo 1), FA on at 4K -> attention on CPU, matmuls on GPU
# At the end: summary + an auto-filled reply draft.
#
# Settings that stay fixed (my Meridian GPU config): cores 6-8, 3 threads, 40 GPU layers, KV f16.
# 1 repetition per run, 30 s cooldown between runs (phone thermals). About 12-15 minutes.

M=${M:-~/models/smollm2-360m-instruct-q8_0.gguf}
OLD=${OLD:-$(command -v llama-bench)}      # stock Termux build (fastest GPU prefill so far)
NEW=${NEW:-~/llb/v86fp16-nokleidi}         # my custom build, newer llama.cpp Vulkan code
OUT=~/bench/fa-sweep; mkdir -p "$OUT"
COOL=${COOL:-30}
eval M=$M; eval NEW=$NEW

run() {  # run <name> <binary> <llama-bench args...>
  local name=$1 bin=$2; shift 2
  [ -x "$bin" ] || { echo ">> $name: binary not found ($bin), skipped"; return; }
  sleep "$COOL"
  printf ">> %-20s " "$name"
  taskset -c 6-8 "$bin" -m "$M" -t 3 -ngl 40 -r 1 "$@" 2>/dev/null \
    | grep -E "\| *(pp|tg)[0-9]+" \
    | awk -F'|' '{t=$(NF-2); v=$(NF-1); gsub(/^ +| +$/,"",t); gsub(/^ +| +$/,"",v); split(v,a," "); printf "%s=%s  ", t, a[1]}' \
    | tee "$OUT/$name.txt"
  echo
}
val() {  # val <name> <test-prefix, e.g. 'pp256 @ d4096'> -> number
  grep -oE "$2=[0-9.]+" "$OUT/$1.txt" 2>/dev/null | head -1 | grep -oE '[0-9.]+$'
}

echo "FA prefill sweep  model=$(basename "$M")  old=$OLD  new=$NEW  cooldown=${COOL}s"

# A. ubatch sweep, FA on, 4K depth (prefill only)
for ub in 64 128 256 512 1024; do
  run "A-fa1-ub$ub-d4k" "$OLD" -fa 1 -ub $ub -p 256 -n 0 -d 4096
done
# B. FA off reference, small ubatch
run "B-fa0-ub256-d4k" "$OLD" -fa 0 -ub 256 -p 256 -n 0 -d 4096
# C. depth curve, FA on vs off (ub 1024)
run "C-fa1-depth" "$OLD" -fa 1 -ub 1024 -p 256 -n 0 -d 0,1024,2048
run "C-fa0-depth" "$OLD" -fa 0 -ub 1024 -p 256 -n 0 -d 0,1024,2048
# D. newer llama.cpp Vulkan code, 4K, prefill + generation
run "D-new-fa1-d4k" "$NEW" -fa 1 -ub 1024 -p 256 -n 64 -d 4096
run "D-new-fa0-d4k" "$NEW" -fa 0 -ub 1024 -p 256 -n 64 -d 4096
# E. KV cache on CPU, FA on, 4K
run "E-nkvo-fa1-d4k" "$OLD" -fa 1 -ub 1024 -nkvo 1 -p 256 -n 64 -d 4096

# ---------------------------------------------------------------- summary
S="$OUT/summary.txt"
{
echo "FA prefill sweep - Galaxy S26 / Xclipse 960 - SmolLM2-360M Q8_0 - $(date '+%Y-%m-%d %H:%M')"
echo "Prefill tok/s (pp256) unless noted. Cores 6-8, 3 threads, 40 GPU layers, KV f16, 1 rep."
echo
echo "A. FA on, 4K depth, by ubatch:"
for ub in 64 128 256 512 1024; do echo "   ub $ub: $(val A-fa1-ub$ub-d4k 'pp256 @ d4096')"; done
echo "B. FA off, 4K depth, ub 256: $(val B-fa0-ub256-d4k 'pp256 @ d4096')"
echo "C. depth curve (ub 1024):"
for d in 0 1024 2048; do
  t="pp256"; [ $d -gt 0 ] && t="pp256 @ d$d"
  echo "   depth $d: FA on $(val C-fa1-depth "$t")   FA off $(val C-fa0-depth "$t")"
done
echo "D. newer llama.cpp build, 4K: FA on pp $(val D-new-fa1-d4k 'pp256 @ d4096') / gen $(val D-new-fa1-d4k 'tg64 @ d4096')   FA off pp $(val D-new-fa0-d4k 'pp256 @ d4096') / gen $(val D-new-fa0-d4k 'tg64 @ d4096')"
echo "E. KV on CPU (-nkvo), FA on, 4K: pp $(val E-nkvo-fa1-d4k 'pp256 @ d4096') / gen $(val E-nkvo-fa1-d4k 'tg64 @ d4096')"
} | tee "$S"

# ---------------------------------------------------------------- auto verdict + reply draft
best_ub=""; best=0
for ub in 64 128 256 512 1024; do
  v=$(val A-fa1-ub$ub-d4k 'pp256 @ d4096'); [ -z "$v" ] && continue
  if awk "BEGIN{exit !($v > $best)}"; then best=$v; best_ub=$ub; fi
done
base=$(val A-fa1-ub1024-d4k 'pp256 @ d4096'); off=$(val B-fa0-ub256-d4k 'pp256 @ d4096')
if [ -n "$base" ] && [ -n "$best_ub" ]; then
  gain=$(awk "BEGIN{printf \"%.2f\", $best/$base}")
  if [ "$best_ub" = 1024 ]; then
    verdict="You were right: smaller ubatch does not help, ub 1024 is already the fastest (${base} tok/s)."
  elif awk "BEGIN{exit !($gain < 1.3)}"; then
    verdict="You were right: smaller ubatch barely moves it (best ${best} tok/s at ub ${best_ub}, x${gain} vs ub 1024)."
  else
    verdict="Smaller ubatch does help: ${best} tok/s at ub ${best_ub} vs ${base} at ub 1024 (x${gain})."
  fi
  [ -n "$off" ] && verdict="$verdict FA off is still ${off} tok/s, so FA prefill at depth looks broken on this GPU/driver."
  {
  echo
  echo "Reply draft (tag him when you post):"
  echo "Sweep is up. S26 / Xclipse 960, 360M Q8_0, 4K depth, FA on, prefill by ubatch:"
  printf "64 %s | 128 %s | 256 %s | 512 %s | 1024 %s tok/s\n" \
    "$(val A-fa1-ub64-d4k 'pp256 @ d4096')" "$(val A-fa1-ub128-d4k 'pp256 @ d4096')" "$(val A-fa1-ub256-d4k 'pp256 @ d4096')" \
    "$(val A-fa1-ub512-d4k 'pp256 @ d4096')" "$(val A-fa1-ub1024-d4k 'pp256 @ d4096')"
  echo "$verdict"
  } | tee -a "$S"
fi
echo; echo "Saved: $OUT"
