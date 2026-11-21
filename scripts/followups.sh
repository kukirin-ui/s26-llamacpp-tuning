#!/data/data/com.termux/files/usr/bin/bash
# Follow-up checks after fa_sweep.sh. Author: Alen Pozun (@termuxSolutions)
mkdir -p ~/bench/fa-sweep
M=~/models/smollm2-360m-instruct-q8_0.gguf
# 1) Do CPU cores change FA-on prefill at 4K? (3 / 4 / 10 threads, ub 64 and 1024)
for c in "6-8 3" "6-9 4" "0-9 10"; do set -- $c; sleep 30; echo ">> cores $1, $2 threads"
  taskset -c $1 llama-bench -m $M -t $2 -ngl 40 -fa 1 -ub 64,1024 -p 256 -n 0 -d 4096 -r 1 2>/dev/null | grep "pp256" | awk -F'|' '{print "   ub" $(NF-4) ":" $(NF-1)}'
done | tee ~/bench/fa-sweep/cores-ub.txt
# 2) Is it the model? Same test on Qwen2.5-Coder-0.5B
m=~/models/qwen2.5-coder-0.5b-instruct-q8_0.gguf; sleep 30
taskset -c 6-9 llama-bench -m "$m" -t 4 -ngl 40 -fa 0,1 -p 256 -n 0 -d 0,4096 -r 1 2>/dev/null | grep pp256 | awk -F'|' '{print "   fa" $(NF-3) " " $(NF-2) ":" $(NF-1)}' | tee ~/bench/fa-sweep/model-0.5b-69.txt
# 3) Synthetic tool-overhead prompts (same token counts as a 3090 / 27B test): 0/10/20/40 tools
for m in qwen2.5-coder-0.5b-instruct-q8_0 smollm2-360m-instruct-q8_0; do sleep 30
  taskset -c 6-8 llama-bench -m ~/models/$m.gguf -t 3 -ngl 40 -fa 0,1 -p 21,1081,1960,3718 -n 0 -r 1 2>/dev/null | grep -E "pp[0-9]+" | awk -F'|' '{print "fa" $(NF-3) " " $(NF-2) ":" $(NF-1)}' | tee ~/bench/fa-sweep/tools-prefill-$m.txt
done
# 4) Quantized KV cache with flash attention off (does llama.cpp allow it?)
M=~/models/smollm2-360m-instruct-q8_0.gguf
for b in llama-bench ~/llb/v86fp16-nokleidi; do for kv in "f16 f16" "q8_0 f16" "q8_0 q8_0" "q4_0 q4_0"; do set -- $kv; sleep 20
  r=$(taskset -c 6-8 $b -m $M -t 3 -ngl 40 -fa 0 -ctk $1 -ctv $2 -p 256 -n 64 -d 4096 -r 1 2>&1)
  echo ">> $(basename $b) K=$1 V=$2"; echo "$r" | grep -E "pp256|tg64" | awk -F'|' '{print "   " $(NF-2) ":" $(NF-1)}'; echo "$r" | grep -iE "error|requires" | head -2
done; done | tee ~/bench/fa-sweep/kv-quant-fa0.txt
