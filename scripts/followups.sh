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
