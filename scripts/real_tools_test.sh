#!/data/data/com.termux/files/usr/bin/bash
# Real MCP-style tool overhead test on Galaxy S26 (Termux).
# Renders 0/10/20/40 real JSON tool schemas through the model's own chat template
# (llama-server --jinja) and reads the true prompt token count and prefill time
# from the server's response. Prompt cache off, 2 runs per size.
# Author: Alen Pozun (@termuxSolutions)
M=${M:-~/models/qwen2.5-coder-0.5b-instruct-q8_0.gguf}
eval M=$M
PORT=8097
OUT=~/bench/fa-sweep/real-tools-$(basename "$M" .gguf).txt
mkdir -p ~/bench/fa-sweep
pkill -f "llama-server.*--port $PORT" 2>/dev/null
taskset -c 6-8 llama-server -m "$M" --port $PORT -ngl 40 -t 3 -c 8192 -fa off --jinja -np 1 >$PREFIX/tmp/rt-server.log 2>&1 &
for i in $(seq 1 60); do curl -s localhost:$PORT/health | grep -q ok && break; sleep 1; done
curl -s localhost:$PORT/health | grep -q ok || { echo "server did not start:"; tail -n 15 $PREFIX/tmp/rt-server.log; exit 1; }

python3 - "$PORT" "$M" <<'EOF' | tee "$OUT"
import json, sys, time, urllib.request
port, model = sys.argv[1], sys.argv[2]
def p(props, req): return {"type": "object", "properties": props, "required": req}
S = {"type": "string"}; I = {"type": "integer"}; B = {"type": "boolean"}
BASE = [
 ("read_file", "Read a text file from the workspace and return its content.", p({"path": S, "max_bytes": I}, ["path"])),
 ("write_file", "Write or overwrite a text file in the workspace.", p({"path": S, "content": S, "append": B}, ["path", "content"])),
 ("list_directory", "List files and folders in a directory, optionally recursive.", p({"path": S, "recursive": B}, ["path"])),
 ("search_files", "Search file contents for a regular expression.", p({"pattern": S, "path": S, "max_results": I}, ["pattern"])),
 ("git_status", "Show the working tree status of a git repository.", p({"repo_path": S}, ["repo_path"])),
 ("git_diff", "Show changes between the working tree and a commit.", p({"repo_path": S, "ref": S, "path": S}, ["repo_path"])),
 ("git_commit", "Create a commit with the staged changes.", p({"repo_path": S, "message": S}, ["repo_path", "message"])),
 ("git_log", "Show the commit history of a repository.", p({"repo_path": S, "max_count": I}, ["repo_path"])),
 ("web_search", "Search the web and return the top results with titles and URLs.", p({"query": S, "num_results": I}, ["query"])),
 ("fetch_url", "Fetch a web page and return it as plain text.", p({"url": S, "max_chars": I}, ["url"])),
]
def tools(n):
    out = []
    for i in range(n):
        name, desc, params = BASE[i % 10]
        out.append({"type": "function", "function": {"name": name if i < 10 else f"{name}_{i // 10 + 1}", "description": desc, "parameters": params}})
    return out
print(f"Real tool schemas through the chat template - {model.split('/')[-1]} - llama-server --jinja, -fa off, cache off")
print(f"{'tools':>5} {'prompt_tokens':>14} {'prefill_ms':>11} {'tok/s':>8}  run")
base = None
for n in (0, 10, 20, 40):
    for run in (1, 2):
        body = {"messages": [{"role": "user", "content": "What changed in my repo since yesterday?"}], "max_tokens": 1, "cache_prompt": False}
        if n: body["tools"] = tools(n)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        t = d.get("timings", {}); pt = d.get("usage", {}).get("prompt_tokens") or t.get("prompt_n")
        ms = t.get("prompt_ms", 0)
        print(f"{n:>5} {pt:>14} {ms:>11.0f} {t.get('prompt_per_second', 0):>8.0f}  {run}")
        if n == 0 and run == 1: base = pt
        if n == 40 and run == 2 and base: print(f"\n~{(pt - base) / 40:.0f} tokens per tool on this template (40 tools: {pt} tokens)")
        time.sleep(3)
EOF
pkill -f "llama-server.*--port $PORT"
echo "Saved: $OUT"
