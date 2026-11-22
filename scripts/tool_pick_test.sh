#!/data/data/com.termux/files/usr/bin/bash
# Tool-call accuracy with many tools loaded, on Galaxy S26 (Termux).
# 20 prompts, each needs one specific tool with specific arguments.
# Run 1: only the 10 needed tools. Run 2: the same 10 + 30 distractors (40 total),
# including near-misses like delete_file, git_branch, download_file, http_request.
# Scored per prompt: called a tool / right tool / right arguments.
# Uses the model's own chat template (llama-server --jinja), temperature 0, FA off.
# Author: Alen Pozun (@termuxSolutions)
M=${M:-~/models/qwen2.5-coder-0.5b-instruct-q8_0.gguf}
eval M=$M
PORT=${PORT:-8098}
OUT=~/bench/fa-sweep/tool-pick-$(basename "$M" .gguf)${SHARP:+-sharp}${FETCH_FIRST:+-fetchfirst}${ONLY_URL:+-url}-cache${CACHE:-1}.txt
mkdir -p ~/bench/fa-sweep
if [ -z "$NO_SERVER" ]; then
  pkill -f "llama-server.*--port $PORT" 2>/dev/null
  taskset -c 6-8 llama-server -m "$M" --port $PORT -ngl 40 -t 3 -c 8192 -fa off --jinja -np 1 >$PREFIX/tmp/tp-server.log 2>&1 &
  for i in $(seq 1 60); do curl -s localhost:$PORT/health | grep -q ok && break; sleep 1; done
  curl -s localhost:$PORT/health | grep -q ok || { echo "server did not start:"; tail -n 15 $PREFIX/tmp/tp-server.log; exit 1; }
fi

python3 - "$PORT" "$M" <<'EOF' | tee "$OUT"
import json, re, sys, time, urllib.request
def from_text(txt):
    """Fallback: model wrote the call as JSON text instead of a native tool call."""
    for blob in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", txt or "", re.S) + re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", txt or "", re.S) + [txt or ""]:
        try:
            o = json.loads(blob.strip())
            if isinstance(o, dict) and "name" in o:
                a = o.get("arguments", o.get("parameters", {}))
                return o["name"], (json.loads(a) if isinstance(a, str) else a) or {}
        except Exception:
            pass
    return None
port, model = sys.argv[1], sys.argv[2]
S = {"type": "string"}; I = {"type": "integer"}; B = {"type": "boolean"}
def t(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}
NEEDED = [
 t("read_file", "Read a text file from the workspace and return its content.", {"path": S, "max_bytes": I}, ["path"]),
 t("write_file", "Write or overwrite a text file in the workspace.", {"path": S, "content": S, "append": B}, ["path", "content"]),
 t("list_directory", "List files and folders in a directory, optionally recursive.", {"path": S, "recursive": B}, ["path"]),
 t("search_files", "Search file contents for a regular expression.", {"pattern": S, "path": S, "max_results": I}, ["pattern"]),
 t("git_status", "Show the working tree status of a git repository.", {"repo_path": S}, ["repo_path"]),
 t("git_diff", "Show changes between the working tree and a commit.", {"repo_path": S, "ref": S, "path": S}, ["repo_path"]),
 t("git_commit", "Create a commit with the staged changes.", {"repo_path": S, "message": S}, ["repo_path", "message"]),
 t("git_log", "Show the commit history of a repository.", {"repo_path": S, "max_count": I}, ["repo_path"]),
 t("web_search", "Search the web and return the top results with titles and URLs.", {"query": S, "num_results": I}, ["query"]),
 t("fetch_url", "Fetch a web page and return it as plain text.", {"url": S, "max_chars": I}, ["url"]),
]
DISTRACTORS = [
 t("delete_file", "Delete a file from the workspace.", {"path": S}, ["path"]),
 t("move_file", "Move or rename a file in the workspace.", {"src": S, "dst": S}, ["src", "dst"]),
 t("file_info", "Return size and modification time of a file.", {"path": S}, ["path"]),
 t("git_branch", "List, create or delete branches in a repository.", {"repo_path": S, "name": S, "delete": B}, ["repo_path"]),
 t("git_checkout", "Switch to a branch or commit.", {"repo_path": S, "ref": S}, ["repo_path", "ref"]),
 t("git_push", "Push commits to the remote repository.", {"repo_path": S, "remote": S, "branch": S}, ["repo_path"]),
 t("download_file", "Download a file from a URL into the workspace.", {"url": S, "path": S}, ["url", "path"]),
 t("http_request", "Send an HTTP request (GET/POST/PUT/DELETE) and return status and body.", {"method": S, "url": S, "body": S}, ["method", "url"]),
 t("dns_lookup", "Resolve a hostname to IP addresses.", {"host": S}, ["host"]),
 t("ping_host", "Ping a host and report latency.", {"host": S, "count": I}, ["host"]),
 t("run_shell", "Run a shell command in Termux and return its output.", {"command": S, "timeout_s": I}, ["command"]),
 t("get_battery", "Return the phone's battery level and charging state.", {}, []),
 t("get_memory", "Return total and available RAM.", {}, []),
 t("get_storage", "Return free and total storage.", {}, []),
 t("list_processes", "List running processes with CPU and memory use.", {"limit": I}, []),
 t("kill_process", "Stop a running process by PID.", {"pid": I}, ["pid"]),
 t("sqlite_query", "Run an SQL query on a local SQLite database.", {"db_path": S, "sql": S}, ["db_path", "sql"]),
 t("csv_summary", "Summarize the columns of a CSV file.", {"path": S}, ["path"]),
 t("json_validate", "Check whether a file contains valid JSON.", {"path": S}, ["path"]),
 t("send_email", "Send an email.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
 t("send_sms", "Send an SMS message.", {"to": S, "text": S}, ["to", "text"]),
 t("create_calendar_event", "Create a calendar event.", {"title": S, "start": S, "end": S}, ["title", "start"]),
 t("set_alarm", "Set an alarm on the phone.", {"time": S, "label": S}, ["time"]),
 t("set_timer", "Start a countdown timer.", {"seconds": I}, ["seconds"]),
 t("get_weather", "Get the current weather for a city.", {"city": S}, ["city"]),
 t("translate_text", "Translate text into another language.", {"text": S, "target_lang": S}, ["text", "target_lang"]),
 t("calculator", "Evaluate an arithmetic expression.", {"expression": S}, ["expression"]),
 t("unit_convert", "Convert a value between units.", {"value": {"type": "number"}, "from_unit": S, "to_unit": S}, ["value", "from_unit", "to_unit"]),
 t("get_time", "Return the current date and time.", {"timezone": S}, []),
 t("take_note", "Save a short note in the notes app.", {"text": S}, ["text"]),
]
import os
if os.environ.get("SHARP"):
    SHARP_DESC = {
        "read_file": "Read a LOCAL file from the phone's workspace by its file path. Never use for web addresses or URLs.",
        "fetch_url": "Open a web page by its http:// or https:// URL and return the page text. Use this for any link or website.",
        "http_request": "Low-level API call with a method (GET/POST/PUT/DELETE) for REST APIs. Not for reading web pages.",
        "web_search": "Search the internet when the user wants to look something up online and has no URL.",
        "search_files": "Search inside LOCAL files on the phone for a text pattern. Not for the internet.",
    }
    for tool in NEEDED + DISTRACTORS:
        f = tool["function"]
        if f["name"] in SHARP_DESC: f["description"] = SHARP_DESC[f["name"]]
if os.environ.get("FETCH_FIRST"):
    NEEDED.sort(key=lambda t: t["function"]["name"] != "fetch_url")  # fetch_url moved to slot 1
has = lambda a, k, sub: sub.lower() in str(a.get(k, "")).lower()
CASES = [
 ("Show me what's in notes/todo.txt", "read_file", lambda a: has(a, "path", "notes/todo.txt")),
 ("Open config.yaml and show me its contents.", "read_file", lambda a: has(a, "path", "config.yaml")),
 ("Save the text 'hello world' to out/hello.txt", "write_file", lambda a: has(a, "path", "out/hello.txt") and has(a, "content", "hello world")),
 ("Create a file named log.md containing '# Log'", "write_file", lambda a: has(a, "path", "log.md") and has(a, "content", "# Log")),
 ("What files are in the src folder?", "list_directory", lambda a: has(a, "path", "src")),
 ("List everything under projects, including subfolders.", "list_directory", lambda a: has(a, "path", "projects") and a.get("recursive") in (True, "true")),
 ("Find every place where TODO appears in the code.", "search_files", lambda a: has(a, "pattern", "TODO")),
 ("Search for 'api_key' inside the config directory.", "search_files", lambda a: has(a, "pattern", "api_key") and has(a, "path", "config")),
 ("What's the git status of the repo at ~/meridian?", "git_status", lambda a: has(a, "repo_path", "meridian")),
 ("Are there uncommitted changes in the ~/app repository?", "git_status", lambda a: has(a, "repo_path", "app")),
 ("Show me the diff of ~/app against HEAD~1.", "git_diff", lambda a: has(a, "repo_path", "app") and has(a, "ref", "HEAD~1")),
 ("What changed in README.md in the ~/app repo?", "git_diff", lambda a: has(a, "repo_path", "app") and has(a, "path", "README.md")),
 ("Commit the staged changes in ~/app with the message 'fix login bug'.", "git_commit", lambda a: has(a, "repo_path", "app") and has(a, "message", "fix login bug")),
 ("Commit what's staged in ~/meridian, message: update docs", "git_commit", lambda a: has(a, "repo_path", "meridian") and has(a, "message", "update docs")),
 ("Show the last 5 commits in ~/proj.", "git_log", lambda a: has(a, "repo_path", "proj") and str(a.get("max_count")) == "5"),
 ("Who made the most recent commit in ~/app?", "git_log", lambda a: has(a, "repo_path", "app")),
 ("Search the web for llama.cpp vulkan flash attention.", "web_search", lambda a: has(a, "query", "vulkan")),
 ("Look up the latest Termux release online.", "web_search", lambda a: has(a, "query", "termux")),
 ("Get the text of https://example.com/docs", "fetch_url", lambda a: has(a, "url", "example.com/docs")),
 ("Read this page for me: https://news.ycombinator.com", "fetch_url", lambda a: has(a, "url", "news.ycombinator.com")),
]
URL_CASES = [
 ("Get the text of https://example.com/docs", "fetch_url", lambda a: has(a, "url", "example.com/docs")),
 ("Read this page for me: https://news.ycombinator.com", "fetch_url", lambda a: has(a, "url", "news.ycombinator.com")),
 ("Open https://github.com/ggml-org/llama.cpp and tell me what it says.", "fetch_url", lambda a: has(a, "url", "github.com/ggml-org/llama.cpp")),
 ("What's on https://termux.dev/en/ right now?", "fetch_url", lambda a: has(a, "url", "termux.dev")),
 ("Summarize this article: https://www.bbc.com/news/technology", "fetch_url", lambda a: has(a, "url", "bbc.com/news/technology")),
 ("Fetch https://raw.githubusercontent.com/ggml-org/llama.cpp/master/README.md", "fetch_url", lambda a: has(a, "url", "raw.githubusercontent.com")),
 ("Can you load https://docs.python.org/3/ for me?", "fetch_url", lambda a: has(a, "url", "docs.python.org")),
 ("Pull the content of https://huggingface.co/Qwen", "fetch_url", lambda a: has(a, "url", "huggingface.co/Qwen")),
 ("Check what this link says: https://en.wikipedia.org/wiki/Android_(operating_system)", "fetch_url", lambda a: has(a, "url", "wikipedia.org")),
 ("Give me the text from https://example.org", "fetch_url", lambda a: has(a, "url", "example.org")),
]
if os.environ.get("ONLY_URL"):
    CASES = URL_CASES
SYSTEM = "You are an assistant with tools. When a request needs a tool, call exactly one tool with the correct arguments."
def ask(prompt, tools):
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "tools": tools, "tool_choice": "auto", "temperature": 0, "max_tokens": 256,
            "cache_prompt": os.environ.get("CACHE", "1") != "0"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return d, time.time() - t0

print(f"Tool-call accuracy - {model.split('/')[-1]} - llama-server --jinja, temperature 0, FA off - descriptions: {'sharpened' if os.environ.get('SHARP') else 'original'}, fetch_url {'in slot 1' if os.environ.get('FETCH_FIRST') else 'in slot 10'}, prompts: {'10 URL prompts' if os.environ.get('ONLY_URL') else 'all 20'}, prompt cache {'off' if os.environ.get('CACHE') == '0' else 'on'}")
summary = []
for label, tools in (("10 tools", NEEDED), ("40 tools", NEEDED + DISTRACTORS)):
    called = right = args_ok = native = 0; secs = []; misses = []; tim = []
    print(f"\n== {label} ==")
    for prompt, want, check in CASES:
        try:
            d, dt = ask(prompt, tools)
        except Exception as e:
            print(f"ERR  {prompt[:50]} | {e}"); misses.append(f"{want}: request error"); continue
        secs.append(dt)
        tm = d.get("timings") or {}
        tim.append((tm.get("prompt_n", 0), tm.get("prompt_ms", 0.0), tm.get("predicted_n", 0), tm.get("predicted_ms", 0.0)))
        msg = d["choices"][0]["message"]; tcs = msg.get("tool_calls") or []
        if tcs:
            native += 1; kind = "native"
            name = tcs[0]["function"]["name"]
            try: args = json.loads(tcs[0]["function"].get("arguments") or "{}")
            except Exception: args = {}
        else:
            got = from_text(msg.get("content"))
            if not got:
                print(f"NONE {dt:5.1f}s  want {want:<14} | text: {(msg.get('content') or '')[:60]!r}")
                misses.append(f"{want}: no tool call"); continue
            name, args = got; kind = "text"
        called += 1
        if name != want:
            print(f"WRONG {dt:4.1f}s  [{kind}] want {want:<14} got {name} {json.dumps(args)[:50]}")
            misses.append(f"{want} -> {name}"); continue
        right += 1
        ok = bool(check(args)); args_ok += ok
        print(f"{'PASS' if ok else 'ARGS'} {dt:5.1f}s  [{kind}] {want:<14} {json.dumps(args)[:60]}")
        if not ok: misses.append(f"{want}: bad args {json.dumps(args)[:60]}")
    n = len(CASES)
    line = f"{label}: native tool_calls {native}/{n}, any call (incl. JSON in text) {called}/{n}, right tool {right}/{n}, right tool + args {args_ok}/{n} ({100*args_ok/n:.0f}%), median {sorted(secs)[len(secs)//2] if secs else 0:.1f} s/request"
    if tim:
        md = lambda i: sorted(x[i] for x in tim)[len(tim) // 2]
        dec_rate = sum(x[2] for x in tim) / max(1e-9, sum(x[3] for x in tim) / 1000)
        line += (f"\n      timing (median): prompt {md(0)} tok in {md(1):.0f} ms | output {md(2)} tok in {md(3):.0f} ms"
                 f" | decode {dec_rate:.0f} tok/s | prefill share {100*sum(x[1] for x in tim)/max(1e-9, sum(x[1]+x[3] for x in tim)):.0f}%")
    summary.append(line)
    print(line)
    for m in misses: print("   miss:", m)
print("\nSUMMARY")
for s in summary: print(" ", s)
EOF
[ -z "$NO_SERVER" ] && pkill -f "llama-server.*--port $PORT"
echo "Saved: $OUT"
