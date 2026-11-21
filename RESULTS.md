# llama.cpp tuning on Galaxy S26 (Exynos 2600, Xclipse 960) — results

Author: Alen Požun (@termuxSolutions) · measured 21 Nov 2026 in Termux, no root, no PC.
Model unless noted: SmolLM2-360M-Instruct Q8_0 (367 MiB). KV cache f16. GPU = Vulkan, Samsung proprietary driver.
pp = prompt processing (prefill) tok/s, tg = generation tok/s. Most points are 1 repetition with 30–60 s cooldown; phone thermals add about ±10%.

## Device / driver facts
- Vulkan: `Samsung Xclipse 960 (Samsung Proprietary driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | int dot: 1 | matrix cores: none` (custom build). The stock Termux package reported `int dot: 0`.
- The driver exposes `VK_KHR_shader_integer_dot_product`; cooperative matrix ("matrix cores") is not used.
- CPU features of the stock Termux build: NEON, ARM_FMA, LLAMAFILE, REPACK only (no dotprod / i8mm / fp16).
- SVE vector length is 128-bit (`SVE_CNT = 16`).
- clang 21.1.8 crashes (exit 139) compiling ggml with `+sme`.

## 1. Context depth, stock build, GPU (cores 6–8, -t 3, -ngl 40, FA on)
| depth | tg tok/s |
|---|---|
| 0 | 109.9 |
| 4K | 76.1 (−31%) |
| 8K | 60.2 (−45%) |
KV cache q8_0 instead of f16: 101.8 (empty) / 59.5 (8K) — no gain.

## 2. Build comparison (pp512 / tg128, 3 reps, cores 6–8, -t 3)
| test | stock (old) | custom armv8.6 | custom armv9.2+SVE2+KleidiAI |
|---|---|---|---|
| CPU pp512 | 92.9 | 265.2 | 149.7 |
| CPU tg128 | 51.8 | 96.9 | 95.2 |
| GPU pp512 @0 | 2406.3 | 1865.7 | 1869.5 |
| GPU pp512 @8K | 41.9 | 33.8 | 35.7 |
| GPU tg128 @0 | 108.7 | 106.6 | 106.1 |
| GPU tg128 @8K | 53.2 | 45.2 | 60.8 |

## 3. CPU A/B: SVE2 vs KleidiAI (pp256 / tg64, -ngl 0)
| build | cores 6–8, -t 3 | cores 6–9, -t 4 |
|---|---|---|
| armv8.6 (RelWithDebInfo) | 279.1 / 107.5 | 464.6 / 120.5 |
| armv9.2 + SVE2 + KleidiAI | 191.9 / 101.8 | 273.0 / 115.4 |
| armv8.6+fp16 + KleidiAI (Release) | 278.0 / 105.9 | 462.7 / 121.8 |
| armv8.6+fp16, no KleidiAI (Release) | 279.3 / 107.2 | 464.6 / 121.2 |
Findings: KleidiAI makes no measurable difference on Q8_0; SVE2 (128-bit) costs −31…−41% prompt speed; adding core 9 (boost core) gives +66% prompt and +13% generation on CPU.

## 4. GPU, stock vs custom (pp256 / tg64, FA on, -ngl 40)
| build / cores | pp256 | tg64 |
|---|---|---|
| stock, 6–8 -t 3 | 2376.1 | 109.4 |
| stock, 6–9 -t 4 | 2187.0 | 109.3 |
| custom, 6–8 -t 3 | 1727.9 | 108.5 |
| custom, 6–9 -t 4 | 1720.5 | 109.5 |

## 5. 4K context depth: CPU vs GPU
| setup | pp256 @4K | tg64 @4K |
|---|---|---|
| CPU custom, cores 6–9 -t 4 | 99.9 | 27.6 |
| GPU stock, 6–8 -t 3, FA on | 78.0 | 79.5 |
| GPU stock, 6–8 -t 3, FA off | 733.8 | 60.2 |

## 6. Flash-attention prefill sweep (fa_sweep.sh, stock build, cores 6–8, -t 3)
FA on, 4K depth, by ubatch: 64 → 87.2 · 128 → 78.2 · 256 → 77.4 · 512 → 75.2 · 1024 → 82.0
FA off, 4K depth, ub 256: 803.6

Depth curve (ub 1024), pp256:
| depth | FA on | FA off |
|---|---|---|
| 0 | 2297.9 | 2209.8 |
| 1K | 267.8 | 526.7 (likely a noisy point) |
| 2K | 147.0 | 1118.0 |
| 4K | ~82 | ~804 |

Newer llama.cpp (custom build) at 4K: FA on pp 76.2 / tg 74.9 · FA off pp 705.5 / tg 61.3
KV cache on CPU (-nkvo 1), FA on, 4K: pp 72.9 / tg 22.3

## 7. Follow-ups: cores and model
FA on, 4K, pp256 by cores/threads: 6–8 -t 3: ub64 86.8 / ub1024 77.3 · 6–9 -t 4: 89.3 / 81.0 · 0–9 -t 10: 86.3 / 76.6

Qwen2.5-Coder-0.5B Q8_0, pp256:
| setup | FA off @0 | FA off @4K | FA on @0 | FA on @4K |
|---|---|---|---|---|
| cores 6–8 -t 3, -ngl 99 | 2086.7 | 1001.4 | 2146.7 | 111.4 |
| cores 6–9 -t 4, -ngl 40 | 1926.0 | 955.5 | 2139.5 | 103.1 |

Conclusion: FA-on prefill drops ~9–10× at 4K on both models, independent of ubatch, CPU threads, llama.cpp version and KV placement. FA off is the fix for long context on this GPU/driver (cost: about −24% generation at 4K).

## Recommended settings on this phone
- Short chats: CPU, custom build (armv8.6-a+fp16, no SVE, no KleidiAI), cores 6–9, -t 4 → 121 tok/s generation.
- Long chats / pasted documents: GPU, stock build, -ngl 40 -fa 0, cores 6–8, -t 3 → ~800 tok/s prompt at 4K, ~60 tok/s generation.
- KV cache f16.

## 8. Tool-definition overhead, synthetic prompts (same token counts), cold prefill
These are synthetic prompts (llama-bench), not real tool schemas: only the prompt sizes match an RTX 3090 / Qwen 27B Q4_K_M measurement posted by @MythomancerX (0/10/20/40 tools = 21/1,081/1,960/3,718 tokens). Cores 6–8, -t 3, -ngl 40, prompt cache off. Seconds of prefill = tokens / pp tok/s.
| tools | tokens | 3090 27B (his) | Qwen2.5-0.5B FA off | SmolLM2 FA off | Qwen2.5-0.5B FA on | SmolLM2 FA on |
|---|---|---|---|---|---|---|
| 0 | 21 | 0.17 s | 0.04 s (546 tok/s) | 0.03 s (654) | 0.03 s (608) | 0.03 s (778) |
| 10 | 1,081 | 1.1 s | 0.57 s (1,894) | 0.59 s (1,837) | 1.36 s (797) | 1.69 s (641) |
| 20 | 1,960 | 1.7 s | 1.10 s (1,788) | 1.19 s (1,643) | 4.35 s (450) | 5.60 s (350) |
| 40 | 3,718 | 3.0 s | 2.48 s (1,499) | 2.87 s (1,297) | 14.38 s (259) | 19.98 s (186) |
SmolLM2-360M (32 layers × 15 heads, 5 KV heads) does ~1.4× more attention work than Qwen2.5-0.5B (24 × 14, 2 KV heads); the FA-on gap (259 vs 186 tok/s) matches that ratio.

## 9. Real tool schemas through the chat template (real_tools_test.sh)
40 real JSON tool schemas (file, git, web; 10 distinct, repeated to 40), rendered by the model's own chat template in llama-server (--jinja), prompt cache off, FA off, cores 6–8, -t 3, -ngl 40, 2 runs each. Token counts come from the server response. Schemas are my own, not identical to the 3090 test, so this is not 1:1.

Qwen2.5-Coder-0.5B Q8_0:
| tools | prompt tokens | prefill ms (run 1 / run 2) | tok/s (run 1 / run 2) |
|---|---|---|---|
| 0 | 37 | 298 / 68 | 124 / 547 |
| 10 | 842 | 470 / 476 | 1,792 / 1,770 |
| 20 | 1,590 | 1,252 / 911 | 1,270 / 1,746 |
| 40 | 3,086 | 2,273 / 2,026 | 1,358 / 1,523 |
About 76 tokens per tool on this template (the 3090 test reported ~87 per tool on its template and schemas). The first 0-tool run includes warm-up.

## 10. Quantized KV cache with flash attention off (SmolLM2-360M, 4K depth, pp256 / tg64)
| build | K / V | pp256 | tg64 |
|---|---|---|---|
| stock | f16 / f16 | 772.6 | 60.0 |
| stock | q8_0 / f16 | 137.2 | 13.2 |
| stock | q8_0 / q8_0 | refused | refused |
| stock | q4_0 / q4_0 | refused | refused |
| custom | f16 / f16 | 694.1 | 60.5 |
| custom | q8_0 / f16 | 246.9 | 10.5 |
| custom | q8_0 / q8_0 | refused | refused |
| custom | q4_0 / q4_0 | refused | refused |
Refusal message: "quantized V cache requires flash_attn to be enabled". K-only quantization loads without FA but is 3–6× slower on this device. On Xclipse the practical choice is FA (slow prefill at depth) or f16 KV (more memory).

## 11. Tool-call accuracy with 10 vs 40 tools loaded (tool_pick_test.sh)
20 prompts, each needing one specific tool with specific arguments (file, git, web). Run 1 loads only the 10 needed tools; run 2 adds 30 distractors (40 total), including near-misses like delete_file, git_branch, download_file, http_request. Scored per prompt: native tool call / right tool / right arguments. llama-server --jinja, temperature 0, FA off, cores 6–8, -t 3, -ngl 40, 1 run.

| model | descriptions | tools | native tool calls | right tool | right tool + args | median s/request |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B Q8_0 | original | 10 | 0/20 | 10/20 | 10/20 (50%) | 0.5 |
| Qwen2.5-Coder-0.5B Q8_0 | original | 40 | 0/20 | 8/20 | 8/20 (40%) | 0.7 |
| Qwen2.5-0.5B-Instruct Q8_0 | original | 10 | 20/20 | 15/20 | 14/20 (70%) | 0.7 |
| Qwen2.5-0.5B-Instruct Q8_0 | original | 40 | 20/20 | 13/20 | 12/20 (60%) | 1.0 |
| Qwen2.5-0.5B-Instruct Q8_0 | sharpened (SHARP=1) | 10 | 20/20 | 15/20 | 15/20 (75%) | 0.6 |
| Qwen2.5-0.5B-Instruct Q8_0 | sharpened (SHARP=1) | 40 | 20/20 | 13/20 | 12/20 (60%) | 0.7 |

Notes:
- The Coder model never emits native tool calls; it writes the call as JSON in plain text (counted via a fallback parser for the "right tool" columns). A real agent without that parser would get 0 calls.
- Typical misses: fetch_url → read_file / web_search (0/2 in every run, even with "never use for URLs" in read_file's description), git_log → git_commit, and with 40 tools picking distractors (take_note, create_calendar_event, http_request, get_weather).
- Sharpening the descriptions moved 10 tools from 14 to 15/20 and left 40 tools at 12/20: at 0.5B the model size is the limit, not the wording.
- Run: `M=~/models/qwen2.5-0.5b-instruct-q8_0.gguf bash scripts/tool_pick_test.sh` (add `SHARP=1` for sharpened descriptions).
