# Integration tests

These binaries link the inference engine and kernel layer together.

- `packed_checkpoint_smoke.c` validates packed KV/indexer persistence and
  logits after restore.
- `synth_frontier_bench.c` materializes deterministic long-context frontiers
  and measures exact decode without a million-token prefill.
- `generate_long_context_story_prompt.py` and
  `long_context_story_prompt.txt` support semantic fact-recall regression.

Build them with `make perf`; run the packed checkpoint test with
`make checkpoint-smoke DS4_TEST_MODEL=/path/to/model.gguf`.
