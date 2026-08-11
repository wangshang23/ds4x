#include "ds4_help.h"

#include <string.h>

static void print_common(FILE *fp) {
    fputs(
        "  -m, --model FILE          DeepSeek-V4-Flash Q2 GGUF\n"
        "  --cuda                    Use the GB10 CUDA backend (default)\n"
        "  -c, --ctx N               Context capacity\n"
        "  --prefill-chunk N         Override prefill chunk size\n"
        "  --power N                 Runtime power percentage (1-100)\n"
        "  --quality                 Prefer strict numerical paths\n"
        "  --warm-weights            Touch model pages during startup\n"
        "  --simulate-used-memory N  Reserve unified memory for OOM tests\n",
        fp);
}

static void print_ds4(FILE *fp) {
    fputs("Usage: ds4 [(-p PROMPT | --prompt-file FILE)] [options]\n\n", fp);
    fputs("Single-request DeepSeek-V4-Flash-Q2 inference for one DGX Spark GB10.\n\n", fp);
    fputs("Prompt and generation:\n", fp);
    fputs(
        "  -p, --prompt TEXT         Run one prompt; omit for interactive mode\n"
        "  --prompt-file FILE        Read the prompt from a file\n"
        "  --raw                     Skip chat-template rendering\n"
        "  -sys, --system TEXT       System prompt\n"
        "  -n, --tokens N            Maximum output tokens\n"
        "  --temp F                  Sampling temperature (0 = greedy)\n"
        "  --top-p F                 Nucleus sampling threshold\n"
        "  --min-p F                 Minimum probability threshold\n"
        "  --seed N                  Sampling seed\n"
        "  --think | --think-max | --nothink\n",
        fp);
    fputs("\nDSpark:\n", fp);
    fputs(
        "  --dspark-model FILE       Quantized DSpark support checkpoint\n"
        "  --dspark                  Enable single-request speculative decode\n"
        "  --dspark-confidence F     Draft confidence threshold\n"
        "  --dspark-strict           Preserve one-token target behavior\n",
        fp);
    fputs("\nDiagnostics:\n", fp);
    fputs(
        "  --inspect                 Print checkpoint metadata and exit\n"
        "  --dump-logits FILE        Write prompt logits\n"
        "  --dump-logprobs FILE      Write teacher-forced log probabilities\n"
        "  --decode-consistency N    Compare live and rebuilt decode states\n"
        "  --perplexity-file FILE    Score raw text\n",
        fp);
    fputs("\nRuntime:\n", fp);
    print_common(fp);
}

static void print_bench(FILE *fp) {
    fputs("Usage: ds4-bench (--prompt-file FILE | --chat-prompt-file FILE) [options]\n\n", fp);
    fputs("Single-session prefill and decode frontier benchmark.\n\n", fp);
    fputs(
        "  --ctx-start N             First measured frontier\n"
        "  --ctx-max N               Last measured frontier\n"
        "  --ctx-alloc N             Allocated session capacity\n"
        "  --step-incr N             Linear frontier increment\n"
        "  --step-mul F              Geometric frontier multiplier\n"
        "  --gen-tokens N            Decode tokens per frontier\n"
        "  --csv FILE                Write CSV output\n"
        "  --show-output             Print decoded text\n"
        "  --dump-frontier-logits-dir DIR\n",
        fp);
    fputs("\nRuntime:\n", fp);
    print_common(fp);
}

void ds4_help_print(FILE *fp, ds4_help_tool tool, const char *topic) {
    if (topic && topic[0] && strcmp(topic, "runtime") != 0 &&
        strcmp(topic, "dspark") != 0) {
        fprintf(fp, "Unknown help topic: %s\n\n", topic);
    }
    if (tool == DS4_HELP_BENCH) print_bench(fp);
    else print_ds4(fp);
}
