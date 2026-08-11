#include <stdbool.h>
#include <stdio.h>
#include <string.h>

bool ds4_test_dspark_cache_window_crop(void);

static void print_help(const char *program) {
    printf("Usage: %s [--dspark-cache-window | --all]\n", program);
    puts("  --dspark-cache-window  Validate DSpark cache crop/append invariants.");
    puts("  --all                  Run every single-request engine invariant.");
}

int main(int argc, char **argv) {
    bool run_dspark = argc == 1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--all") ||
            !strcmp(argv[i], "--dspark-cache-window")) {
            run_dspark = true;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            print_help(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown test: %s\n", argv[i]);
            print_help(argv[0]);
            return 2;
        }
    }

    if (run_dspark && !ds4_test_dspark_cache_window_crop()) {
        fputs("dspark-cache-window: FAILED\n", stderr);
        return 1;
    }
    puts("dspark-cache-window: OK");
    return 0;
}
