#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "apps/ds4_server.c"

#ifndef DS4_NO_GPU
#include "parts/00_cuda_and_shared_helpers.inc"
#if defined(__APPLE__)
#include "parts/01_apple_kernel_reference_a.inc"
#include "parts/02_apple_kernel_reference_b.inc"
#endif
#if defined(__APPLE__)
#include "parts/03_apple_attention_reference.inc"
#endif
#if defined(__APPLE__)
#include "parts/04_apple_moe_reference.inc"
#endif
#include "parts/05_model_and_vector_tests.inc"
#include "parts/06_equivalence_and_speculative.inc"
#endif

#include "parts/07_test_runner.inc"
