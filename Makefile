CC ?= cc
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

SRC_DIR := src
APP_DIR := $(SRC_DIR)/apps
ENGINE_DIR := $(SRC_DIR)/engine
ENGINE_INCLUDE_DIR := $(ENGINE_DIR)/include
ENGINE_INTERNAL_DIR := $(ENGINE_DIR)/internal
KERNEL_DIR := $(SRC_DIR)/ds4x_kernel
KERNEL_BACKEND_DIR := $(KERNEL_DIR)/backend
KERNEL_MODERN_BACKEND_DIR := $(KERNEL_DIR)/backends
KERNEL_INCLUDE_DIR := $(KERNEL_DIR)/include
KERNEL_TABLE_DIR := $(KERNEL_DIR)/tables
MMQ_DIR := $(KERNEL_DIR)/quantization/mmq
CUTLASS_DIR ?= third_party/cutlass
STORAGE_DIR := $(SRC_DIR)/storage
SUPPORT_DIR := $(SRC_DIR)/support

KERNEL_TEST_DIR := tests/ds4x_kernel
KERNEL_MMQ_TEST_DIR := $(KERNEL_TEST_DIR)/mmq
ENGINE_TEST_DIR := tests/engine
INTEGRATION_TEST_DIR := tests/integration

BUILD_DIR ?= build
OBJ_DIR := $(BUILD_DIR)/obj

INCLUDE_DIRS := -I$(SRC_DIR) -I$(APP_DIR) \
	-I$(ENGINE_DIR) -I$(ENGINE_INCLUDE_DIR) -I$(ENGINE_INTERNAL_DIR) \
	-I$(KERNEL_INCLUDE_DIR) -I$(KERNEL_TABLE_DIR) \
	-I$(MMQ_DIR) -I$(STORAGE_DIR) -I$(SUPPORT_DIR) \
	-I$(KERNEL_MMQ_TEST_DIR) -I$(CUDA_HOME)/include
CPPFLAGS += -D_GNU_SOURCE -DDS4_CUDA_SPARK_ONLY=1 $(INCLUDE_DIRS)
CFLAGS ?= -O3 -g -ffast-math -fno-finite-math-only -march=native \
	-Wall -Wextra -std=c99
NVCC_DEFS := -DDS4_CUDA_HAVE_MXF4=1
CUTLASS_CPPFLAGS := -I$(CUTLASS_DIR)/include
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math \
	-gencode arch=compute_121a,code=sm_121a \
	-Xcompiler -march=native -Xcompiler -pthread
DEPFLAGS := -MMD -MP
CUDA_LDLIBS ?= -lm -Xcompiler -pthread \
	-L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 \
	-lcudart -lcublas
DS4_TEST_MODEL ?= ds4flash.gguf

MMQ_OBJS := \
	$(OBJ_DIR)/ds4x_kernel/quantization/mmq/ds4_ggml_stubs.o \
	$(OBJ_DIR)/ds4x_kernel/quantization/mmq/ds4_mmq.o \
	$(OBJ_DIR)/ds4x_kernel/quantization/mmq/ds4_mmq_d2r.o \
	$(OBJ_DIR)/ds4x_kernel/quantization/mmq/quantize.o \
	$(OBJ_DIR)/ds4x_kernel/quantization/mmq/mmid.o \
	$(OBJ_DIR)/ds4x_kernel/quantization/mmq/mmvq.o \
	$(OBJ_DIR)/ds4x_kernel/quantization/mmq/ds4_repack.o

BACKEND_OBJS := \
	$(OBJ_DIR)/ds4x_kernel/backend/runtime/runtime.o \
	$(OBJ_DIR)/ds4x_kernel/backend/runtime/storage.o \
	$(OBJ_DIR)/ds4x_kernel/backend/ops/linear.o \
	$(OBJ_DIR)/ds4x_kernel/backend/ops/attention_decode.o \
	$(OBJ_DIR)/ds4x_kernel/backend/ops/indexer.o \
	$(OBJ_DIR)/ds4x_kernel/backend/ops/attention_prefill.o \
	$(OBJ_DIR)/ds4x_kernel/backend/ops/moe_quantized.o \
	$(OBJ_DIR)/ds4x_kernel/backend/ops/moe_dispatch.o \
	$(OBJ_DIR)/ds4x_kernel/backend/compat/compat_api.o

KERNEL_OBJS := \
	$(BACKEND_OBJS) \
	$(OBJ_DIR)/ds4x_kernel/backends/cutlass_fp16_gemm.o \
	$(OBJ_DIR)/ds4x_kernel/backends/fp16_projection.o \
	$(OBJ_DIR)/ds4x_kernel/backends/operator_adapters.o \
	$(MMQ_OBJS)

ENGINE_OBJS := \
	$(OBJ_DIR)/engine/core/platform.o \
	$(OBJ_DIR)/engine/model/gguf.o \
	$(OBJ_DIR)/engine/model/validation.o \
	$(OBJ_DIR)/engine/model/dspark_weights.o \
	$(OBJ_DIR)/engine/model/math.o \
	$(OBJ_DIR)/engine/model/tokenizer.o \
	$(OBJ_DIR)/engine/model/context_memory.o \
	$(OBJ_DIR)/engine/runtime/diagnostics.o \
	$(OBJ_DIR)/engine/runtime/graph_state.o \
	$(OBJ_DIR)/engine/runtime/graph_setup.o \
	$(OBJ_DIR)/engine/runtime/decode_primitives.o \
	$(OBJ_DIR)/engine/runtime/decode_layer.o \
	$(OBJ_DIR)/engine/runtime/output.o \
	$(OBJ_DIR)/engine/runtime/dspark_prefill.o \
	$(OBJ_DIR)/engine/runtime/prefill_attention.o \
	$(OBJ_DIR)/engine/runtime/prefill_ffn.o \
	$(OBJ_DIR)/engine/runtime/verifier.o \
	$(OBJ_DIR)/engine/runtime/dspark_verify.o \
	$(OBJ_DIR)/engine/runtime/prefill_runner.o \
	$(OBJ_DIR)/engine/runtime/long_prompt.o \
	$(OBJ_DIR)/engine/session/engine_lifecycle.o \
	$(OBJ_DIR)/engine/session/checkpoint.o \
	$(OBJ_DIR)/engine/session/spec_frontier.o \
	$(OBJ_DIR)/engine/session/session_state.o \
	$(OBJ_DIR)/engine/session/model_cache.o \
	$(OBJ_DIR)/engine/session/generation.o \
	$(OBJ_DIR)/engine/session/session_sync.o \
	$(OBJ_DIR)/engine/session/batch.o

ENGINE_HOOK_OBJS := \
	$(OBJ_DIR)/tests/engine/tokenizer_hooks.o \
	$(OBJ_DIR)/tests/engine/model_cache_hooks.o

ENGINE_INTEGRATION_OBJS := \
	$(filter-out $(OBJ_DIR)/engine/model/tokenizer.o \
		$(OBJ_DIR)/engine/session/model_cache.o,$(ENGINE_OBJS)) \
	$(ENGINE_HOOK_OBJS)

RUNTIME_OBJS := \
	$(ENGINE_OBJS) \
	$(OBJ_DIR)/storage/ds4_memory.o \
	$(KERNEL_OBJS)

APP_OBJS := \
	$(OBJ_DIR)/apps/ds4_cli.o \
	$(OBJ_DIR)/apps/ds4_bench.o \
	$(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o \
	$(OBJ_DIR)/support/linenoise.o \
	$(OBJ_DIR)/support/rax.o

TEST_OBJS := \
	$(OBJ_DIR)/tests/ds4x_kernel/cuda_long_context_smoke.o \
	$(OBJ_DIR)/tests/ds4x_kernel/cuda_long_context_perf.o \
	$(OBJ_DIR)/tests/integration/synth_frontier_bench.o \
	$(OBJ_DIR)/tests/integration/packed_checkpoint_smoke.o \
	$(OBJ_DIR)/tests/engine/ds4_test.o \
	$(ENGINE_HOOK_OBJS)

ALL_OBJS := $(sort $(RUNTIME_OBJS) $(APP_OBJS) $(TEST_OBJS))
DEPS := $(ALL_OBJS:.o=.d)

.PHONY: all spark smoke perf kernel-test kernel-perf kernel-mmq-test \
	kernel-cutlass-test kernel-cutlass-bench \
	kernel-projection-test kernel-operator-test \
	checkpoint-smoke long-context qkvo-fp4-probe clean

all: spark

spark: ds4 ds4-bench

ds4: $(OBJ_DIR)/apps/ds4_cli.o $(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/support/linenoise.o $(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: $(OBJ_DIR)/apps/ds4_bench.o $(OBJ_DIR)/apps/ds4_help.o \
	$(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(OBJ_DIR)/ds4x_kernel/backend/%.o: $(KERNEL_BACKEND_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) $(DEPFLAGS) \
		-std=c++17 -c -o $@ $<

$(OBJ_DIR)/ds4x_kernel/backends/%.o: $(KERNEL_MODERN_BACKEND_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(CUTLASS_CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) \
		$(DEPFLAGS) -std=c++17 -c -o $@ $<

$(OBJ_DIR)/ds4x_kernel/quantization/mmq/%.o: $(MMQ_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) $(DEPFLAGS) \
		-std=c++17 -c -o $@ $<

$(OBJ_DIR)/tests/ds4x_kernel/%.o: $(KERNEL_TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(OBJ_DIR)/tests/integration/%.o: $(INTEGRATION_TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -DDS4_TEST_HOOKS -c -o $@ $<

$(OBJ_DIR)/tests/engine/ds4_test.o: $(ENGINE_TEST_DIR)/ds4_test.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -Wno-unused-function -c -o $@ $<

$(OBJ_DIR)/tests/engine/tokenizer_hooks.o: $(ENGINE_DIR)/model/tokenizer.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -DDS4_TEST_HOOKS -c -o $@ $<

$(OBJ_DIR)/tests/engine/model_cache_hooks.o: $(ENGINE_DIR)/session/model_cache.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -DDS4_TEST_HOOKS -c -o $@ $<

$(KERNEL_TEST_DIR)/cuda_long_context_smoke: \
	$(OBJ_DIR)/tests/ds4x_kernel/cuda_long_context_smoke.o $(KERNEL_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(KERNEL_TEST_DIR)/cuda_long_context_perf: \
	$(OBJ_DIR)/tests/ds4x_kernel/cuda_long_context_perf.o $(KERNEL_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(KERNEL_MMQ_TEST_DIR)/test_mmq_parity: \
	$(KERNEL_MMQ_TEST_DIR)/test_mmq_parity.cu $(KERNEL_OBJS)
	$(NVCC) $(CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) -std=c++17 \
		-o $@ $^ $(CUDA_LDLIBS)

$(INTEGRATION_TEST_DIR)/synth_frontier_bench: \
	$(OBJ_DIR)/tests/integration/synth_frontier_bench.o \
	$(ENGINE_INTEGRATION_OBJS) \
	$(OBJ_DIR)/storage/ds4_kvstore.o \
	$(OBJ_DIR)/support/rax.o \
	$(OBJ_DIR)/storage/ds4_memory.o \
	$(KERNEL_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(INTEGRATION_TEST_DIR)/packed_checkpoint_smoke: \
	$(OBJ_DIR)/tests/integration/packed_checkpoint_smoke.o \
	$(ENGINE_INTEGRATION_OBJS) \
	$(OBJ_DIR)/storage/ds4_kvstore.o \
	$(OBJ_DIR)/support/rax.o \
	$(OBJ_DIR)/storage/ds4_memory.o \
	$(KERNEL_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(KERNEL_TEST_DIR)/qkvo_fp4_probe: $(KERNEL_TEST_DIR)/qkvo_fp4_probe.cu \
	$(KERNEL_OBJS)
	$(NVCC) $(CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) -std=c++17 \
		-o $@ $^ $(CUDA_LDLIBS)

qkvo-fp4-probe: $(KERNEL_TEST_DIR)/qkvo_fp4_probe
	DS4_CUDA_MMQ_Q81_PERSISTENT=1 ./$(KERNEL_TEST_DIR)/qkvo_fp4_probe

ds4_test: $(OBJ_DIR)/tests/engine/ds4_test.o $(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o $(OBJ_DIR)/support/rax.o \
	$(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

smoke: $(KERNEL_TEST_DIR)/cuda_long_context_smoke
	./$(KERNEL_TEST_DIR)/cuda_long_context_smoke

kernel-mmq-test: $(KERNEL_MMQ_TEST_DIR)/test_mmq_parity
	DS4_CUDA_MMQ_Q81_PERSISTENT=1 ./$(KERNEL_MMQ_TEST_DIR)/test_mmq_parity

$(KERNEL_TEST_DIR)/backends/test_cutlass_fp16_gemm: \
	$(KERNEL_TEST_DIR)/backends/test_cutlass_fp16_gemm.cu \
	$(OBJ_DIR)/ds4x_kernel/backends/cutlass_fp16_gemm.o
	$(NVCC) $(CPPFLAGS) $(CUTLASS_CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) \
		-std=c++17 -o $@ $^ $(CUDA_LDLIBS)

kernel-cutlass-test: $(KERNEL_TEST_DIR)/backends/test_cutlass_fp16_gemm
	./$(KERNEL_TEST_DIR)/backends/test_cutlass_fp16_gemm

kernel-cutlass-bench: $(KERNEL_TEST_DIR)/backends/test_cutlass_fp16_gemm
	./$(KERNEL_TEST_DIR)/backends/test_cutlass_fp16_gemm --benchmark

$(KERNEL_TEST_DIR)/backends/test_fp16_projection: \
	$(KERNEL_TEST_DIR)/backends/test_fp16_projection.cu \
	$(OBJ_DIR)/ds4x_kernel/backends/fp16_projection.o
	$(NVCC) $(CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) -std=c++17 \
		-o $@ $^ $(CUDA_LDLIBS)

kernel-projection-test: $(KERNEL_TEST_DIR)/backends/test_fp16_projection
	./$(KERNEL_TEST_DIR)/backends/test_fp16_projection

$(KERNEL_TEST_DIR)/backends/test_operator_adapters: \
	$(KERNEL_TEST_DIR)/backends/test_operator_adapters.cu $(KERNEL_OBJS)
	$(NVCC) $(CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) -std=c++17 \
		-o $@ $^ $(CUDA_LDLIBS)

kernel-operator-test: $(KERNEL_TEST_DIR)/backends/test_operator_adapters
	./$(KERNEL_TEST_DIR)/backends/test_operator_adapters

kernel-test: smoke kernel-mmq-test kernel-cutlass-test kernel-projection-test \
	kernel-operator-test

kernel-perf: $(KERNEL_TEST_DIR)/cuda_long_context_perf

perf: kernel-perf $(INTEGRATION_TEST_DIR)/synth_frontier_bench \
	$(INTEGRATION_TEST_DIR)/packed_checkpoint_smoke

checkpoint-smoke: $(INTEGRATION_TEST_DIR)/packed_checkpoint_smoke
	./$(INTEGRATION_TEST_DIR)/packed_checkpoint_smoke "$(DS4_TEST_MODEL)" 8192

long-context: ds4_test
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" \
	DS4_TEST_LONG_PROMPT="$(INTEGRATION_TEST_DIR)/long_context_story_prompt.txt" \
	./ds4_test --long-context

clean:
	rm -rf $(BUILD_DIR)
	rm -f ds4 ds4-bench ds4_test \
		$(KERNEL_TEST_DIR)/cuda_long_context_smoke \
		$(KERNEL_TEST_DIR)/cuda_long_context_perf \
		$(KERNEL_TEST_DIR)/qkvo_fp4_probe \
		$(KERNEL_TEST_DIR)/backends/test_cutlass_fp16_gemm \
		$(KERNEL_TEST_DIR)/backends/test_fp16_projection \
		$(KERNEL_TEST_DIR)/backends/test_operator_adapters \
		$(KERNEL_MMQ_TEST_DIR)/test_mmq_parity \
		$(INTEGRATION_TEST_DIR)/synth_frontier_bench \
		$(INTEGRATION_TEST_DIR)/packed_checkpoint_smoke

-include $(DEPS)
