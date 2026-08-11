CC ?= cc
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

SRC_DIR := src
APP_DIR := $(SRC_DIR)/apps
ENGINE_DIR := $(SRC_DIR)/engine
ENGINE_INCLUDE_DIR := $(ENGINE_DIR)/include
ENGINE_MODEL_DIR := $(ENGINE_DIR)/model
ENGINE_CONFIG_DIR := $(ENGINE_DIR)/config
KERNEL_DIR := $(SRC_DIR)/ds4x_kernel
KERNEL_BACKEND_DIR := $(KERNEL_DIR)/backend
KERNEL_INCLUDE_DIR := $(KERNEL_DIR)/include
KERNEL_TABLE_DIR := $(KERNEL_DIR)/tables
MMQ_DIR := $(KERNEL_DIR)/quantization/mmq
DIST_DIR := $(SRC_DIR)/distributed
STORAGE_DIR := $(SRC_DIR)/storage
SUPPORT_DIR := $(SRC_DIR)/support

KERNEL_TEST_DIR := tests/ds4x_kernel
KERNEL_MMQ_TEST_DIR := $(KERNEL_TEST_DIR)/mmq
ENGINE_TEST_DIR := tests/engine
INTEGRATION_TEST_DIR := tests/integration

BUILD_DIR ?= build
OBJ_DIR := $(BUILD_DIR)/obj

INCLUDE_DIRS := -I$(SRC_DIR) -I$(APP_DIR) \
	-I$(ENGINE_DIR) -I$(ENGINE_INCLUDE_DIR) -I$(ENGINE_MODEL_DIR) \
	-I$(ENGINE_CONFIG_DIR) -I$(KERNEL_INCLUDE_DIR) -I$(KERNEL_TABLE_DIR) \
	-I$(MMQ_DIR) -I$(DIST_DIR) -I$(STORAGE_DIR) -I$(SUPPORT_DIR) \
	-I$(KERNEL_MMQ_TEST_DIR) -I$(CUDA_HOME)/include
CPPFLAGS += -D_GNU_SOURCE -DDS4_CUDA_SPARK_ONLY=1 $(INCLUDE_DIRS)
CFLAGS ?= -O3 -g -ffast-math -fno-finite-math-only -march=native \
	-Wall -Wextra -std=c99
NVCC_DEFS := -DDS4_CUDA_HAVE_MXF4=1
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

KERNEL_OBJS := \
	$(OBJ_DIR)/ds4x_kernel/backend/ds4x_kernel.o \
	$(MMQ_OBJS)

RUNTIME_OBJS := \
	$(OBJ_DIR)/engine/ds4_engine.o \
	$(OBJ_DIR)/engine/model/ds4_layer_pack.o \
	$(OBJ_DIR)/distributed/ds4_distributed.o \
	$(OBJ_DIR)/distributed/ds4_tp.o \
	$(OBJ_DIR)/storage/ds4_ssd.o \
	$(KERNEL_OBJS)

APP_OBJS := \
	$(OBJ_DIR)/apps/ds4_cli.o \
	$(OBJ_DIR)/apps/ds4_server.o \
	$(OBJ_DIR)/apps/ds4_bench.o \
	$(OBJ_DIR)/apps/ds4_eval.o \
	$(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/engine/config/ds4_gpu_args.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o \
	$(OBJ_DIR)/support/linenoise.o \
	$(OBJ_DIR)/support/rax.o

TEST_OBJS := \
	$(OBJ_DIR)/tests/ds4x_kernel/cuda_long_context_smoke.o \
	$(OBJ_DIR)/tests/ds4x_kernel/cuda_long_context_perf.o \
	$(OBJ_DIR)/tests/integration/synth_frontier_bench.o \
	$(OBJ_DIR)/tests/integration/packed_checkpoint_smoke.o \
	$(OBJ_DIR)/tests/engine/ds4_test.o \
	$(OBJ_DIR)/engine/ds4_test_hooks.o

ALL_OBJS := $(sort $(RUNTIME_OBJS) $(APP_OBJS) $(TEST_OBJS))
DEPS := $(ALL_OBJS:.o=.d)

.PHONY: all spark smoke perf kernel-test kernel-perf kernel-mmq-test \
	checkpoint-smoke long-context qkvo-fp4-probe clean

all: spark

spark: ds4 ds4-server ds4-bench ds4-eval

ds4: $(OBJ_DIR)/apps/ds4_cli.o $(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/support/linenoise.o $(OBJ_DIR)/engine/config/ds4_gpu_args.o \
	$(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: $(OBJ_DIR)/apps/ds4_server.o $(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o $(OBJ_DIR)/support/rax.o \
	$(OBJ_DIR)/engine/config/ds4_gpu_args.o $(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: $(OBJ_DIR)/apps/ds4_bench.o $(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/engine/config/ds4_gpu_args.o $(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: $(OBJ_DIR)/apps/ds4_eval.o $(OBJ_DIR)/apps/ds4_help.o \
	$(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(OBJ_DIR)/ds4x_kernel/backend/ds4x_kernel.o: $(KERNEL_BACKEND_DIR)/ds4x_kernel.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) $(DEPFLAGS) -c -o $@ $<

$(OBJ_DIR)/ds4x_kernel/quantization/mmq/%.o: $(MMQ_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) $(DEPFLAGS) \
		-std=c++17 -c -o $@ $<

$(OBJ_DIR)/engine/ds4_test_hooks.o: $(ENGINE_DIR)/ds4_engine.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -Wno-unused-function \
		-DDS4_TEST_HOOKS -c -o $@ $<

$(OBJ_DIR)/tests/ds4x_kernel/%.o: $(KERNEL_TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(OBJ_DIR)/tests/integration/%.o: $(INTEGRATION_TEST_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -DDS4_TEST_HOOKS -c -o $@ $<

$(OBJ_DIR)/tests/engine/ds4_test.o: $(ENGINE_TEST_DIR)/ds4_test.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -Wno-unused-function -c -o $@ $<

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
	$(OBJ_DIR)/engine/ds4_test_hooks.o \
	$(OBJ_DIR)/engine/config/ds4_gpu_args.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o \
	$(OBJ_DIR)/support/rax.o \
	$(OBJ_DIR)/distributed/ds4_distributed.o \
	$(OBJ_DIR)/distributed/ds4_tp.o \
	$(OBJ_DIR)/storage/ds4_ssd.o \
	$(OBJ_DIR)/engine/model/ds4_layer_pack.o $(KERNEL_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(INTEGRATION_TEST_DIR)/packed_checkpoint_smoke: \
	$(OBJ_DIR)/tests/integration/packed_checkpoint_smoke.o \
	$(OBJ_DIR)/engine/ds4_test_hooks.o \
	$(OBJ_DIR)/engine/config/ds4_gpu_args.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o \
	$(OBJ_DIR)/support/rax.o \
	$(OBJ_DIR)/distributed/ds4_distributed.o \
	$(OBJ_DIR)/distributed/ds4_tp.o \
	$(OBJ_DIR)/storage/ds4_ssd.o \
	$(OBJ_DIR)/engine/model/ds4_layer_pack.o $(KERNEL_OBJS)
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

kernel-test: smoke kernel-mmq-test

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
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4_test \
		$(KERNEL_TEST_DIR)/cuda_long_context_smoke \
		$(KERNEL_TEST_DIR)/cuda_long_context_perf \
		$(KERNEL_TEST_DIR)/qkvo_fp4_probe \
		$(KERNEL_MMQ_TEST_DIR)/test_mmq_parity \
		$(INTEGRATION_TEST_DIR)/synth_frontier_bench \
		$(INTEGRATION_TEST_DIR)/packed_checkpoint_smoke

-include $(DEPS)
