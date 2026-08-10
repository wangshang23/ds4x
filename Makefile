CC ?= cc
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

SRC_DIR := src
APP_DIR := $(SRC_DIR)/apps
CORE_DIR := $(SRC_DIR)/core
CUDA_DIR := $(SRC_DIR)/cuda
DIST_DIR := $(SRC_DIR)/distributed
STORAGE_DIR := $(SRC_DIR)/storage
SUPPORT_DIR := $(SRC_DIR)/support
MMQ_DIR := $(CUDA_DIR)/mmq
BUILD_DIR ?= build
OBJ_DIR := $(BUILD_DIR)/obj

INCLUDE_DIRS := -I$(SRC_DIR) -I$(APP_DIR) -I$(CORE_DIR) -I$(CUDA_DIR) \
	-I$(DIST_DIR) -I$(STORAGE_DIR) -I$(SUPPORT_DIR) -I$(MMQ_DIR) \
	-I$(CUDA_HOME)/include
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
	$(OBJ_DIR)/cuda/mmq/ds4_ggml_stubs.o \
	$(OBJ_DIR)/cuda/mmq/ds4_mmq.o \
	$(OBJ_DIR)/cuda/mmq/ds4_mmq_d2r.o \
	$(OBJ_DIR)/cuda/mmq/quantize.o \
	$(OBJ_DIR)/cuda/mmq/mmid.o \
	$(OBJ_DIR)/cuda/mmq/mmvq.o \
	$(OBJ_DIR)/cuda/mmq/ds4_repack.o

RUNTIME_OBJS := \
	$(OBJ_DIR)/core/ds4.o \
	$(OBJ_DIR)/distributed/ds4_distributed.o \
	$(OBJ_DIR)/distributed/ds4_tp.o \
	$(OBJ_DIR)/storage/ds4_ssd.o \
	$(OBJ_DIR)/cuda/ds4_cuda.o \
	$(OBJ_DIR)/core/ds4_layer_pack.o \
	$(MMQ_OBJS)

APP_OBJS := \
	$(OBJ_DIR)/apps/ds4_cli.o \
	$(OBJ_DIR)/apps/ds4_server.o \
	$(OBJ_DIR)/apps/ds4_bench.o \
	$(OBJ_DIR)/apps/ds4_eval.o \
	$(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/cuda/ds4_gpu_args.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o \
	$(OBJ_DIR)/support/linenoise.o \
	$(OBJ_DIR)/support/rax.o

TEST_OBJS := \
	$(OBJ_DIR)/tests/cuda_long_context_smoke.o \
	$(OBJ_DIR)/tests/cuda_long_context_perf.o \
	$(OBJ_DIR)/tests/synth_frontier_bench.o \
	$(OBJ_DIR)/tests/packed_checkpoint_smoke.o \
	$(OBJ_DIR)/tests/ds4_test.o \
	$(OBJ_DIR)/core/ds4_test_hooks.o

ALL_OBJS := $(sort $(RUNTIME_OBJS) $(APP_OBJS) $(TEST_OBJS))
DEPS := $(ALL_OBJS:.o=.d)

.PHONY: all spark smoke perf checkpoint-smoke long-context clean

all: spark

spark: ds4 ds4-server ds4-bench ds4-eval

ds4: $(OBJ_DIR)/apps/ds4_cli.o $(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/support/linenoise.o $(OBJ_DIR)/cuda/ds4_gpu_args.o \
	$(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: $(OBJ_DIR)/apps/ds4_server.o $(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o $(OBJ_DIR)/support/rax.o \
	$(OBJ_DIR)/cuda/ds4_gpu_args.o $(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: $(OBJ_DIR)/apps/ds4_bench.o $(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/cuda/ds4_gpu_args.o $(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: $(OBJ_DIR)/apps/ds4_eval.o $(OBJ_DIR)/apps/ds4_help.o \
	$(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(OBJ_DIR)/cuda/ds4_cuda.o: $(CUDA_DIR)/ds4_cuda.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) $(DEPFLAGS) -c -o $@ $<

$(OBJ_DIR)/cuda/mmq/%.o: $(MMQ_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCC_DEFS) $(NVCCFLAGS) $(DEPFLAGS) \
		-std=c++17 -c -o $@ $<

$(OBJ_DIR)/core/ds4_test_hooks.o: $(CORE_DIR)/ds4.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -Wno-unused-function \
		-DDS4_TEST_HOOKS -c -o $@ $<

$(OBJ_DIR)/tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(OBJ_DIR)/tests/cuda_long_context_perf.o: tests/cuda_long_context_perf.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(OBJ_DIR)/tests/synth_frontier_bench.o: tests/synth_frontier_bench.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -DDS4_TEST_HOOKS \
		-c -o $@ $<

$(OBJ_DIR)/tests/packed_checkpoint_smoke.o: tests/packed_checkpoint_smoke.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -DDS4_TEST_HOOKS \
		-c -o $@ $<

$(OBJ_DIR)/tests/ds4_test.o: tests/ds4_test.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -Wno-unused-function \
		-c -o $@ $<

tests/cuda_long_context_smoke: \
	$(OBJ_DIR)/tests/cuda_long_context_smoke.o \
	$(OBJ_DIR)/cuda/ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/cuda_long_context_perf: \
	$(OBJ_DIR)/tests/cuda_long_context_perf.o \
	$(OBJ_DIR)/cuda/ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/synth_frontier_bench: \
	$(OBJ_DIR)/tests/synth_frontier_bench.o \
	$(OBJ_DIR)/core/ds4_test_hooks.o \
	$(OBJ_DIR)/cuda/ds4_gpu_args.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o \
	$(OBJ_DIR)/support/rax.o \
	$(OBJ_DIR)/distributed/ds4_distributed.o \
	$(OBJ_DIR)/distributed/ds4_tp.o \
	$(OBJ_DIR)/storage/ds4_ssd.o \
	$(OBJ_DIR)/cuda/ds4_cuda.o \
	$(OBJ_DIR)/core/ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/packed_checkpoint_smoke: \
	$(OBJ_DIR)/tests/packed_checkpoint_smoke.o \
	$(OBJ_DIR)/core/ds4_test_hooks.o \
	$(OBJ_DIR)/cuda/ds4_gpu_args.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o \
	$(OBJ_DIR)/support/rax.o \
	$(OBJ_DIR)/distributed/ds4_distributed.o \
	$(OBJ_DIR)/distributed/ds4_tp.o \
	$(OBJ_DIR)/storage/ds4_ssd.o \
	$(OBJ_DIR)/cuda/ds4_cuda.o \
	$(OBJ_DIR)/core/ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: $(OBJ_DIR)/tests/ds4_test.o $(OBJ_DIR)/apps/ds4_help.o \
	$(OBJ_DIR)/storage/ds4_kvstore.o $(OBJ_DIR)/support/rax.o \
	$(RUNTIME_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

smoke: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

perf: tests/cuda_long_context_perf tests/synth_frontier_bench \
	tests/packed_checkpoint_smoke

checkpoint-smoke: tests/packed_checkpoint_smoke
	./tests/packed_checkpoint_smoke "$(DS4_TEST_MODEL)" 8192

long-context: ds4_test
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" \
	DS4_TEST_LONG_PROMPT="tests/long_context_story_prompt.txt" \
	./ds4_test --long-context

clean:
	rm -rf $(BUILD_DIR)
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4_test \
		tests/cuda_long_context_smoke tests/cuda_long_context_perf \
		tests/synth_frontier_bench tests/packed_checkpoint_smoke

-include $(DEPS)
