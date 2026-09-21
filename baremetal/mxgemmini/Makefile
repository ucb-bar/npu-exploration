# Hand-written MxGemmini application kernels: PyTorch capture -> generated data -> baremetal ELF.
#
# These used to live in gemmini-rocc-tests/bareMetalC, which is the ISA-level test suite -- one test
# per instruction behaviour. A TinyLlama decoder layer is not that, so it lives here instead. What
# stays over there is what these still borrow: the ISA headers (gemmini.h, gemmini_params.h,
# gemmini_testutils.h), the riscv-tests baremetal runtime, and gen_matmul_llama.py, which generates
# the matmul_*.h data for the ISA tests themselves.
#
#   make            # build every kernel for spike AND the RTL path
#   make spike      # spike only  (-DSPIKE_SIM, real RoCC)
#   make mx_rocket  # RTL only    (-DMX_ROCKET)
#   make data       # regenerate data/ from the capture npz (see gen/)
#   make run        # build for spike and run each one
#   make clean
#
# ELFs land in ../../out/baremetal/{spike,mx_rocket}/ -- out/ is gitignored, like every other
# build artifact in this repo.

REPO      := $(abspath $(CURDIR)/../..)
ROCC      ?= $(abspath $(REPO)/../software/gemmini-rocc-tests)
LIBGEMMINI?= $(abspath $(REPO)/../software/libgemmini/libgemmini.so)
VENV      := $(REPO)/.venv/bin/python3

XLEN      ?= 64
CC        := riscv$(XLEN)-unknown-elf-gcc
OBJCOPY   := riscv$(XLEN)-unknown-elf-objcopy
BENCH     := $(ROCC)/riscv-tests/benchmarks/common
OUT       := $(REPO)/out/baremetal

KERNELS   := llama_mlp llama_mlp_small llama_mlp_tiny \
             llama_attention llama_attention_small llama_attention_tiny llama_attention_full

# The MX bisection ladder: fifteen tiny self-checking ELFs. mxl0..mxl9 each take ONE delta from
# the 64x64 ISA shape that passes, walking to the llama projection that does not; mxl10..mxl14 are
# the second tier, separating the four things that change across the mxl3-to-mxl4 cliff. Run in
# order; the first that fails names the feature. See src/mx_ladder.c and gen/gen_mx_ladder.py.
LADDER    := mxl0 mxl1 mxl2 mxl3 mxl4 mxl5 mxl6 mxl7 mxl8 mxl9 \
             mxl10 mxl11 mxl12 mxl13 mxl14 \
             mxl15 mxl16 mxl17 mxl18

# Matches gemmini-rocc-tests/bareMetalC/Makefile's flags; -I order puts our headers first so
# mx_host.h and data/*.h resolve here, while "include/gemmini_testutils.h" resolves in $(ROCC).
CFLAGS := -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math \
          -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns \
          -march=rv64gc -Wa,-march=rv64gc -lgcc \
          -I$(CURDIR)/include -I$(CURDIR)/data \
          -I$(ROCC) -I$(ROCC)/riscv-tests -I$(ROCC)/riscv-tests/env -I$(BENCH) \
          -DID_STRING= -DPRINT_TILE=0
CFLAGS_BM := $(CFLAGS) -nostdlib -nostartfiles -static -T $(BENCH)/test.ld -DBAREMETAL=1
# AFTER the sources, which is where a library has to be to resolve anything: -lm inside CFLAGS
# precedes them and resolves nothing. expf (SiLU) is the only libm call; mx_host.h supplies the
# __errno stub that -nostdlib leaves undefined.
LIBS := -lm
RUNTIME := $(wildcard $(BENCH)/*.c) $(wildcard $(BENCH)/*.S)

DATA := $(wildcard $(CURDIR)/data/*.h)
DEPS := $(CURDIR)/include/mx_host.h $(CURDIR)/include/mx_mesh.h $(DATA) \
        $(ROCC)/include/gemmini.h $(ROCC)/include/gemmini_params.h \
        $(ROCC)/include/gemmini_testutils.h

.PHONY: all spike mx_rocket ladder ladder-spike data run run-ladder clean
all: spike mx_rocket
spike:     $(addprefix $(OUT)/spike/,$(KERNELS) $(LADDER))
mx_rocket: $(addprefix $(OUT)/mx_rocket/,$(KERNELS) $(LADDER))

# Just the ladder, both targets -- what you want when bisecting an FPGA failure.
ladder:       $(addprefix $(OUT)/mx_rocket/,$(LADDER)) $(addprefix $(OUT)/spike/,$(LADDER))
ladder-spike: $(addprefix $(OUT)/spike/,$(LADDER))

# Every rung is src/mxlN.c including src/mx_ladder.c, so the pattern rule's own prerequisite
# misses the file that actually carries the driver (same reason llama_attention_small needs it).
$(foreach k,$(LADDER),$(eval $(OUT)/spike/$(k) $(OUT)/mx_rocket/$(k): $(CURDIR)/src/mx_ladder.c))

# The sliced variants are the base kernel with a different data header selected, so the pattern
# rule's own prerequisite misses the source that actually carries the kernel.
$(foreach k,llama_attention_small llama_attention_tiny,\
  $(eval $(OUT)/spike/$(k) $(OUT)/mx_rocket/$(k): $(CURDIR)/src/llama_attention.c))
$(foreach k,llama_mlp_small llama_mlp_tiny,\
  $(eval $(OUT)/spike/$(k) $(OUT)/mx_rocket/$(k): $(CURDIR)/src/llama_mlp.c))

# llama_attention_full carries 10.8 MB of weights. As C initializers that is ~56 MB of source, so
# the data is a binary blob linked as its own section. objcopy derives the symbol names from the
# input PATH, so the `cd data` is load-bearing: it is what makes them
# _binary_llama_attn_full_bin_*. Default section alignment is 1, but the header casts to uint16_t*.
$(OUT)/%/llama_attn_full_blob.o: $(CURDIR)/data/llama_attn_full.bin
	@mkdir -p $(dir $@)
	cd $(CURDIR)/data && $(OBJCOPY) -I binary -O elf64-littleriscv -B riscv \
		--set-section-alignment .data=64 llama_attn_full.bin $@

$(OUT)/spike/llama_attention_full:     $(OUT)/spike/llama_attn_full_blob.o
$(OUT)/mx_rocket/llama_attention_full: $(OUT)/mx_rocket/llama_attn_full_blob.o

$(OUT)/spike/%: $(CURDIR)/src/%.c $(DEPS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_BM) -DSPIKE_SIM $< -o $@ $(RUNTIME) \
		$(if $(filter llama_attention_full,$*),$(OUT)/spike/llama_attn_full_blob.o,) $(LIBS)

$(OUT)/mx_rocket/%: $(CURDIR)/src/%.c $(DEPS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_BM) -DMX_ROCKET $< -o $@ $(RUNTIME) \
		$(if $(filter llama_attention_full,$*),$(OUT)/mx_rocket/llama_attn_full_blob.o,) $(LIBS)

# Regenerate every data header from the newest capture. The full-attention one needs an
# --all-heads capture; see gen/gen_llama_attn_full.py.
data:
	cd $(CURDIR)/gen && $(VENV) gen_llama_layer.py mlp attn
	cd $(CURDIR)/gen && $(VENV) gen_llama_layer.py mlp  --d 256 --tag _small
	cd $(CURDIR)/gen && $(VENV) gen_llama_layer.py attn --d 256 --tag _small
	cd $(CURDIR)/gen && $(VENV) gen_llama_layer.py mlp attn --d 64 --tag _tiny
	cd $(CURDIR)/gen && $(VENV) gen_llama_attn_full.py
	cd $(CURDIR)/gen && $(VENV) gen_mx_ladder.py

run: spike
	@for k in $(KERNELS); do \
	  echo "=== $$k ==="; \
	  spike --extlib=$(LIBGEMMINI) --extension=gemmini $(OUT)/spike/$$k \
	    2>&1 | grep -vE '^Gemmini ext|^    dim'; \
	done

# The ladder on spike, in bisection order. A rung that fails HERE is a bug in the ladder itself,
# not in the FPGA -- spike is the reference the goldens were written against.
run-ladder: ladder-spike
	@for k in $(LADDER); do \
	  spike --extlib=$(LIBGEMMINI) --extension=gemmini $(OUT)/spike/$$k \
	    2>&1 | grep -vE '^Gemmini ext|^    dim'; \
	  echo; \
	done

clean:
	rm -rf $(OUT)
