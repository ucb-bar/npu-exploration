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

KERNELS   := llama_mlp llama_mlp_small llama_mlp_tiny llama_mlp_tiny_t1 llama_mlp_tiny_db llama_mlp_tiny_native llama_mlp_small_native llama_mlp_tiny_native_ua llama_mlp_native llama_mlp_full \
             llama_attention llama_attention_small llama_attention_tiny llama_attention_tiny_t1 llama_attention_tiny_native llama_attention_small_native llama_attention_native llama_attention_full \
             llama_attention_full_dgb \
             llama_qkv_tiny llama_proj_q llama_proj_k llama_proj_v llama_sep_qkv llama_two_mm llama_two_mm_micro \
             llama_layer_full llama_layer_full_native llama_layer_full_native_gh \
             mx_bench_matmul mx_bench_matmul_m32 mx_bench_matmul_m32_blk mx_bench_matmul_m64_proj \
             attn_vpu attn_vpu_mid attn_vpu_fa attn_flash attn_flash_mid attn_flash_llama attn_flash_llama_wide attn_flash_llama_causal attn_flash_llama_vb attn_flash_llama_2h attn_flash_llama_vb_fused attn_flash_llama_2h_fused attn_flash_llama_vb_fused_dbg attn_flash_layer5 attn_flash_layer5_fused attn_flash_llama7b attn_flash_llama7b_fused mx_fp4_dramloop attn_flash_llama_2h_fused_fp4 attn_flash_llama7b_fused_fp4 llama_mlp_e2e llama_mlp_e2e_fp4 mlp_proj_fp4 mlp_proj_fp4_short mlp_proj_fp4_ares mlp_proj_fp4_short_ares mlp_proj_fp4_short_chain mlp_proj_fp4_short_m128_chain \
             llama_layer_e2e llama_e2e_elemwise

# Kernels whose data is a linked binary blob rather than C initializers, and the blob each uses.
# See the objcopy rule below for why the basenames matter.
BLOB_KERNELS := llama_attention_full llama_attention_full_dgb llama_mlp_full llama_layer_full llama_layer_full_native \
                llama_layer_full_native_gh mx_bench_matmul mx_bench_matmul_m32 mx_bench_matmul_m32_blk mx_bench_matmul_m64_proj llama_model llama_model_l2 \
                llama_model_m64 llama_model_m64l2 mx_bench llama_layer_e2e llama_e2e_elemwise llama_mlp_e2e llama_mlp_e2e_fp4 mlp_proj_fp4 mlp_proj_fp4_short mlp_proj_fp4_ares mlp_proj_fp4_short_ares mlp_proj_fp4_short_chain mlp_proj_fp4_short_m128_chain
BLOBS        := llama_attn_full llama_mlp_full llama_layer_full llama_model llama_model_l2 \
                llama_model_m64 llama_model_m64l2 mx_bench llama_layer_e2e llama_mlp_fp4 mlp_proj_fp4 mlp_proj_fp4_short mlp_proj_fp4_short_m128
blob_llama_attention_full := llama_attn_full
blob_llama_attention_full_dgb := llama_attn_full
blob_llama_mlp_full       := llama_mlp_full
blob_llama_layer_full     := llama_layer_full
blob_llama_layer_full_native := llama_layer_full
blob_llama_layer_full_native_gh := llama_layer_full
blob_mx_bench_matmul     := mx_bench
blob_mx_bench_matmul_m32 := mx_bench
blob_mx_bench_matmul_m32_blk := mx_bench
blob_mx_bench_matmul_m64_proj := mx_bench
blob_llama_model          := llama_model
blob_llama_model_l2       := llama_model_l2
blob_llama_model_m64      := llama_model_m64
blob_llama_model_m64l2    := llama_model_m64l2
blob_llama_layer_e2e      := llama_layer_e2e
blob_llama_e2e_elemwise   := llama_layer_e2e
blob_llama_mlp_e2e        := llama_layer_e2e
blob_llama_mlp_e2e_fp4    := llama_mlp_fp4
blob_mlp_proj_fp4         := mlp_proj_fp4
blob_mlp_proj_fp4_short   := mlp_proj_fp4_short
blob_mlp_proj_fp4_ares    := mlp_proj_fp4
blob_mlp_proj_fp4_short_ares := mlp_proj_fp4_short
blob_mlp_proj_fp4_short_chain := mlp_proj_fp4_short
blob_mlp_proj_fp4_short_m128_chain := mlp_proj_fp4_short_m128
$(foreach d,spike mx_rocket,$(foreach k,mlp_proj_fp4_ares mlp_proj_fp4_short_ares mlp_proj_fp4_short_chain mlp_proj_fp4_short_m128_chain,$(eval $(OUT)/$(d)/$(k): $(CURDIR)/src/mlp_proj_fp4.c $(CURDIR)/include/mx_native.h)))
$(foreach d,spike mx_rocket,$(eval $(OUT)/$(d)/mlp_proj_fp4_short: $(CURDIR)/src/mlp_proj_fp4.c))
blob_of = $(blob_$(1))

# `llama_model_l2` is llama_model.c against a 2-layer data header -- the bring-up build. The layer
# count lives in the generated data, so both are the same source; src/llama_model_l2.c is a shim
# that selects the header, exactly as llama_mlp_small.c does. The shim is the pattern rule's own
# prerequisite, so the real source has to be named as an extra one.
$(foreach d,spike mx_rocket,\
  $(foreach k,llama_model_l2 llama_model_m64 llama_model_m64l2,$(eval $(OUT)/$(d)/$(k): $(CURDIR)/src/llama_model.c)))

# The MX bisection ladder: fifteen tiny self-checking ELFs. mxl0..mxl9 each take ONE delta from
# the 64x64 ISA shape that passes, walking to the llama projection that does not; mxl10..mxl14 are
# the second tier, separating the four things that change across the mxl3-to-mxl4 cliff. Run in
# order; the first that fails names the feature. See src/mx_ladder.c and gen/gen_mx_ladder.py.
LADDER    := mxl0 mxl1 mxl2 mxl3 mxl4 mxl5 mxl6 mxl7 mxl8 mxl9 \
             mxl10 mxl11 mxl12 mxl13 mxl14 \
             mxl15 mxl16 mxl17 mxl18 mxl19 mxl20

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
$(foreach k,llama_attention_small llama_attention_tiny llama_attention_tiny_t1 llama_attention_tiny_native llama_attention_small_native llama_attention_native llama_qkv_tiny llama_proj_q llama_proj_k llama_proj_v llama_sep_qkv llama_two_mm llama_two_mm_micro,\
  $(eval $(OUT)/spike/$(k) $(OUT)/mx_rocket/$(k): $(CURDIR)/src/llama_attention.c))
$(foreach d,spike mx_rocket,\
  $(foreach k,llama_layer_full_native llama_layer_full_native_gh,$(eval $(OUT)/$(d)/$(k): $(CURDIR)/src/llama_layer_full.c $(CURDIR)/include/mx_native.h)))
$(foreach d,spike mx_rocket,\
  $(foreach k,mx_bench_matmul mx_bench_matmul_m32 mx_bench_matmul_m32_blk mx_bench_matmul_m64_proj,$(eval $(OUT)/$(d)/$(k): $(CURDIR)/src/mx_bench_matmul.c $(CURDIR)/include/mx_native.h $(CURDIR)/data/mx_bench.h)))
$(foreach d,spike mx_rocket,$(foreach k,attn_vpu_mid attn_vpu_fa,$(eval $(OUT)/$(d)/$(k): $(CURDIR)/src/attn_vpu.c)))
$(foreach d,spike mx_rocket,$(foreach k,attn_flash_mid attn_flash_llama attn_flash_llama_wide attn_flash_llama_causal attn_flash_llama_vb attn_flash_llama_2h attn_flash_llama_vb_fused attn_flash_llama_2h_fused attn_flash_llama_vb_fused_dbg attn_flash_layer5 attn_flash_layer5_fused attn_flash_llama7b attn_flash_llama7b_fused,$(eval $(OUT)/$(d)/$(k): $(CURDIR)/src/attn_flash.c)))
$(foreach d,spike mx_rocket,$(eval $(OUT)/$(d)/attn_flash_llama7b_fused: $(CURDIR)/src/attn_flash_llama7b.c))
$(foreach d,spike mx_rocket,$(eval $(OUT)/$(d)/llama_mlp_e2e_fp4: $(CURDIR)/src/llama_mlp_e2e.c $(CURDIR)/include/mx_native.h))
$(foreach d,spike mx_rocket,$(foreach k,llama_layer_e2e llama_e2e_elemwise,$(eval $(OUT)/$(d)/$(k): $(CURDIR)/src/attn_flash.c $(CURDIR)/include/mx_native.h $(CURDIR)/src/llama_layer_e2e.c)))
$(foreach k,llama_mlp_small llama_mlp_tiny llama_mlp_tiny_t1 llama_mlp_tiny_db llama_mlp_tiny_native llama_mlp_small_native llama_mlp_tiny_native_ua llama_mlp_native,\
  $(eval $(OUT)/spike/$(k) $(OUT)/mx_rocket/$(k): $(CURDIR)/src/llama_mlp.c))

# The full sub-layer kernels carry their weights as a linked BINARY BLOB, not C initializers:
# 10.8 MB (attention) and 37.2 MB (MLP), which as `0x%02x, ` text would be ~56 MB and ~220 MB of
# source -- minutes of GCC and gigabytes of RAM, if it compiled at all. objcopy derives the symbol
# names from the input PATH, so the `cd data` is load-bearing: it is what makes them
# _binary_<name>_bin_*. Default section alignment is 1, but the headers cast to uint16_t*.
define BLOB_RULE
$(OUT)/$(1)/$(2)_blob.o: $(CURDIR)/data/$(2).bin
	@mkdir -p $$(dir $$@)
	cd $(CURDIR)/data && $(OBJCOPY) -I binary -O elf64-littleriscv -B riscv \
		--set-section-alignment .data=64 $(2).bin $$@
endef
$(foreach d,spike mx_rocket,$(foreach b,$(BLOBS),$(eval $(call BLOB_RULE,$(d),$(b)))))
$(foreach d,spike mx_rocket,$(foreach k,$(BLOB_KERNELS),\
  $(eval $(OUT)/$(d)/$(k): $(OUT)/$(d)/$(call blob_of,$(k))_blob.o)))

$(OUT)/spike/%: $(CURDIR)/src/%.c $(DEPS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_BM) -DSPIKE_SIM $(EXTRA_$*) $< -o $@ $(RUNTIME) \
		$(if $(filter $(BLOB_KERNELS),$*),$(OUT)/spike/$(call blob_of,$*)_blob.o,) $(LIBS)

$(OUT)/mx_rocket/%: $(CURDIR)/src/%.c $(DEPS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_BM) -DMX_ROCKET $(EXTRA_$*) $< -o $@ $(RUNTIME) \
		$(if $(filter $(BLOB_KERNELS),$*),$(OUT)/mx_rocket/$(call blob_of,$*)_blob.o,) $(LIBS)

# The stacked-model kernels are kept OUT of the default KERNELS list: the full one carries ~1 GB of
# weights, so `make` should not build it incidentally, and both need data that only
# `gen_llama_model.py` produces.
.PHONY: model model-l2 model-m64 model-m64l2
model:    $(OUT)/spike/llama_model    $(OUT)/mx_rocket/llama_model
model-l2: $(OUT)/spike/llama_model_l2 $(OUT)/mx_rocket/llama_model_l2
model-m64:   $(OUT)/spike/llama_model_m64   $(OUT)/mx_rocket/llama_model_m64
model-m64l2: $(OUT)/spike/llama_model_m64l2 $(OUT)/mx_rocket/llama_model_m64l2

# Regenerate every data header from the newest capture. The full-attention one needs an
# --all-heads capture; see gen/gen_llama_attn_full.py.
data:
	cd $(CURDIR)/gen && $(VENV) gen_llama_layer.py mlp attn
	cd $(CURDIR)/gen && $(VENV) gen_llama_layer.py mlp  --d 256 --tag _small
	cd $(CURDIR)/gen && $(VENV) gen_llama_layer.py attn --d 256 --tag _small
	cd $(CURDIR)/gen && $(VENV) gen_llama_layer.py mlp attn --d 64 --tag _tiny
	cd $(CURDIR)/gen && $(VENV) gen_llama_attn_full.py
	cd $(CURDIR)/gen && $(VENV) gen_llama_mlp_full.py
	cd $(CURDIR)/gen && $(VENV) gen_llama_layer_full.py
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
