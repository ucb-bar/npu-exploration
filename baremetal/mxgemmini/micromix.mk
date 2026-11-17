# Opt-in MicroMX targets. Existing all/spike/mx_rocket behavior is unchanged.
HOST_CC ?= cc
PYTHON ?= python3
MICROMIX_RISCV ?= $(RISCV)
MICROMIX_ROCC ?= $(ROCC)
MICROMIX_LIB ?= $(abspath $(REPO)/../software/libgemmini/libgemmini_mxbf16_dim16.so)
MICROMIX_EXTENSION ?= gemmini_mxbf16_dim16
.PHONY: micromix-test micromix-spike-test micromix-attention-test
micromix-test:
	$(PYTHON) tests/run_micromix_tests.py --cc $(HOST_CC)
	$(PYTHON) -m unittest discover -s tests -p test_micromix_port_audit.py -v
micromix-spike-test:
	$(PYTHON) tests/run_micromix_spike.py --riscv "$(MICROMIX_RISCV)" --rocc "$(MICROMIX_ROCC)" --library "$(MICROMIX_LIB)" --extension "$(MICROMIX_EXTENSION)"
micromix-attention-test:
	$(PYTHON) tests/run_mx_llm_tests.py --cc $(HOST_CC)
.PHONY: micromix-gen-test micromix-prefill
micromix-gen-test:
	$(PYTHON) tests/test_micromix_generators.py -v
# User supplies capture, weight and independent numeric-reference paths.
# Model/mix and output namespace are explicit to avoid implicit profile changes.
micromix-prefill:
	$(PYTHON) gen/build_micromix_prefill.py --riscv "$(MICROMIX_RISCV)" --rocc "$(MICROMIX_ROCC)" $(MICROMIX_PREFILL_ARGS)

.PHONY: micromix-layer-spike-test
micromix-layer-spike-test:
	$(PYTHON) tests/run_micromix_layer_spike.py --riscv "$(MICROMIX_RISCV)" --rocc "$(MICROMIX_ROCC)" --library "$(MICROMIX_LIB)" --extension "$(MICROMIX_EXTENSION)"

.PHONY: micromix-decode micromix-decode-test
# Builds only; explicit input exports and a fresh output root are required.
micromix-decode:
	$(PYTHON) gen/build_micromix_decode.py $(MICROMIX_DECODE_ARGS)
micromix-decode-test:
	$(PYTHON) tests/test_micromix_decode.py $(MICROMIX_DECODE_TEST_ARGS)
