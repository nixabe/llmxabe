# Builds the llama.cpp reference-tensor capture program.
#
# This is the generator for the golden data that `crates/xabe-engine/tests/`
# gates the forward pass against. It builds OUT OF TREE against llama.cpp's
# installed headers and shared objects — no llama.cpp source is modified, and
# it registers the same public `llama_context_params::cb_eval` hook that
# `llama-eval-callback` uses, with a binary sink instead of a printer.
#
# See ../../docs/ORACLE.md for the full procedure, the prompt, and the
# fidelity controls.

LLAMA ?= /home/nixabe/llama.cpp
OUT   ?= capture

$(OUT): capture.cpp
	g++ -O2 -std=c++17 capture.cpp -o $(OUT) \
	    -I$(LLAMA)/include -I$(LLAMA)/ggml/include \
	    -L$(LLAMA)/build/bin -lllama -lggml-base -lggml \
	    -Wl,-rpath,$(LLAMA)/build/bin

clean:
	rm -f $(OUT)

.PHONY: clean
