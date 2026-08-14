# tests/conftest.py
#
# Runs before any test module is imported.
#
# WHY THE OPENMP SETTINGS:
# Two copies of libomp end up in the same process on macOS — one inside
# faiss-cpu, one that PyTorch brings in. PyTorch is not a direct dependency of
# these tests; it arrives transitively because langchain_core imports
# `transformers` for GPT2TokenizerFast when transformers happens to be
# installed, and transformers imports torch. The second runtime to initialise
# aborts the interpreter:
#
#   OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
#   initialized.
#
# The abort lands inside faiss's search, which makes it look like a bug in the
# matcher. It is not — it is a packaging collision in the local environment. CI
# installs faiss-cpu without torch, so it never fires there.
#
# KMP_DUPLICATE_LIB_OK allows the process to continue. Its documented risk is
# silently wrong results from OpenMP-parallel numeric code, so the thread limits
# below remove the parallelism that risk depends on: the tests search a handful
# of ten-dimensional vectors, where threading buys nothing anyway.

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
