#!/usr/bin/env python3
"""Run pytest with one Torch thread after the repository conftest is loaded."""

import os
import sys

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1")

import pytest


class CPUProfile:
    @pytest.hookimpl(trylast=True)
    def pytest_configure(self, config):
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:], plugins=[CPUProfile()]))
