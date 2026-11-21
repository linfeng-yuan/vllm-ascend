# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path


def test_hisparse_in_three_suite_contexts():
    tests = Path(__file__).with_name("singlecard_ops")
    hisparse = tests / "test_hisparse_arch22_ops.py"
    earlier_modules = sorted(case.stem for case in tests.glob("test_*.py") if case.name <= hisparse.name)
    assert hisparse.stem in earlier_modules
    print(f"Suite prefix modules: {earlier_modules}", flush=True)

    cases = (
        ("isolated", [str(hisparse)]),
        ("full collection", [str(tests), "-k", hisparse.stem]),
        ("full collection and prefix", [str(tests), "-k", " or ".join(earlier_modules)]),
    )
    results = {}
    for label, args in cases:
        print(f"HiSparse diagnostic start: {label}", flush=True)
        process = subprocess.run([sys.executable, "-m", "pytest", "-sv", *args], check=False, timeout=900)
        results[label] = process.returncode
        print(f"HiSparse diagnostic result: {label} exit={process.returncode}", flush=True)

    assert all(code == 0 for code in results.values()), results
