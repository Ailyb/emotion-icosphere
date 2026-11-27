#!/usr/bin/env python3
import numpy as np
import probekit_judge_gen as G
assert abs(G.kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) - 1) < 1e-9
assert abs(G.kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9
assert abs(G.kappa(["a", None, "b"], ["a", "a", "b"]) - 1) < 1e-9
lo, hi = G.boot_kappa(["a", "b"] * 20, ["a", "b"] * 20, B=50); assert lo >= 0.99
print("PASS test_judge_gen")
