#!/usr/bin/env python3
import numpy as np
import probekit_viz_stream as S

st = np.arange(20, dtype=float)[:, None] * np.ones((1, 3))
M, ts = S.window_means(st, 4, 4)
assert ts[0] == 4 and abs(M[0, 0] - 4) < 1e-6 and abs(M[4, 0] - np.mean([5, 6, 7, 8])) < 1e-6          # t=8 window = tokens 5..8
M, ts = S.window_means(st, 4, "prefix"); assert abs(M[3, 0] - np.mean([4, 5, 6, 7])) < 1e-6
ts = np.arange(4, 40); flag = np.zeros(len(ts), bool); top = np.zeros(len(ts), int)
flag[(ts >= 14) & (ts <= 22)] = True; top[:] = 2
m = S.splice_metrics(flag, top, ts, e_start=12, e_end=20, intended=2)
assert m["hit"] and m["onset"] == 2 and m["offset"] == 2 and m["offset_run"] == 2
m = S.splice_metrics(flag, top, ts, 12, 20, intended=1); assert not m["hit"] and m["onset"] is None
flag[:] = False; m = S.splice_metrics(flag, top, ts, 12, 20, 2); assert not m["hit"] and m["offset"] == 0
streams = S.build_streams([dict(bucket="8-11", intended=c, text=f"{c} {i}") for c in ["neutral"] + S.EMO for i in range(40)], "val")
assert sum(s["kind"] == "neutral" for s in streams) == 140 and sum(s["kind"] == "splice" for s in streams) == 140
assert all(len(s["sents"]) == 5 and s["sents"][2].split()[0] == s["emotion"] for s in streams if s["kind"] == "splice")
val_txt = {t for s in streams for t in s["sents"] if s["kind"] == "splice"}; test = S.build_streams([dict(bucket="8-11", intended=c, text=f"{c} {i}") for c in ["neutral"] + S.EMO for i in range(40)], "test")
assert not (val_txt & {t for s in test for t in s["sents"] if s["kind"] == "splice"})
print("PASS test_viz_stream")
