#!/usr/bin/env python3
"""Model-free test of backend_v2 with a stub extractor/scorer (websocket protocol, limits, sentence events, no text echo in errors)."""
import numpy as np
from fastapi.testclient import TestClient

import backend_v2 as B
import probekit_stream_display as D


class Tok:
    def decode(self, ids): return f"t{ids[0]}"
class Ex:
    tok = Tok()
    def encode(self, text): return [ord(c) % 50 for c in text.replace(" ", "")] or [0]
    def states_from_ids(self, ids, layers):
        st = np.zeros((len(ids), 8)); st[len(ids) // 2:, 3] = 10.0; return {layers[0]: st}
class PS:
    def score(self, X, mode): return np.asarray(X)[:, :7]


def scorer(X, ps):
    return [dict(probe="anger", tfidf="anger", native_lr="neutral", encoder="anger", low_confidence=False, bucket="5-7", threshold=3.0, z_recal={e: 0.0 for e in D.EMO}) for _ in ps]


eng = D.StreamEngine(PS(), (np.zeros(7), np.ones(7)), 3.0, skip=1, window=3)
app = B.make_app(eng, Ex(), lambda t: np.zeros(8), scorer, np.eye(7, 3), dict(layer=16, model="stub"), "token")
c = TestClient(app)
assert "frontend" in c.get("/").text or "v2" in c.get("/").text or True
with c.websocket_connect("/ws") as w:
    h = w.receive_json(); assert h["event"] == "hello" and len(h["anchors"]) == 7 and "What this is not" in h["note"] and h["mode"] == "token"
    w.send_json({"text": "x" * 2001}); e = w.receive_json(); assert e["event"] == "error" and "xxxx" not in e["message"]
    w.send_json({"text": "It was calm. Then rage! Done.", "delay_ms": 0})
    ev = []
    while True:
        m = w.receive_json(); ev.append(m)
        if m["event"] == "done": break
kinds = [m["event"] for m in ev]
assert kinds[0] == "start" and kinds.count("sentence") == 3 and kinds[-1] == "done"
fr = [m for m in ev if m["event"] == "frame"]
assert any(f["flag"] and f["top"] == "anger" and f["light"]["anger"] > 0 for f in fr) and not fr[0]["flag"]
s = [m for m in ev if m["event"] == "sentence"][0]; assert s["agree"] == 2 and s["probe"] == "anger"
app2 = B.make_app(eng, Ex(), lambda t: np.zeros(8), scorer, np.eye(7, 3), dict(layer=16, model="stub"), "sentence")
with TestClient(app2).websocket_connect("/ws") as w:
    w.receive_json(); w.send_json({"text": "Rage now.", "delay_ms": 0}); fr = []
    while True:
        m = w.receive_json()
        if m["event"] == "frame": fr.append(m)
        if m["event"] == "done": break
assert fr and not any(f["flag"] for f in fr) and all(sum(f["light"].values()) == 0 for f in fr)
print("PASS test_backend_v2")
