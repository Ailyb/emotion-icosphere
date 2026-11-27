#!/usr/bin/env python3
"""Model-free test of probekit_viz_live request logic and the HTTP shell (stub extractor/scorer)."""
import json
import threading
import urllib.request
from http.server import HTTPServer

import numpy as np
import probekit_viz_live as L
import probekit_viz_static as V

row = dict(n_words=3, bucket="3-4", z_recal={"joy": 1.0}, threshold=3.0, low_confidence=True, probe="neutral", tfidf="joy", native_lr="joy", encoder="joy", text="SECRET", id="live", intended="unknown")
seen = []
def scorer(X, ps):
    seen.append((X.shape, ps)); return [dict(row)]
extract = lambda t: np.zeros(5)

assert L.handle_score({"text": ""}, extract, scorer)[0] == 400 and L.handle_score({}, extract, scorer)[0] == 400 and L.handle_score([], extract, scorer)[0] == 400
assert L.handle_score({"text": "x" * 2001}, extract, scorer)[0] == 400
code, body = L.handle_score({"text": "hello there friend"}, extract, scorer)
assert code == 200 and "text" not in body and body["probe"] == "neutral" and seen[-1][0] == (1, 5)

srv = HTTPServer(("127.0.0.1", 0), L.make_handler(extract, scorer)); port = srv.server_address[1]
threading.Thread(target=lambda: [srv.handle_request() for _ in range(3)], daemon=True).start()
page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
assert "What this is not" in page and "not stored or logged" in page
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/score", data=json.dumps({"text": "hi there"}).encode(), headers={"Content-Type": "application/json"}))
assert json.loads(r.read())["bucket"] == "3-4"
try:
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/score", data=b"{}", headers={"Content-Type": "application/json"})); raise SystemExit("expected 400")
except urllib.error.HTTPError as e:
    assert e.code == 400
assert V.NOT in L._page()
print("PASS test_viz_live")
