"""Model-free tests for probekit_judge parsing and threaded judging against a fake server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import probekit_judge as pj

assert pj.parse_label("joy") == "joy" and pj.parse_label("  Anger.\n") == "anger" and pj.parse_label("The answer is fear") == "fear"
assert pj.parse_label("") is None and pj.parse_label(None) is None and pj.parse_label("happiness") is None
assert pj.parse_label("disgusted") is None                              # whole words only, no stemming

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert b["temperature"] == 0 and b["reasoning_effort"] == "none" and b["max_tokens"] == 8
        t = b["messages"][0]["content"]
        ans = "sadness" if "grief" in t else "joy" if "party" in t else "I do not know"
        out = json.dumps({"choices": [{"message": {"content": ans}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}/v1"
got = pj.judge_texts(["a party with cake", "so much grief today", "nothing special", 'quote " inside party'], base=base, workers=3)
assert got == ["joy", "sadness", None, "joy"], got                       # order preserved under threads; unparsed -> None; quotes sanitised
srv.shutdown()
print("JUDGE TESTS PASSED")
