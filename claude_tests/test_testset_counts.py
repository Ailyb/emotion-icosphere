"""Model-free test of probekit_testset_counts agreement rules on synthetic rater data (skipped without pandas)."""
try:
    import pandas as pd
except ImportError:
    print("pandas not installed: skipped"); raise SystemExit(0)
import probekit_testset_counts as tc

F = frozenset
# per-item rater label sets: (sets, very_unclear_flag, text)
items = {
    "a": ([F({"joy"}), F({"joy"}), F({"joy"})], False, "word " * 12),                       # strict joy
    "b": ([F({"joy"}), F({"joy", "love"})], False, "word " * 12),                          # shared joy, not strict
    "c": ([F({"joy"}), F({"anger"})], False, "word " * 12),                                # no agreement
    "d": ([F({"admiration"}), F({"admiration"})], False, "word " * 12),                    # agreed but unmapped
    "e": ([F({"neutral"}), F({"neutral"})], True, "word " * 12),                           # flagged unclear: fails strict
    "f": ([F({"love"})], False, "word " * 12),                                             # single rater: never agreed
    "g": ([F({"sadness"}), F({"sadness"})], False, "too short"),                           # fails length filter only
}
assert tc.agreed_label(*items["a"][:2], "strict") == "joy" and tc.agreed_label(*items["a"][:2], "shared") == "joy"
assert tc.agreed_label(*items["b"][:2], "strict") is None and tc.agreed_label(*items["b"][:2], "shared") == "joy"
assert tc.agreed_label(*items["c"][:2], "shared") is None
assert tc.agreed_label(*items["e"][:2], "strict") is None and tc.agreed_label(*items["e"][:2], "shared") == "neutral"
assert tc.agreed_label(*items["f"][:2], "strict") is None
t = tc.count_table(items, list(items), {"a": "joy", "d": "admiration", "x": "fear"})
s_all, s_len = t[("strict", False)], t[("strict", True)]
assert s_all["joy"] == 1 and s_all["sadness"] == 1 and s_all["_unmapped"] == 1 and s_all["_no_agreement"] == 4, s_all
assert s_len["sadness"] == 0 and s_len["joy"] == 1                                         # length filter drops "too short"
assert t[("shared", False)]["joy"] == 2 and t[("shared", False)]["neutral"] == 1
assert t[("simple", False)]["joy"] == 1 and t[("simple", False)]["_unmapped"] == 1 and t[("simple", False)]["fear"] == 0
md = tc.render(t, 7, 7); assert "rule=strict" in md and "classes with n >= 50" in md
print("TESTSET COUNTS TESTS PASSED")
