"""Model-free tests for probekit_steer_c1: pilot rule, strength plan, and the pre-registered H7a/H7b decision rules on synthetic judged rows."""
import numpy as np

import probekit_steer as S
import probekit_steer_c1 as C


def rows_for(effect_md, effect_lda, s_f, n=40, seed=0):
    """Synthetic judged rows: base text is never 'joy'/'anger'; steered text becomes the target emotion with the given probability."""
    rng = np.random.default_rng(seed); rows = []
    for p in range(n):
        rows.append({"cond": "base", "emotion": None, "rid": None, "strength": 0.0, "pid": p, "label": "neutral", "ppl_ratio": 1.0})
    for s in sorted({C.KEY(s_f["probe"]), C.KEY(s_f["v1"])}):
        for r in range(20):
            for p in range(n):
                rows.append({"cond": "random", "emotion": None, "rid": r, "strength": s, "pid": p, "label": "neutral", "ppl_ratio": 1.1})
        for cond, eff in (("probe", effect_lda), ("v1", effect_md)):
            for e in S.EMOTIONS:
                for p in range(n):
                    hit = rng.random() < eff.get(e, 0.0)
                    rows.append({"cond": cond, "emotion": e, "rid": None, "strength": s, "pid": p, "label": e if hit else "neutral", "ppl_ratio": 1.2})
    return rows


def main():
    # pilot rule: every direction must be under the limit; fall back to the smallest grid value
    t = {f"{s}|{n}": 1.0 + (s if n == "r" else 0.0) for s in C.GRID_F for n in ("a", "r")}
    assert C.pilot_rule(t, ["a", "r"], C.GRID_F, 1.3) == 0.3 and C.pilot_rule(t, ["a", "r"], C.GRID_F, 0.5) == C.GRID_F[0]
    # strength plan: shared strengths merge, types are tracked
    n = C.plan_strengths(0.4, {"probe": 0.4, "v1": 0.2})
    assert set(n) == {0.1, 0.2, 0.4} and n[0.2] == {"probe", "v1"} and n[0.4] == {"probe", "v1"}
    n2 = C.plan_strengths(0.4, {"probe": 0.6, "v1": 0.2})
    assert n2[0.3] == {"probe"} and n2[0.6] == {"probe"} and n2[0.1] == {"probe", "v1"}
    # decision rules: strong mean-difference effect on joy, anger, love; LDA nothing
    s_f = {"probe": 0.4, "v1": 0.2}
    r = C.h7(rows_for({"joy": 0.9, "anger": 0.9, "love": 0.9}, {}, s_f), 0.4, s_f)
    assert not r["H7a_claim"] and r["H7b_claim"] and set(r["H7b_above"]) >= {"joy", "anger", "love"} and len(r["H7b_above_and_md_specific"]) >= 2
    assert r["replication_joy_anger"]["joy"]["F"] and r["replication_joy_anger"]["anger"]["F"]
    # only one emotion steers -> H7b fails; all LDA steer -> H7a holds
    r1 = C.h7(rows_for({"joy": 0.9}, {}, s_f), 0.4, s_f); assert not r1["H7b_claim"]
    r2 = C.h7(rows_for({}, {e: 0.9 for e in S.EMOTIONS[:5]}, s_f), 0.4, s_f); assert r2["H7a_claim"]
    md = C.render_h7(r, 0.4, s_f); assert "H7b" in md and "claim holds" in md
    print("test_steer_c1: OK")


if __name__ == "__main__":
    main()
