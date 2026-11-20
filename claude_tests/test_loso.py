"""Model-free test of probekit_loso.loso on a synthetic result set."""
import probekit_loso as lo


def cell(m, lo_, hi): return [m, lo_, hi]


def main():
    sets = {s: {"probe_minus": {"tfidf": cell(.1, .05, .15), "encoder": cell(-.05, -.08, -.02)}} for s in lo.SOURCES}
    sets["dair"]["probe_minus"]["encoder"] = cell(.04, .01, .07)             # probe beats the encoder on dair only
    st = {f"{s}|{c}": {"stable": True} for s in lo.SOURCES for c in ("tfidf", "encoder")}
    r = lo.loso({"m": {"sets": sets, "h6_stability": st}})
    assert r["isear"]["m"]["h1"] and not r["isear"]["m"]["h3"]                 # dair + dailydialog: probe loses to encoder on dailydialog
    assert not r["dailydialog"]["m"]["h3"] and not r["dair"]["m"]["h3"]
    sets["dailydialog"]["probe_minus"]["encoder"] = cell(.04, .01, .07)
    assert lo.loso({"m": {"sets": sets, "h6_stability": st}})["isear"]["m"]["h3"]   # both remaining sources favour the probe
    print("test_loso: OK")


if __name__ == "__main__":
    main()
