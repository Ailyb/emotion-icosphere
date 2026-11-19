"""Model-free tests for probekit_t11 helpers (EmoLex predictor, id-parity split, tokenizer)."""
import numpy as np

import probekit_t11 as t


def main():
    lex = {e: set() for e in t.LEX_EMOTIONS}
    lex["joy"] |= {"happy", "glad"}; lex["anger"] |= {"hate"}; lex["sadness"] |= {"cry"}
    p = t.emolex_predict(["So happy and glad", "I hate it", "The table is brown", "happy but I hate it and cry", "It’s fine"], lex,
                         ["joy", "anger", "sadness", "neutral"])
    assert list(p) == ["joy", "anger", "neutral", "joy", "neutral"], p            # ties -> fixed order (joy before sadness/anger)
    q = t.emolex_predict(["so happy", "I hate it"], lex, ["anger", "sadness"])     # joy not a class of the source: never predicted
    assert list(q) == ["neutral", "anger"], q
    assert t.emolex_predict(["love love"], lex, ["love"])[0] == "neutral"          # no lexicon entry for love
    assert t.toks("Don’t STOP, it's 3am!") == ["don't", "stop", "it's", "am"]
    ids = ["b", "d", "a", "c", "e"]
    even = t.parity_split(ids)
    assert list(np.array(ids)[even]) == ["a", "c", "e"] and list(np.array(ids)[~even]) == ["b", "d"], even
    assert t._key("x") == t._key("x") and t._key("x") != t._key("y")
    print("test_t11: OK")


if __name__ == "__main__":
    main()
