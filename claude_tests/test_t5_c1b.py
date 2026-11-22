"""Model-free tests for probekit_steer_c1b (strength plan) and probekit_t5 (split rule, label mapping)."""
import probekit_steer_c1b as B
import probekit_t5 as T


def main():
    assert B.plan({"probe": 0.4, "v1": 0.3}) == {0.2: {"probe"}, 0.4: {"probe"}, 0.15: {"v1"}, 0.3: {"v1"}}
    assert B.plan({"probe": 0.4, "v1": 0.4}) == {0.2: {"probe", "v1"}, 0.4: {"probe", "v1"}}
    sp = [T.split_of(i) for i in range(100)]
    assert sp.count("test") == 20 and sp.count("val") == 20 and sp.count("train") == 60
    assert sp[0] == "test" and sp[1] == "val" and sp[2] == "train"
    assert set(T.MAP.values()) == {"joy", "sadness", "fear", "anger", "surprise", "disgust", "love"}
    print("T5/C1b TESTS PASSED")


if __name__ == "__main__":
    main()
