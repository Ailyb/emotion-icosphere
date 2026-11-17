"""Model-free tests for probekit_testset.py on a tiny synthetic GoEmotions-shaped dataset."""
import csv
import json
import tempfile
from pathlib import Path

import probekit_testset as ts

NAMES = ["admiration", "joy", "sadness", "fear", "anger", "surprise", "disgust", "love", "neutral"]
IDX = {n: i for i, n in enumerate(NAMES)}

def tsv(path, rows):
    Path(path).write_text("".join(f"{t}\t{','.join(str(IDX[l]) for l in labs)}\t{i}\n" for t, labs, i in rows))

def words(tag, k=14):
    return " ".join(f"{tag}{j}" for j in range(k))

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "emotions.txt").write_text("\n".join(NAMES))
    test = [(f"{words('t' + c + str(i))} extra", [c], f"te{c}{i}") for c in ("joy", "anger", "neutral") for i in range(6)]
    test += [("I am not happy about this at all really", ["sadness"], "teneg"),          # negation subtype
             ("so much joy and happiness in this one thread", ["neutral"], "tenw"),      # emotion_word_neutral
             ("what a plain and ordinary afternoon it was", ["anger"], "tenone"),         # no_emotion_lexicon_match
             ("multi label item here goes two", ["joy", "love"], "temulti"),              # dropped (multi)
             ("lovely admiration only item for this test", ["admiration"], "teadm"),     # dropped (unmapped)
             (words("dupdev"), ["joy"], "tedup"),
             (words("neardup", 20), ["anger"], "tenear")]                                 # 5-grams mostly inside a train passage                                          # text also in dev -> dropped
    train = [(words(f"tr{c}{i}", 12), [c], f"tr{c}{i}") for c in ("joy", "anger", "neutral", "sadness") for i in range(40)]
    dev = [(words("dupdev"), ["joy"], "dv1"), (words("neardup", 20) + " zz1 zz2 zz3 zz4 zz5 zz6", ["joy"], "dv2")]
    tsv(d / "test.tsv", test); tsv(d / "train.tsv", train); tsv(d / "dev.tsv", dev)
    for k in (1, 2, 3):                                                   # per-rater CSV: 2 raters/item, agree 100% on joy items
        with open(d / f"goemotions_{k}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "text"] + NAMES); w.writeheader()
            if k == 1:
                for t, labs, i in test:
                    for r in range(2):
                        w.writerow({"id": i, "text": t, **{n: int(n in labs and (r == 0 or n == "joy")) for n in NAMES}})
    out = d / "out"; corpus = d / "human.jsonl"

    class A: pass
    a = A(); a.data_dir, a.out_dir, a.corpus_out, a.calibration_n = str(d), str(out), str(corpus), 20
    ts.cmd_build(a)
    rows, m = ts.load_frozen(out / "goemotions_test_frozen.manifest.json")
    ids = {r["id"] for r in rows}
    assert not ({"temulti", "teadm", "tedup", "tenear"} & ids), ids                 # multi-label, unmapped, dev-duplicate dropped
    assert m["drops"]["multi_label"] == 1 and m["drops"]["unmapped_label"] == 1 and m["drops"]["text_also_in_official_train_or_dev"] == 1 and m["drops"]["near_duplicate_of_official_train_or_dev"] == 1
    cnt = m["class_counts"]; assert cnt["neutral"] <= max(cnt["joy"], cnt["anger"], cnt["sadness"]) and cnt["joy"] == 6, cnt   # neutral capped
    by = {r["id"]: r for r in rows}
    assert by["teneg"]["subtypes"] == ["negation"] or "negation" in by["teneg"]["subtypes"], by["teneg"]
    assert "emotion_word_neutral" in by["tenw"]["subtypes"] and "no_emotion_lexicon_match" in by["tenone"]["subtypes"]
    assert by["tejoy0"]["agree"] == 1.0 and by["tejoy0"]["n_raters"] == 2
    # human train corpus: only TRAIN items, none in calibration, none equal to test text, split sizes respected
    crows = [json.loads(l) for l in corpus.read_text().splitlines()]
    calib = set(json.loads((out / "goemotions_calibration_ids.json").read_text())["ids"])
    assert crows and all(r["id"].startswith("hf|goemotions|train|tr") for r in crows)
    assert not (calib & {r["id"].split("|")[-1] for r in crows}) and len(calib) == 20
    assert all(c.startswith("trneutral") for c in calib)
    sm = ts.build_human_train(ts.read_tsv(d / "train.tsv", NAMES), sorted(calib), set(), n_train=30, n_val=10)
    assert sum(r["category"] == "joy" and r["split"] == "train" for r in sm) == 30 and sum(r["category"] == "joy" and r["split"] == "val" for r in sm) == 10
    assert {r["split"] for r in sm} == {"train", "val"} and not any(r["id"].split("|")[-1] in calib for r in sm)
    # pool: existing rows first and in order, extras never in calibration/test/existing, per-class cap respected
    ex_train = [r for r in crows if r["category"] == "joy" and r["split"] == "train"]
    pl = ts.build_pool(ts.read_tsv(d / "train.tsv", NAMES), sorted(calib), {" ".join(ts.norm(r["text"])) for r in rows}, crows, cap=15, min_words=4)
    jt = [r for r in pl if r["category"] == "joy" and r["split"] == "train"]
    assert len(jt) == 15 and [r["id"] for r in jt[:len(ex_train)]] == [r["id"] for r in ex_train][:15] and [r["pool_rank"] for r in jt] == list(range(15))
    assert len({r["id"] for r in pl}) == len(pl) and not (calib & {r["id"].split("|")[-1] for r in pl})
    # leakage check passes on clean data, fails when a training text equals a test text
    b = A(); b.out_dir, b.data_dir, b.corpus, b.retired = str(out), str(d), [str(corpus)], []
    assert ts.cmd_leakage(b) == 0
    bad = crows + [dict(crows[0], id="hf|goemotions|train|xx", text=rows[0]["text"])]
    ts.write_jsonl(d / "bad.jsonl", bad); b.corpus = [str(d / "bad.jsonl")]
    assert ts.cmd_leakage(b) == 1
    # hash guard: editing the frozen set must make load_frozen refuse
    tp = out / "goemotions_test_frozen.jsonl"; tp.write_text(tp.read_text() + " ")
    try:
        ts.load_frozen(out / "goemotions_test_frozen.manifest.json"); raise AssertionError("expected refusal")
    except RuntimeError as e:
        assert "changed" in str(e)
    # determinism: rebuilding gives identical bytes
    tp.write_text(tp.read_text().rstrip(" "))
    h1 = ts.sha256_file(tp); ts.cmd_build(a); assert ts.sha256_file(tp) == h1
print("TESTSET BUILD TESTS PASSED")
