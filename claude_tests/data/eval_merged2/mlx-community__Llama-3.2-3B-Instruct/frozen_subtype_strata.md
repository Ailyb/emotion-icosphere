# Lexicon-subtype strata on the GoEmotions frozen set (NRC EmoLex sidecar; exploratory, post hoc)

Hypothesis stated before running: probe - TF-IDF is larger (less negative) in `no_lexicon_match` than in `lexicon_match`. Macro-F1 over the classes present in each stratum; a stratum with a single class is not meaningful.

| stratum | n | probe_lda | native_lr | embed | tfidf | fusion_tfidf_native (post hoc) | probe - TF-IDF [95% CI] |
|---|---|---|---|---|---|---|---|
| no_lexicon_match | 151 | 0.515 | 0.513 | 0.381 | 0.473 | 0.566 | +0.042 [-0.060, +0.149] |
| lexicon_match | 538 | 0.608 | 0.591 | 0.483 | 0.699 | 0.701 | -0.091 [-0.142, -0.036]* |
| emotion_word_neutral | 96 | 0.345 | 0.476 | 0.286 | 0.523 | 0.500 | -0.177 [-0.323, -0.026]* |
| plain_neutral | 55 | 0.281 | 0.514 | 0.358 | 0.625 | 0.659 | -0.346 [-0.509, -0.170]* |
| negation | 54 | 0.500 | 0.481 | 0.416 | 0.619 | 0.515 | -0.112 [-0.299, +0.072] |

Sidecar manifest: {"lexicon_match": 538, "emotion_word_neutral": 96, "negation": 54, "no_lexicon_match": 151, "plain_neutral": 55}; lexicon sha256 02c661544f4f...
