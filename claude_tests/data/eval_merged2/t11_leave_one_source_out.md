# Leave-one-source-out robustness of the cross-domain conclusions (POST HOC; from the committed T11 reports, no refitting)

Each block drops one external source and re-applies the rules to the other two (both must hold). ISEAR: Hub copy with unverified provenance, 5 classes (no neutral, surprise or love), 300 per class; the one source where the sentence encoder's lead over the probe is largest.

| dropped | model | H1 probe > TF-IDF on both remaining | H3 probe > encoder on both remaining | probe - encoder (means) | seed sign stable: TF-IDF / encoder |
|---|---|---|---|---|---|
| dair | Llama-3.2-3B-Instruct | yes | no | isear -0.090, dailydialog -0.038 | yes / yes |
| dair | Mistral-7B-Instruct-v0.3-4bit | yes | no | isear -0.095, dailydialog -0.087 | yes / yes |
| dair | gemma-2-2b-it-4bit | yes | no | isear -0.032, dailydialog -0.046 | yes / yes |
| isear | Llama-3.2-3B-Instruct | yes | no | dair +0.003, dailydialog -0.038 | yes / no |
| isear | Mistral-7B-Instruct-v0.3-4bit | yes | no | dair +0.004, dailydialog -0.087 | yes / no |
| isear | gemma-2-2b-it-4bit | yes | no | dair -0.011, dailydialog -0.046 | yes / yes |
| dailydialog | Llama-3.2-3B-Instruct | yes | no | dair +0.003, isear -0.090 | yes / no |
| dailydialog | Mistral-7B-Instruct-v0.3-4bit | yes | no | dair +0.004, isear -0.095 | yes / no |
| dailydialog | gemma-2-2b-it-4bit | yes | no | dair -0.011, isear -0.032 | yes / yes |

## Reading

- Without ISEAR: H1 holds for 3/3 models; H3 holds for 0/3. With dair and DailyDialog alone the probe ties the encoder on dair and loses on DailyDialog for every model, so the H3 verdict does not depend on ISEAR.
- H1 (probe beats TF-IDF) holds under every single-source drop for 3/3 models.
- H3 holds under any single-source drop for 0/3 models.
