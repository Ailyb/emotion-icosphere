# T11 summary across models (H3 rule: probe - encoder CI > 0 on >= 2 of 3 external sources, on >= 2 of 3 models)

| model | external sources with probe - encoder CI > 0 | per-model condition |
|---|---|---|
| mlx-community__gemma-2-2b-it-4bit | 0/3 [] | not met |
| mlx-community__Llama-3.2-3B-Instruct | 0/3 [] | not met |
| mlx-community__Mistral-7B-Instruct-v0.3-4bit | 0/3 [] | not met |

Models meeting the condition: 0 of 3 -> H3 NOT SUPPORTED (variant B).

| model | tau_h | odd-half neutral recall (orig -> tau_h) | frozen F1 diff [CI] | fixes? |
|---|---|---|---|---|
| mlx-community__gemma-2-2b-it-4bit | 5.43 | 0.31 -> 0.90 | -0.173 [-0.213, -0.135]* | no |
| mlx-community__Llama-3.2-3B-Instruct | 4.77 | 0.18 -> 0.90 | -0.121 [-0.163, -0.084]* | no |
| mlx-community__Mistral-7B-Instruct-v0.3-4bit | 4.42 | 0.14 -> 0.90 | -0.134 [-0.173, -0.096]* | no |
