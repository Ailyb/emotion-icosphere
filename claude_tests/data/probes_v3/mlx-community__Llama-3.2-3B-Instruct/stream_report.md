# Stream check (92 hand-written passages, layer 14, method meandiff)

| rule | 8-way accuracy (emotional passages) | emotional-vs-neutral AUC |
|---|---|---|
| pooled | 0.562 | 0.833 |
| mean_z | 0.548 | 0.826 |
| max_z | 0.562 | 0.927 |
| late_mean_z | 0.589 | 0.812 |

- true emotion reaches z>=2 in 89% of emotional passages; median onset at 0.2857142857142857 of the passage (0 = first token, 1 = last)
- share of tokens where the true emotion is the top emotion: 0.41

Open stream_view.html and read ~10 passages: do the highlighted tokens make sense, or does it light up on stray words (names, punctuation)?
