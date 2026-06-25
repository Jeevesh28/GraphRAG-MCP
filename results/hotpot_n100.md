## hotpot | embedder=st:BAAI/bge-small-en-v1.5 | extractor=openai:qwen2.5:7b | test queries=50

| system | resolution | nDCG@10 | Recall@10 | all-evidence@10 | MRR |
|---|---|---|---|---|---|
| dense only | n/a | 0.880 | 0.960 | 0.940 | 0.937 |
| graph only (tuned) | full | 0.722 | 0.890 | 0.780 | 0.755 |
| hybrid (default params) | full | 0.832 | 0.990 | 0.980 | 0.820 |
| hybrid (tuned) | full | 0.884 | 0.960 | 0.940 | 0.953 |
| graph only (tuned) | none | 0.740 | 0.960 | 0.920 | 0.765 |
| hybrid (default params) | none | 0.828 | 0.990 | 0.980 | 0.824 |
| hybrid (tuned) | none | 0.885 | 0.960 | 0.940 | 0.953 |

Graph size and tuned-hybrid gain over dense (paired bootstrap, 95% CI):

- full: 5341 entities, 5607 relations; delta nDCG = +0.004 [-0.022, +0.030]
- none: 5845 entities, 5725 relations; delta nDCG = +0.005 [-0.021, +0.032]

nDCG@10 by question type:

| type | dense | hybrid tuned (full) | hybrid tuned (none) |
|---|---|---|---|
| hotpot_bridge | 0.854 | 0.858 | 0.859 |
| hotpot_comparison | 0.984 | 0.984 | 0.992 |
