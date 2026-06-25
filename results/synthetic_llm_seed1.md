## synthetic | embedder=st:BAAI/bge-small-en-v1.5 | extractor=openai:qwen2.5:7b | test queries=100

| system | resolution | nDCG@10 | Recall@10 | all-evidence@10 | MRR |
|---|---|---|---|---|---|
| dense only | n/a | 0.383 | 0.572 | 0.220 | 0.874 |
| graph only (tuned) | full | 0.635 | 0.913 | 0.840 | 0.900 |
| hybrid (default params) | full | 0.543 | 0.850 | 0.690 | 0.868 |
| hybrid (tuned) | full | 0.642 | 0.922 | 0.860 | 0.912 |
| graph only (tuned) | none | 0.447 | 0.678 | 0.420 | 0.816 |
| hybrid (default params) | none | 0.434 | 0.683 | 0.420 | 0.852 |
| hybrid (tuned) | none | 0.444 | 0.683 | 0.420 | 0.840 |

Graph size and tuned-hybrid gain over dense (paired bootstrap, 95% CI):

- full: 191 entities, 309 relations; delta nDCG = +0.259 [+0.212, +0.304]
- none: 292 entities, 310 relations; delta nDCG = +0.061 [+0.019, +0.103]

Entity resolution vs gold mentions (B-cubed):

| resolution | precision | recall | F1 | mention coverage | gold entities | predicted nodes |
|---|---|---|---|---|---|---|
| full | 0.988 | 0.987 | 0.987 | 0.993 | 176 | 179 |
| none | 0.997 | 0.749 | 0.855 | 0.993 | 176 | 277 |

nDCG@10 by question type:

| type | dense | hybrid tuned (full) | hybrid tuned (none) |
|---|---|---|---|
| 1hop_hq | 0.895 | 0.900 | 0.800 |
| 2hop_company_birthplace | 0.257 | 0.510 | 0.339 |
| 2hop_founder_hq | 0.209 | 0.438 | 0.310 |
| 2hop_product_founder | 0.275 | 0.766 | 0.440 |
| 3hop_product_mentor | 0.256 | 0.627 | 0.323 |
