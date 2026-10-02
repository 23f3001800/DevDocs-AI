# Clean retrieval benchmark — 2 October 2026

[Successful run](https://github.com/23f3001800/DevDocs-AI/actions/runs/37021038575), commit `11ba42d511c72d8da6a453315a4e8b3ff9e4d1ff`.

The earlier run `37018616052` indexed the entire evaluation PDF, including questions and expected answers. Its scores are invalid as retrieval-quality evidence and must not be used in the resume or portfolio.

This rerun extracts only the PDF's 13 source documents, stores document identity in metadata and excludes the answer key. It evaluates 19 answerable questions at k=5, repeated three times. TC-13 is unanswerable and is excluded from retrieval averages; abstention and answer generation were not tested. Eleven local evaluation-contract tests passed, including answer-key exclusion and identity scoring.

| Retriever | Recall@5 | MRR@5 | Hit@5 | Warm p50 ms | Warm p95 ms |
|---|---:|---:|---:|---:|---:|
| Dense | 0.8605 | 0.9211 | 0.9474 | 2.84 | 2.88 |
| BM25 | 0.8605 | 0.8526 | 0.9474 | 0.17 | 0.25 |
| Hybrid | 0.8605 | 0.8860 | 0.9474 | 2.24 | 2.35 |
| Hybrid + reranking | 0.8605 | 0.9474 | 0.9474 | 291.88 | 309.71 |

All three passes produced the same quality scores. Latency summarizes passes two and three (38 observations), with query embedding caches warm; it is not cold-start or production latency. The CPU runner reported four logical CPUs. Dependency versions are recorded alongside this report.

Reranking improved MRR relative to hybrid retrieval on this fixture, without increasing recall. It also added substantial CPU latency. That tradeoff does not establish a production-wide improvement. Dense retrieval had higher MRR than hybrid without reranking here.

## Evidence and remaining limits

- `receipt.json` retains run metadata, all three metric sets, per-query timing and ranked document IDs.
- `source-corpus.json` contains the exact extracted source text; original per-case contexts remain in the run artifact and are reconstructible from this corpus.
- This is a small controlled fixture, not a held-out external documentation benchmark. Its source documents describe intended behaviour and are not proof every described feature exists in the application.
- The broad pipeline and ingestion questions still miss some required documents at k=5. Do not tune repeatedly on these questions and call them independent test results.
- Full RAG evaluation still requires real generated answers, abstention checks and groundedness assessment with a configured provider. No faithfulness, answer-accuracy or TTFT claim is made.

The corrected results replace invalid evidence; a before/after percentage improvement would be misleading because the corpus and scoring protocol changed.
