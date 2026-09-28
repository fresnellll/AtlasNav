# AtlasNav

AtlasNav is a persistent multi-view Corpus Atlas for finite-budget agentic
research. It organizes every canonical file once through four complementary
views—Topic, Identity, Episode, and Relation—and trains a lightweight
query-adaptive Router to combine those semantic channels with BM25. The full
corpus remains searchable and directly openable: the Atlas is a map, not a
query-specific candidate boundary.

This public release supports two different, explicit workflows:

1. **Build and run from upstream inputs.** Start with canonical documents and
   questions, build the four-view embeddings, multiplex hierarchy, Router,
   runtime, trajectories, judgments, and evaluation report.
2. **Replay the available frozen paper records.** Download the companion
   artifact release and recompute the covered tables from checksummed
   trajectories and per-query records without paid API calls. EnterpriseRAG
   also exposes an optional OpenAI-compatible selector for independent
   document-metric reproduction.

The workflows are intentionally separate. Rebuilding a stochastic Atlas or
Router is a method reproduction; replaying a released frozen artifact is an
exact result reproduction for that artifact's declared coverage.

Auxiliary benchmarks use the same pipeline, not separate research launchers.
`atlasnav benchmarks prepare` converts upstream PhantomWiki,
EnterpriseRAG-Bench, FanOutQA, 2WikiMultiHopQA, or BEIR-format data into one
leakage-isolated bundle. See [Benchmarks](docs/BENCHMARKS.md).

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[all]'
atlasnav doctor
pytest -q
```

`doctor` validates optional build dependencies, all model profiles, release
hygiene, and accidental credential or machine-path leakage. No API host or
secret is stored in this repository.

## Frozen result replay

```bash
export ATLASNAV_ARTIFACT_REPO='fresnellll/atlasnav-artifacts'
atlasnav artifacts download --output artifacts/release
atlasnav artifacts verify --artifact-root artifacts/release --deep
atlasnav reproduce \
  --suite paper \
  --artifact-root artifacts/release \
  --output reproduced
```

The reproduction command verifies the artifact contract and recomputes the
released BrowseComp-Plus endpoint, recorded query-time agent inference cost,
paired outcome cells, passive
turn/cost endpoints, frozen checkpoint-manifest integrity, Evidence Blindness,
and available auxiliary results. It also asserts the corresponding
paper-precision endpoint, Evidence-Blindness, checkpoint, and auxiliary table
cells. DeepSeek and ChatGPT include one reported full trajectory for every
query under AtlasNav,
DR-DCI, and raw DCI. The ChatGPT evidence-supplied Gold-document Reference is
also a complete 830-query trajectory package, so its reported `801/830`
reference accuracy is recomputed from per-query judgments rather than read from
a summary. Qwen and MiMo are summary-only in the local release because their
final externally consolidated trajectories are not recoverable from this
server. Counterfactual active Safe Release is kept
separate from passive replay and is never reconstructed by pretending an
unfinished trajectory had already answered.
Agent and judge costs retain separate ledgers and currencies. The paper cost
metric uses query-time agent inference only; judge cost is never added to that
metric, and unlike currencies are never silently combined.

EnterpriseRAG frozen answers and official judgements are replayed directly;
Document Recall and Invalid Extra can additionally be recomputed with
`atlasnav enterprise evaluate` using an OpenAI-compatible API. The selector
does not alter the frozen answer evaluation.

For the released package, the no-API command is:

```bash
atlasnav enterprise evaluate \
  --questions artifacts/release/enterpriserag/questions.jsonl \
  --answers artifacts/release/enterpriserag/answers.jsonl \
  --results artifacts/release/enterpriserag/results.json \
  --selection-results artifacts/release/enterpriserag/document_selection/document_selection_results.json \
  --output reproduced/enterpriserag
```

To reconstruct document selection with an API, add `--audit`, `--catalog`,
`--base-url`, `--model`, and `--api-key-env` instead of `--selection-results`.

The companion release also contains the exact final frozen Atlas and four-view
query vectors for every reported auxiliary benchmark: PhantomWiki 10K/50K/
100K/1M, EnterpriseRAG-Bench, FanOutQA, TREC-COVID, 2Wiki-Global-400,
SciFact, and ArguAna. Each Atlas contains the four reduced document-vector
matrices, PCA transforms, sparse graphs, hierarchy arrays, cards, labels, and
aligned catalog. The full 2,560-dimensional document matrices are a
reproducible construction intermediate and are not duplicated in the release.
Use the upstream corpus with the frozen Atlas and query vectors for an exact
runtime rebuild, or rerun `atlasnav build embeddings` for a construction
reproduction.

For BrowseComp-Plus, the original full-830 query-vector bundle is no longer
available. The release therefore preserves the exact checksummed post-Router
weighted-RRF score for all 830 x 100,195 query/document pairs, plus its query-ID
catalog and retrieval audit. With the upstream corpus/full-text index and the
frozen Atlas, `atlasnav runtime build --frozen-ranking ...` recreates the exact
initial routes without an embedding API call. This later-stage frozen
representation is labeled explicitly; it is not presented as recovered raw
query embeddings.

## Build from zero

The minimal corpus schema is JSONL or Parquet with `docid`, `text`, and an
optional `url`. The minimal question schema is JSONL with `query_id` and
`query`; add `answer` for judging.

```bash
atlasnav corpus prepare --input upstream/documents.jsonl --output work/corpus
atlasnav corpus build-index --corpus work/corpus --output work/fulltext.sqlite3

export ATLASNAV_EMBEDDING_API_KEY='...'
atlasnav build embeddings \
  --corpus work/corpus --output work/document_embeddings \
  --base-url 'https://YOUR_EMBEDDING_ENDPOINT/v1'
atlasnav build atlas \
  --embeddings work/document_embeddings --output work/atlas

atlasnav build query-embeddings \
  --dataset upstream/questions.jsonl --output work/query_embeddings \
  --base-url 'https://YOUR_EMBEDDING_ENDPOINT/v1'
atlasnav runtime build \
  --corpus work/corpus --fulltext-index work/fulltext.sqlite3 \
  --atlas work/atlas --query-embeddings work/query_embeddings \
  --router-model frozen/router_model.npz \
  --dataset upstream/questions.jsonl \
  --output work/runtime --state work/state
atlasnav runtime audit --output work/runtime
```

To retrain the Router from the corpus instead of using the frozen model:

```bash
atlasnav router build-tasks \
  --corpus work/corpus --atlas work/atlas --output work/router_tasks
export ATLASNAV_SYNTHESIS_API_KEY='...'
atlasnav router synthesize \
  --tasks work/router_tasks --output work/router_questions \
  --base-url 'https://YOUR_CHAT_ENDPOINT/v1' --model 'YOUR_MODEL'
atlasnav build query-embeddings \
  --dataset work/router_questions/questions.jsonl \
  --output work/router_query_embeddings \
  --base-url 'https://YOUR_EMBEDDING_ENDPOINT/v1'
atlasnav router build-arrays \
  --questions work/router_questions \
  --query-embeddings work/router_query_embeddings \
  --atlas work/atlas --fulltext-index work/fulltext.sqlite3 \
  --output work/router_arrays
atlasnav router train --arrays work/router_arrays --output work/router
```

The original final 7,163 Router examples cannot be reconstructed byte-for-byte
from the current server. The exact frozen Router is included for deployment
and exact result reproduction; the public corpus-only pipeline creates a new,
leakage-controlled training set with the same declared cardinalities and
objective. See [Router training](docs/ROUTER_TRAINING.md).

## Run, judge, and evaluate

Provider-neutral profiles read host and key values from environment variables.
Edit a copied profile when using a different compatible model or price.

```bash
export ATLASNAV_AGENT_BASE_URL='https://YOUR_AGENT_ENDPOINT/v1'
export ATLASNAV_AGENT_API_KEY='...'
atlasnav run \
  --runtime work/runtime \
  --profile configs/paper/deepseek-v4-flash.toml \
  --output runs/atlasnav

export ATLASNAV_AGENT_BASE_URL='https://YOUR_JUDGE_ENDPOINT/v1'
export ATLASNAV_AGENT_API_KEY='...'
atlasnav judge \
  --run-dir runs/atlasnav --dataset upstream/questions.jsonl \
  --profile configs/paper/deepseek-v4-flash.toml \
  --output runs/judge

atlasnav evaluate \
  --run-dir runs/atlasnav --judge-dir runs/judge \
  --qrels frozen/evidence_span_qrels.jsonl --corpus work/corpus \
  --turn-checkpoints 15 30 60 120 300 \
  --cost-checkpoints 0.15 0.40 1.20 2.75 \
  --output runs/evaluation
atlasnav report --evaluation runs/evaluation --output runs/report.md
```

The runner writes delta-only events, the complete conversation, usage, cost,
termination reason, and final answer under one durable directory per query.
Completed queries are skipped on resume. Cost is computed from the selected
model profile and currencies are never silently mixed.

Frozen Safe Release thresholds are profile-specific rather than global:

| Backbone profile | Currency | Threshold |
|---|---:|---:|
| DeepSeek-V4-Flash | CNY | 2.75 |
| MiMo-V2.5 | CNY | 1.40 |
| ChatGPT-5.6-Luna | USD | 0.65 |
| Qwen-3.7-Flash | CNY | 1.95 |

At the threshold the harness sends a finalization instruction and disables
further tools; it does not fabricate an answer. Passive historical checkpoints
instead count unfinished trajectories as wrong and perform no model call.

## Declarative operation

`configs/examples/pipeline.toml` contains a provider-neutral runbook:

```bash
atlasnav pipeline build --config configs/examples/pipeline.toml
atlasnav pipeline run --config configs/examples/pipeline.toml
```

Every stage is finalized by a manifest and checksum before the next begins.
Existing finalized stages are reused; partial provider work remains resumable.
`configs/examples/pipeline_retrain_router.toml` shows the longer one-command
build that also reconstructs corpus-only Router supervision and retrains it.

For an auxiliary bundle, feed `documents.parquet` to `corpus prepare` and
`questions.jsonl` to query embedding/runtime construction. Answers, qrels, and
decompositions remain in the physically separate `scoring.jsonl`; no build
stage reads that file. `atlasnav benchmarks audit` checks this contract before
paid construction begins.

## Documentation

- [Architecture and data flow](docs/ARCHITECTURE.md)
- [Router data and objective](docs/ROUTER_TRAINING.md)
- [Evidence Blindness](docs/EVIDENCE_BLINDNESS.md)
- [Frozen artifact contract](docs/ARTIFACT_CONTRACT.md)
- [Code–artifact release matrix](docs/RELEASE_MATRIX.md)
- [Reproduction guide](docs/REPRODUCTION.md)
- [Benchmarks and data responsibility](docs/BENCHMARKS.md)

Benchmark corpora and questions are not redistributed by this code repository.
Obtain them from their upstream sources and follow their licenses. See
`NOTICE.md` for attribution and scope.
