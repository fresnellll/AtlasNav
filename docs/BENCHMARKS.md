# Benchmarks and data responsibility

## BrowseComp-Plus

The primary benchmark has 830 questions over 100,195 canonical files. The code
expects question JSONL with `query_id`, `query`, and `answer`. It does not
redistribute the upstream corpus or questions.

The companion release provides the frozen Atlas, Router, full trajectories for
two recoverable backbones, analyses, endpoints, and the fragment-level Qrel.
The Qrel has 889 required slots and 1,753 acceptable spans across 830 questions.

## PhantomWiki

The scaling study fixes 200 questions and evidence while expanding nested
corpora to about 10K, 50K, 100K, and 1M files. Result records and preprocessing
code are retained; obtain or regenerate the corpus under upstream terms.

The public adapter reconstructs the experiment without evaluation leakage:

```bash
atlasnav benchmarks prepare \
  --adapter phantomwiki \
  --config configs/benchmarks/phantomwiki-100k.toml \
  --output prepared/phantomwiki-100k
```

The four scale configs `configs/benchmarks/phantomwiki-{10k,50k,100k,1m}.toml`
differ only in the number of nested worlds; the 1M config additionally consumes
generated shards.

For exact frozen-method reuse, the companion release provides final Atlases for
10K, 50K, 100K, and 1M and one shared 200-query four-view bundle under
`phantomwiki/frozen/`. These are runtime-ready after acquiring the upstream
corpus; rebuilding the Atlas from API-generated document vectors is optional.

For 1M, first generate independent official distractor worlds. Generation is
resumable at one Parquet shard per world and makes no paid API call:

```bash
atlasnav benchmarks generate-phantomwiki \
  --official-repository upstream/phantom-wiki \
  --output prepared/phantomwiki-worlds \
  --world-start 20 --world-stop 200
atlasnav benchmarks prepare \
  --adapter phantomwiki \
  --config configs/benchmarks/phantomwiki-1m.toml \
  --output prepared/phantomwiki-1m
```

The first 100K rows are the same nested prefix at every applicable treatment;
worlds 20--199 use independent generator seeds and disjoint literal namespaces.

## EnterpriseRAG-Bench

The experiment covers 500 questions over 511,958 unique document identifiers
from heterogeneous enterprise sources. The released package contains the
frozen final answers, the official answer judgements, and the official
document metrics used by the paper. Its Document Recall and Invalid Extra
values are obtained with the official selection semantics over the documents
exposed by the recorded run and final answer, without exposing gold labels to
the selector.

The adapter retains all 511,962 source rows. Four repeated raw document IDs
are not silently deduplicated: later occurrences receive a stable `__dupN`
suffix, and evaluator-side qrels consume them in occurrence order.

```bash
atlasnav benchmarks prepare --adapter enterpriserag \
  --config configs/benchmarks/enterpriserag.toml \
  --output prepared/enterpriserag
atlasnav benchmarks export --bundle prepared/enterpriserag \
  --run-dir runs/enterpriserag --output submissions/enterpriserag.jsonl
```

`benchmarks export` reads only query IDs, final answers, and successful
canonical `open` observations. It does not consult Gold when selecting document
IDs. The resulting JSONL can be passed to the benchmark's official evaluator.

The final 511,962-row Atlas and the 500-query four-view bundle are released
under `enterpriserag/frozen/`. Rejected graph builds and provider caches are
intentionally excluded. The package also contains `questions.jsonl`,
`answers.jsonl`, `results.json`, and `retrieval_audit.jsonl`.

## Additional evaluations

Compact records cover FanOutQA, TREC-COVID, 2Wiki-Global-400, and fixed
50-query SciFact/ArguAna subsets. They are auxiliary transfer or boundary
analyses. Full upstream datasets are not included.

Their final Atlas/query-vector pairs are nevertheless included under each
benchmark's `frozen/` directory. SciFact and ArguAna use their full-corpus and
full-query frozen construction assets even though the paper-facing compact
records emphasize the predeclared 50-query comparison.

All are executable adapters rather than frozen result-only labels:

```bash
atlasnav benchmarks prepare --adapter fanoutqa \
  --config configs/benchmarks/fanoutqa.toml --output prepared/fanoutqa
atlasnav benchmarks prepare --adapter 2wiki \
  --config configs/benchmarks/2wiki-global400.toml --output prepared/2wiki
atlasnav benchmarks prepare --adapter trec-covid \
  --config configs/benchmarks/trec-covid.toml --output prepared/trec-covid
atlasnav benchmarks prepare --adapter scifact \
  --config configs/benchmarks/scifact.toml --output prepared/scifact
```

FanOutQA uses the exact 310-question evidence-union closed corpus (1,594 dated
pages); this is not the open-book leaderboard setting. Its deterministic
loose/strict scorer follows the public normalization and requires
`en_core_web_sm`. BEIR-format adapters preserve graded qrels and report linear-
gain `NDCG@10`, `Recall@10`, and `Recall@100`:

```bash
atlasnav benchmarks score --bundle prepared/trec-covid \
  --run-dir runs/trec-covid --output evaluations/trec-covid
```

## Local schemas

Corpus row:

```json
{"docid":"stable-id","text":"canonical body","url":"optional source URL"}
```

Question row:

```json
{"query_id":"stable-query-id","query":"question text","answer":"reference answer"}
```

Evidence Qrel rows contain required `answer_slots` and acceptable `spans`;
each span contains `docid`, `quote`, and `slot_ids`.

## Shared auxiliary bundle contract

Every adapter writes:

- `documents.parquet`: only `docid`, canonical `text`, and optional `url`;
- `questions.jsonl`: only `query_id` and `query`, the complete Agent input;
- `scoring.jsonl`: answers, qrels, decompositions, facts, and task metadata;
- `manifest.json`: source hashes, selection seed, row counts, and the explicit
  statement that construction never reads `scoring.jsonl`.

Audit before paid work:

```bash
atlasnav benchmarks audit --bundle prepared/trec-covid
atlasnav corpus prepare --input prepared/trec-covid/documents.parquet \
  --output work/trec-covid/corpus
```

After that first normalization command, the ordinary four-view embedding,
Atlas, Router/runtime, run, judge, evaluate, and report commands are identical
across benchmarks. This is the intended zero-to-result path, not a collection
of benchmark-specific launchers.

Users are responsible for acquiring benchmark materials and complying with
their licenses.
