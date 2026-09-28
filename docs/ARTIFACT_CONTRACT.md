# Frozen artifact contract

## Version agreement

The code and companion data releases are coupled by code package version
`atlasnav` 0.2.x and artifact schema `atlasnav_artifact_release_v2` with
`compatible_code=>=0.2.0.dev0,<0.3`.

`atlasnav artifacts verify` checks the code-version range, agreement between
`release_manifest.json` and `MANIFEST.sha256.json`, and every admitted file
path, byte size, and SHA-256 before reproduction.

## Layout

```text
release/
  release_manifest.json
  MANIFEST.sha256.json
  paper_auxiliary_results.json
  browsecomp_plus/
    paper_results.json
    paper_diagnostics.json
    trajectories/*.tar.zst
    analysis/<backbone>/<interface>/
    atlas/
    frozen_ranking/full830/
    router/model/
    router/candidate_tasks/
    qrel/
  enterpriserag/
    README.md
    manifest.json
    questions.jsonl
    answers.jsonl
    results.json
    report.json
    retrieval_audit.jsonl
    document_selection/
    frozen/atlas/
    frozen/queries/
  phantomwiki/frozen/{10k,50k,100k,1m}/{atlas,query_embeddings}/
  fanoutqa/frozen/dev310/{atlas,query_embeddings}/
  trec_covid/frozen/full50/{atlas,query_embeddings}/
  2wiki_global400/frozen/global400/{atlas,query_embeddings}/
  beir_scifact/frozen/full/{atlas,query_embeddings}/
  beir_arguana/frozen/full/{atlas,query_embeddings}/
```

Each BrowseComp-Plus archive contains exactly one reported result per query,
not separate development and remainder shards. Six full packages cover
AtlasNav, DR-DCI, and raw DCI for DeepSeek and ChatGPT; a seventh covers the
ChatGPT Gold-document Reference used for the paper's empirical-reference row.
Each trajectory has model-visible tool observations (as a full conversation or
durable event stream), terminal result, judgment, and stable query identity.

Each trajectory package declares agent and judge currencies independently and
identifies the paper-comparable cost components. The current packages define
that metric uniformly as recorded query-time agent inference cost, and exact
paper replay uses the agent ledger only for every backbone. Judge ledgers
remain available as separate diagnostics and are never folded into the paper
cost metric.

For each of the six DCI-interface packages, the adjacent analysis tree contains one merged
checkpoint manifest with nine boundaries and exactly 830 unique query IDs per
boundary. Exact replay validates these manifests and separately recomputes
passive checkpoint endpoints. It does not claim counterfactual active Safe
Release accuracy unless corresponding branch answers and judgments exist.
The frozen Evidence-Blindness checkpoint summaries are additionally checked
cell-by-cell against the paper lockfile; they are not inferred from passive
answer completion.

The Gold-document Reference is an endpoint intervention rather than a corpus
interface: benchmark-designated Gold document bodies are supplied directly,
the reference answer is withheld, and retrieval/map navigation is bypassed.
Its archive therefore reproduces endpoint accuracy and cost but does not claim
an interface checkpoint or Evidence-Blindness curve.

The frozen Atlas contains reduced view vectors, transforms, sparse graphs,
hierarchy arrays, cards, labels, catalog, and provenance. The frozen Router
contains the deployed model and audit. Candidate support tasks are explicitly
labeled candidates because the exact final 7,163 rows are unavailable.

The BrowseComp-Plus `frozen_ranking/full830` package contains the exact
post-Router weighted-RRF dense score matrix (830 queries by 100,195 documents),
an aligned query-ID-only catalog, retrieval audit, and byte-exact frozen first
viewports with their navigation metadata. The original full-830
query embeddings are unavailable. This frozen later-stage representation is
sufficient to recreate the exact initial routes with the upstream corpus and
full-text index, while making no claim that missing embeddings were recovered.

The auxiliary `frozen/` directories follow the same reusable construction
contract. Document vectors are represented by the exact four 192-dimensional
matrices consumed by runtime construction, together with their fitted PCA
transforms; query vectors retain the full four 2,560-dimensional matrices.
The original full-dimensional document matrices are not runtime inputs and are
omitted as a high-volume, reproducible intermediate. Every admitted frozen
asset is bound into the top-level byte-size/SHA-256 inventory, and deep
verification checks its schema, cardinality, catalog, views, graphs, hierarchy,
and query bundle.

## Scope and provenance

Qwen and MiMo are summary-only and cannot be advertised as raw-trajectory
reproductions from this server.

The EnterpriseRAG package contains the final 500 answers, official answer
judgements, and official document metrics. Document Recall and Invalid Extra
follow the official selection semantics over the documents the recorded run
exposes, using no gold labels. The GitHub package exposes both frozen metric
replay and the OpenAI-compatible LLM selector through
`atlasnav enterprise evaluate`.

Benchmark corpora are not bundled. Qrel quotations and benchmark-derived
content remain subject to the upstream benchmark terms; users are responsible
for complying with them.
