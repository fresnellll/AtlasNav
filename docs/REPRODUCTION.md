# Reproduction guide

## A. Exact zero-API replay

```bash
pip install -e '.[all]'
atlasnav artifacts download --repo-id fresnellll/atlasnav-artifacts --output artifacts/release
atlasnav artifacts verify --artifact-root artifacts/release --deep
atlasnav reproduce --suite paper --artifact-root artifacts/release --output reproduced
```

This replay path verifies checksums, streams seven full BrowseComp-Plus
archives, joins frozen run/judge records, recomputes accuracy, recorded
query-time agent inference
cost, paper consistency, paired outcome cells, passive checkpoint endpoints,
frozen checkpoint-manifest completeness, endpoint and checkpoint Evidence
Blindness, and released auxiliary paper cells. The seventh archive is the
830-query ChatGPT Gold-document Reference; replay independently recovers its
`801/830 = 96.51%` empirical-reference result. It fails rather than silently
continuing on a hash,
cardinality, query ID, currency, checkpoint boundary, or declared endpoint
mismatch. Active Safe Release at a historical boundary requires recorded model
branches; passive replay is explicitly labeled and never presented as active
checkpoint accuracy.

Costs are component-aware. Agent and judge ledgers retain their own currencies,
and the replay refuses to add unlike currencies. For every backbone, the paper
cost metric is reconstructed from the agent ledger only. Judge cost remains
available as a separate diagnostic ledger and is not included in the reported
query-time agent inference cost.

## B. Frozen-method online rerun

Obtain upstream corpus/questions, normalize the corpus, and build its full-text
index. All reported benchmarks have final frozen Atlases in the companion
release; paths are listed in `docs/RELEASE_MATRIX.md`. Auxiliary benchmarks
also include the full four-view query vectors. BrowseComp-Plus instead includes
the exact post-Router weighted-RRF score matrix because its original full-830
query-vector bundle is no longer available.

The BrowseComp-Plus runtime can be rebuilt without an embedding API call:

```bash
atlasnav runtime build \
  --corpus work/bcplus_corpus \
  --fulltext-index work/bcplus_fulltext.sqlite3 \
  --atlas artifacts/release/browsecomp_plus/atlas \
  --frozen-ranking artifacts/release/browsecomp_plus/frozen_ranking/full830 \
  --dataset upstream/bcplus_questions.jsonl \
  --output work/bcplus_runtime --state work/bcplus_state
```

This reconstructs exact initial navigation routes, not the missing raw query
vectors. Build runtime workspaces, then run, judge, evaluate, and report.
Provider agent runs remain stochastic: they test method reproducibility, not
identical answer replay.

Each frozen Atlas is the complete post-construction object: aligned catalog,
four PCA transforms and reduced document-vector matrices, four sparse graphs,
hierarchy memberships, cards, labels, anchors, and bridges. The release also
contains the full four-view query vectors for auxiliary tasks. It does not
contain the upstream corpus itself or the larger pre-PCA document-vector cache.

## C. Full method reconstruction

Run corpus/FTS, document embeddings, Atlas construction, Router support tasks,
question synthesis and verification, Router-question embeddings, training
arrays, CPU Router fitting, target runtime, agent, judge, and evaluation.
Individual commands provide fine control; `atlasnav pipeline build` and
`atlasnav pipeline run` provide declarative operation.

Every stage finalizes a checksummed manifest. Provider stages retain durable
caches, so interruption resubmits only missing work. Because the original final
7,163 generated rows are missing, full reconstruction produces a new valid
Router rather than the byte-identical paper Router. Use the frozen model for
exact result replay.

## Clean-room acceptance

A release candidate is accepted only when installation succeeds in a new
environment, all tests pass, `doctor` finds no release-hygiene violation,
all CLI help commands exit successfully, artifact verification has zero
missing/size/hash mismatches, zero-API replay matches declared endpoints, and
archive scans find no credentials, author paths, or account-provider names.
The current acceptance record checks 7 endpoint rows, 54 endpoint
Evidence-Blindness cells, 54 Locate-All checkpoint cells, and all released
non-Enterprise auxiliary paper cells after a clean-wheel installation.
Enterprise paper-result replay is available from the final frozen package;
the document-selection component can additionally be recomputed with
`atlasnav enterprise evaluate` and an OpenAI-compatible API. It additionally validates ten
frozen auxiliary construction packages and has rebuilt and audited a complete
310-query FanOutQA runtime from the released Atlas/query vectors plus the
upstream corpus.

The released code and the checksummed companion dataset form one versioned
release: the code is this repository, and the dataset is published as
`fresnellll/atlasnav-artifacts` on the Hugging Face Hub.
