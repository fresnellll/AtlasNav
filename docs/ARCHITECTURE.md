# Architecture and data flow

## 1. System boundary

AtlasNav changes the corpus interface while keeping the downstream
search–read–verify agent loop intact. It has an offline reusable plane and an
online query plane:

```text
Upstream corpus
  -> canonical files + full-text index
  -> four grounded signatures per file
  -> four embedding matrices
  -> four sparse kNN graphs
  -> multiplex parent/leaf hierarchy + anchors + bridges
  -> frozen Corpus Atlas

Corpus-only support tasks
  -> generated and independently verified questions
  -> 816-D features + multi-positive candidates
  -> three-head linear Router

User query
  -> four query embeddings + BM25
  -> query-specific five-channel weights
  -> weighted RRF over the full corpus
  -> parent/leaf viewport
  -> search / expand / route / open / verify / answer
  -> judge + cost + checkpoints + Evidence Blindness
```

The Atlas assigns every file exactly one primary parent–leaf address. It never
removes a file from full-text search or direct canonical opening.

## 2. Canonical corpus

`atlasnav corpus prepare` validates unique document IDs, cleans text, and emits
canonical Parquet, a stable catalog, and a checksummed manifest.
`atlasnav corpus build-index` creates a full-corpus SQLite FTS5 index. File
indices remain aligned across corpus, catalog, embeddings, graphs, Atlas
arrays, Router candidates, and runtime handles.

## 3. Four grounded file views

For each canonical file $d$, the signature builder creates:

$$
Z(d)=\{z_d^{T},z_d^{I},z_d^{E},z_d^{R}\},\qquad
z_d^{v}=\operatorname{norm}\!\left(P_v\,\operatorname{Enc}(\operatorname{Sig}_v(d))\right).
$$

- **Topic** preserves subjects, domains, and independent content areas.
- **Identity** emphasizes entities, aliases, works, organizations, and titles.
- **Episode** emphasizes dates, places, events, stages, and chronology.
- **Relation** emphasizes roles, causes, comparisons, quantities, and
  cross-entity connections.

Signatures contain the file title and source domain plus deterministic,
view-scored canonical excerpts. Topic selects up to eight focused and eight
uniform-backfill passages within 6,144 characters. Other views select up to
twelve focused and three backfill passages within 4,096 characters. Token-set
Jaccard diversity is capped at 0.68. Backfill prevents the heuristic scorer
from becoming an information boundary.

Each view is embedded independently at 2,560 dimensions and projected through
its own PCA transform to 192 normalized dimensions. The vectors are never
concatenated before graph construction.

## 4. Sparse graphs and persistent hierarchy

For every view, cosine/IP HNSW finds $k=48$ neighbors with $M=32$,
`efConstruction=160`, and `efSearch=128`. A locally scaled affinity and rank
discount produce one undirected sparse graph per view. One-way edges remain at
half strength so rare modes are not made unreachable.

Multiplex Leiden integrates Topic with weight 1.0 and Identity with weight
0.75 into coarse parent regions. Within each parent, Episode and Relation
induce conditional leaf regions. Resolution is selected from a frozen grid
using cross-seed stability, size balance, within-region edge gain, and
navigability—not by forcing exactly 100 clusters.

For the frozen BrowseComp-Plus Atlas this yields 100,195 addressed files, 77
parent regions, 443 leaf regions, one stable address for every file,
representative anchors, and cross-leaf Episode/Relation bridges.

## 5. Query-adaptive full-corpus fusion

The Router receives four 192-D query vectors, ten corpus-response statistics
for each view, and eight mechanical query features:

$$
4\times192 + 4\times10 + 8 = 816.
$$

Its three linear heads predict semantic facet preference,
semantic-versus-BM25 mass, and confidence. A 0.05 floor on each view's share of
semantic mass prevents a view from vanishing; low confidence shrinks toward
uniform semantic weights and a 1:1 semantic/BM25 mass. The deployed model has
4,902 parameters.

Each channel ranks the full corpus. Weighted Reciprocal Rank Fusion gives:

$$
S_q(d)=\sum_{v\in\{T,I,E,R,\mathrm{BM25}\}}
\frac{w_v(q)}{60+\operatorname{rank}_v(d\mid q)}.
$$

Priorities are projected to leaves and parents. The initial viewport contains
ten distinct parents and up to three leaf anchors per parent. Query routing
changes priorities, not reachability.

## 6. Online tool interface

The runtime supports overview, region expansion, member listing, cross-region
bridges, full-corpus or region-scoped lexical search, rerouting, lead
inspection, canonical file opening, and reversible lead dismissal.

Every query produces an immutable initial observation, delta-only tool events,
the provider-visible full conversation, a terminal record, token usage, and
cost. A 30-second timeout applies to each local tool invocation; the loop
allows at most 300 turns.

## 7. Budget-triggered finalization

`Safe Release` is the implementation name for the paper's budget-triggered
finalization protocol. It is an online budget boundary configured per model. Once recorded
agent cost reaches the profile threshold, the harness injects a finalization
instruction, disables additional tools, and lets the same model produce one
checked terminal answer. It does not fabricate an answer or retry until
correctness. Passive historical checkpoints instead perform no model call.

## 8. Evaluation outputs

`atlasnav evaluate` joins trajectories and judgments and writes strict
accuracy, terminal validity, cost components, turns, passive checkpoints, and
Construction/Surface/Open/Locate Evidence Blindness when Qrels exist. The
evaluation reads actual provider-visible outputs: a filename or preview can
satisfy Surface but never Open; only a successful canonical `open` observation
can satisfy Open or Locate.
