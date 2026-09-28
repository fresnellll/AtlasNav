# Router data construction and training

## Reproducibility status

The exact frozen Router model used by the paper is recoverable and distributed
with the artifact release. The exact final 7,163 question rows and their
derived arrays are not recoverable from this server. A larger set of 30,000
frozen corpus-grounded candidate tasks exists, but it is not the final training
set and is labeled accordingly.

The public code is a complete leakage-controlled retraining implementation. A
new run targets the paper cardinalities and objective but is not expected to
reproduce the frozen model byte-for-byte because question generation is
stochastic. Exact paper results must use the frozen model.

## 1. Outcome-blind support tasks

`atlasnav router build-tasks` reads only the canonical corpus and Atlas. Parent
regions are hash-split 70/15/15 before selection, so a parent never appears in
more than one split. Single tasks use one file; pair tasks follow strong
cross-leaf graph edges; triple tasks form two-edge chains around a common
middle file. No benchmark question, answer, Qrel, trajectory, judgment, or
correctness field is read.

## 2. Generation and independent verification

`atlasnav router synthesize` renders support packets as natural single-answer
questions. An independent request verifies grounding, naturalness, necessity
of every positive, coherent chaining, answer uniqueness, and independence from
a deterministic negative decoy. Source/retrieval wording is rejected.

| Kind | Count |
|---|---:|
| Single | 1,913 |
| Pair | 4,300 |
| Triple | 950 |
| Total | 7,163 |

The parent-disjoint split is 4,972 train, 1,167 validation, and 1,024 test.
SQLite caching makes provider calls resumable and records token usage.

## 3. Candidate and feature arrays

`atlasnav router build-arrays` constructs 257 candidates per question where
the corpus is large enough: all positives, same-leaf negatives, same-parent
other-leaf negatives, graph neighbors, BM25 hits, and deterministic corpus
backfill.

$$
x_q=[z_q^T;z_q^I;z_q^E;z_q^R;m_q^T;m_q^I;m_q^E;m_q^R;h_q]\in\mathbb{R}^{816}.
$$

Each $z_q^v\in\mathbb{R}^{192}$; each $m_q^v\in\mathbb{R}^{10}$ records
calibrated similarity spread, top-score sharpness, region concentration, graph
coherence, lexical agreement, and rank-biased overlap; $h_q\in\mathbb{R}^{8}$
contains query-form statistics. Semantic ranks are estimated against a fixed
sample of up to 8,192 corpus files. BM25 ranks are exact within the top 256.

## 4. Three-head linear Router

After train-split standardization:

$$
p_q=\operatorname{softmax}(W_f x_q+b_f),\qquad
\eta_q^{*}=0.2+0.6\,\sigma(w_\eta^\top x_q+b_\eta),
$$

$$
c_q=\sigma(w_c^\top x_q+b_c).
$$

The floor and confidence shrinkage are:

$$
\tilde p_{qv}=0.05+0.8p_{qv},\qquad
p'_{qv}=(1-c_q)/4+c_q\tilde p_{qv},
$$

$$
\eta_q=0.5+c_q(\eta_q^{*}-0.5),\qquad
w_q=[2\eta_q p'_q;\ 2(1-\eta_q)].
$$

The model has $4\times816+4+816+1+816+1=4{,}902$ parameters.

## 5. Multi-positive objective

For $s_{qi}=\sum_v w_{qv}\operatorname{RRF}_{qiv}$ and positives $P_q$:

$$
\mathcal{L}_{\mathrm{all}}(q)=\frac{1}{|P_q|}
\sum_{i\in P_q}\log\!\left(1+
\frac{\sum_{j\notin P_q}\exp(s_{qj}/\tau_r)}{\exp(s_{qi}/\tau_r)}\right),
\quad \tau_r=0.002.
$$

A smooth bottleneck emphasizes the worst positive against the best negative. A
risk penalty is active when learned routing is worse than the uniform safe
baseline. Auxiliary terms supervise facet, semantic-versus-lexical mass, and
confidence targets derived only from corpus ranks:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{all}}
+0.35\mathcal{L}_{\mathrm{bottleneck}}
+0.25[\mathcal{L}_{\mathrm{all}}-\mathcal{L}_{\mathrm{uniform}}]_+
+0.12\mathcal{L}_{f}+0.10\mathcal{L}_{\eta}
+0.15\mathcal{L}_{c}+2\times10^{-4}\|\theta\|_2^2/2.
$$

CPU L-BFGS-B fits 25%, 50%, 75%, and 100% train prefixes. Temperatures and
confidence bias are selected on validation. Final gates audit all-positive
Recall@30, single/rare-task regressions, confidence calibration, mass collapse,
and the semantic floor before finalization.
