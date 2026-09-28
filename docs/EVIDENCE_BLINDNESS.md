# Evidence Blindness evaluation

## 1. Evidence slots and monotone funnel

For question $q$, let $A_q$ be the required answer-evidence slots. At stage
$k\in\{C,S,O,L\}$, $H_q^k\subseteq A_q$ is the set realized in the actual
provider-visible trajectory:

$$
H_q^C\supseteq H_q^S\supseteq H_q^O\supseteq H_q^L.
$$

- **Construction:** acceptable evidence exists in the reachable corpus.
- **Surface:** a supporting document handle enters a visible observation.
- **Open:** the successful canonical body enters model-visible context.
- **Locate:** an acceptable decision-relevant fragment aligned to the slot
  occurs inside a visible canonical `open` observation.

Names, paths, region cards, and previews never count as canonical body.

## 2. Any, Mean, and All blindness

Let $r_q^k=|H_q^k|/|A_q|$. For question set $Q$:

$$
\operatorname{EB}_{\mathrm{Any}}^k=
\frac{1}{|Q|}\sum_{q\in Q}\mathbb{1}[r_q^k=0],
$$

$$
\operatorname{EB}_{\mathrm{Mean}}^k=
1-\frac{1}{|Q|}\sum_{q\in Q}r_q^k,
$$

$$
\operatorname{EB}_{\mathrm{All}}^k=
\frac{1}{|Q|}\sum_{q\in Q}\mathbb{1}[r_q^k<1].
$$

Any means no required slot was realized; Mean is average missing evidence
mass; All means the complete set was not realized. Lower is better.

Macro recall is the mean of per-question $r_q^k$. Micro recall pools realized
and required slot counts across questions. Mean blindness is one minus macro
recall, not one minus micro recall.

## 3. Fragment-level Qrel and Locate

The BrowseComp-Plus Qrel contains 830 questions, 889 answer slots, and 1,753
acceptable spans. Each span identifies a canonical document, an exact
decision-relevant quotation, and supported slots. Text is NFKC-normalized,
case-folded, punctuation-normalized, and whitespace-collapsed. Locate uses
deterministic substring matching inside successful provider-visible canonical
open outputs; it reads no answer, correctness, embedding, or LLM judgment.

The Qrel is a fixed process target, not an accuracy surrogate. A system can
locate all annotated evidence and synthesize incorrectly; it can answer
correctly from partial evidence, an unannotated alternative path, parametric
knowledge, or a guess. Frozen Qrel results must remain separate from any
post-hoc alternative-evidence audit.

## 4. Fragment precision

$$
\operatorname{Precision}_{\mathrm{fragment}}=
\frac{\#\text{ opened support fragments containing an acceptable span}}
{\#\text{ opened fragments from annotated support documents}}.
$$

This measures whether opening the right file produced a focused usable span.
It is undefined when no annotated support-document fragment was opened.

## 5. Checkpoints

Passive checkpoints replay frozen reported trajectories. A query counts correct at
boundary $b$ only if its terminal already existed by $b$; unfinished queries
count wrong, and later evidence is never backfilled. This performs zero model
calls.

Active Safe Release at a historical cost boundary is different: it invokes the
same model once on the prefix with finalization instructions, then judges that
terminal. Its incremental model/judge cost must be stored separately.

The evaluator emits per-query endpoint rows, per-query and aggregate Evidence
Blindness, strict passive checkpoints, a machine-readable summary, and a
stable Markdown report. Accuracy remains the outcome; Evidence Blindness
explains where evidence realization succeeded or failed before that outcome.
