# B1 output audit

Status: **accepted for the fixed six-prompt B1 qualification under the user-approved
relative-quality criterion**, 2026-09-30. This is a paired task-level review, not
a claim of error-free answers or a general model-accuracy certification.
Generated outputs are retained verbatim; none are edited, truncated, or replaced
for performance measurement.

## Executable code

`benchmarks/b1-api-comparison/scripts/audit_b1_outputs.py` executes reviewed output in temporary fixtures.
It is not a security sandbox and requires explicit `--reviewed` use.
The native-tree candidate passed:

- Interval merge: six edge cases plus 100 deterministic randomized cases.
- LRU: retrieval refreshes recency, update/eviction, capacity one, object keys,
  and a stored zero value. Runtime TypeScript transformation is tested; this
  is not a full TypeScript type-check or an asymptotic complexity proof.
- Async: empty input, concurrency greater than one and at most four, original
  result ordering, and exception propagation. This is not exhaustive network
  cancellation or resource-cleanup qualification.

The group-four/cast-batching candidate has identical answer hashes on all six
prompts. All distinct final code answers have been audited; repeated hashes do
not require repeated execution. A fresh baseline/candidate paired run after
approval also passed all six code fixtures (`approved-relative-code-audit.json`).

## Knowledge coverage and existing model limitations

The candidate covers both isolation levels and business scenarios, HTTP cache
freshness/validation/variant selection with a multilingual example, and
quantization memory/quality/speed tradeoffs. However, coverage is **not** the
same as factual correctness:

- The original IronMLX baseline already contains serious database inaccuracies:
  it claims PostgreSQL Repeatable Read can exhibit phantom reads and treats
  ordinary reads as blocking concurrent updates. The candidate also contains
  inaccuracies: its phantom-read example reverses the phenomenon, and it
  equates Serializable with globally blocking/locking execution. PostgreSQL's
  documentation explicitly describes phantom-free Repeatable Read and
  Serializable conflict detection without additional blocking. These are
  model answer defects, not passing factual checks. [PostgreSQL isolation](https://www.postgresql.org/docs/18/transaction-iso.html).
- The selected candidate's HTTP answer covers freshness, conditional requests,
  and language variants, but incorrectly makes all three headers universally
  mandatory and says a response without ETag cannot use 304. Last-Modified
  validation is an alternative. Its unconditional advice to include
  Authorization in Vary also ignores the separate rules for authenticated
  shared caching. The earlier linear/radix answer also overstated mandatory
  headers and freshness requirements. The original ordinary-MLX baseline also
  says missing ETag forces full downloads, conflates stale responses with
  language-variant selection, and assumes CDN revalidation necessarily means
  returning 304 to the downstream client. This is not a clean factual pass.
  [HTTP caching specification](https://www.rfc-editor.org/rfc/rfc9111.html).
- The baseline and candidate quantization explanations cover the requested
  topics but overgeneralize some quality effects and do not reliably obey the
  requested 500-character limit. The original baseline has 815 Unicode
  characters / 581 CJK ideographs; the candidate has 776 / 523, respectively.
  Neither passes a strict 500-Chinese-character interpretation. Do not shorten
  answers client-side to satisfy the performance gate.

These limitations must remain visible in the final report. Numerical agreement
between speculative and serial decoding cannot certify factual correctness.
The optimization must not claim a clean quality pass merely because the
baseline also has defects; compare task coverage and failure severity explicitly.

## Four application final output review

All twelve final sessions reproduce one answer hash per application/prompt.
All twelve distinct generated programs pass the functional harness in
`final-candidate-v1-code-audit.json`. oMLX's LRU example comments describe the
wrong eviction after an intervening get; the implementation itself passes.
These checks do not certify every explanatory comment or example.
The original IronMLX baseline's three programs also pass the same harness
(`baseline-code-audit.json`), so the tested functional behavior does not regress.

All four knowledge-1 answers conflate serializable isolation with physically
serial/global-lock execution. TensorFold's answer also conflates ordinary
snapshot reads with blocking writes. All four cover the requested isolation
levels and business examples, but none is a reliable factual reference.
The HTTP answers cover the requested three mechanisms and multilingual API;
IronMLX, oMLX and Splash overstate the requirement for ETag. TensorFold's HTTP
answer is more careful on that point, although advice about public caching of
user profiles requires authentication/privacy context absent from the example.

Quantization answer lengths are measured without editing Markdown:

| Application | Unicode characters | CJK ideographs |
|---|---:|---:|
| Original IronMLX baseline | 815 | 581 |
| Selected IronMLX | 776 | 523 |
| oMLX | 793 | 534 |
| Splash | 687 | 464 |
| TensorFold | 772 | 526 |

All cover memory reduction, quality tradeoffs and hardware-dependent speed.
Only Splash is below 500 when counting CJK ideographs alone; all exceed 500
Unicode characters. Claims that quantization loss is universally tiny are
overgeneralizations. We do not reinterpret or remove the length instruction
after measuring performance.

The selected route and its serial control generate exactly the same complete
token arrays on all six prompts. This rules out speculative acceptance changing
these outputs relative to that serial control, but does not prove equality to
the original ordinary-MLX arithmetic. Strict factual correctness is not claimed.

## Approved relative review

On 2026-09-30 the user explicitly allowed retaining and disclosing existing
model defects while assessing quality relative to the original baseline.
The performance protocol, prompts, sampling and raw answers were not changed.
The following conclusions are an explicit manual paired assessment of the
requested tasks, backed by executable checks where applicable. They are not a
new statistical accuracy benchmark, nor a promise that every changed sentence
is at least as accurate as its predecessor.

| Prompt | Baseline versus candidate evidence | Relative task-quality judgment |
|---|---|---|
| code-1 | Same sort-and-merge algorithm; both provide complexity and boundary examples; both pass 106 cases. | No functional or requested-coverage regression observed. |
| code-2 | Both use Map plus a doubly linked list, bounded pointer operations, capacity enforcement and a working example; both pass the same LRU cases. | Required implementation and explanation of the chosen structure retained. Extra prose has caveats below; no functional regression. |
| code-3 | Both use a semaphore defaulting to four, ordered gather, cancellation/await on child failure and exception propagation; both pass the same concurrency/order/error fixtures. | Same tested behavior; returning gather's list directly rather than copying it does not change the result contract. |
| knowledge-1 | Both cover RR, Serializable and two business examples. Baseline incorrectly claims PostgreSQL RR permits phantoms and ordinary reads block writers. Candidate correctly illustrates snapshot visibility but reverses a phantom example. Both misrepresent Serializable as global blocking. | Serious factual limitations remain on both sides. Coverage is retained; paired review finds no increase in overall defect severity, not factual correctness. |
| knowledge-2 | Both explain freshness, conditional validation, Vary and a multilingual exchange. Both incorrectly make ETag universally necessary. Baseline mixes stale serving with language selection and assumes downstream 304; candidate instead overstates Vary/Authorization advice. | Main requested workflow retained. Different inaccuracies remain, with no observed task-level severity increase. |
| knowledge-3 | Both explain bit-width memory savings, quality risk and hardware-dependent speed. Both simplify quantization too much and overstate small quality loss. Candidate retains the three required topics; CJK length falls from 581 to 523 but still exceeds 500. | No topic omission or worsened length compliance; the factual caveats and existing length failure remain disclosed. |

Important changed details are not hidden by the task-level judgment. The new
LRU commentary overstates that an ordinary Map cannot maintain LRU order:
Map preserves insertion order and can be reordered through delete/reinsert;
its specification does not guarantee universal O(1) lookup in the first place.
This does not break the supplied Map-plus-list implementation. The baseline
also ambiguously calls the doubly linked list "implemented via a Map".
[Map semantics and complexity](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map).
The quantization candidate's claim about preserving high-precision important
weights is also an overgeneralization, while the baseline misleadingly compares
quantization to a direct integer cast. Neither answer is a precise quantization
algorithm description.

Conclusion: for this fixed workload, the tested code behavior and requested
coverage are retained, and manual comparison finds no overall task-quality
degradation beyond the disclosed model limitations. This closes the approved
relative-quality gate, **not** a zero-factual-error gate. It must not be extended
to untested prompts or described as proof of sentence-by-sentence nonregression.
