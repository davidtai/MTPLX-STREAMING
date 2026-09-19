# Consensus suffix after native Q4 proposals

The retained run needs 198 expensive target calls and 565.54 GB of expert
reads. Its native draft is inexpensive, and pure text lookup is already part
of the measured winner. Improving verified tokens per call can reduce both
target scheduling cost and expert traffic without changing Q4 weights.

Keep all five native draft decisions and the existing full-proposal lookup.
When that lookup supplies no suffix and minimum native confidence is at least
0.9, search already-observed text for the longest matching suffix of length
six down to three. At least two occurrences must unanimously support each
added token. Propose at most two additional tokens. The target verifier and
committed KV remain authoritative. Proposal selection receives no teacher
future, accepted-prefix count, or future target hidden state.

This differs from the earlier conditioned-tail head: it adds no model pass.
It differs from the existing lookup: repeated shorter suffixes may agree even
when the entire native proposal plus two prior context tokens is absent.
Keep the fixed M8 target bound. Wider neural drafts and longer raw copied
suffixes have already lost their earlier screens and are not repeated.

Risks: generic suffixes can agree on the wrong continuation; changed cycle
boundaries can erase independent-boundary gains; Python index memory can cost
expert capacity; target batch arithmetic can alter the replayed trajectory.
Native confidence and repeated witnesses limit proposal risk, a head-only
trajectory screen measures the boundary effect, and any full candidate must
price a bounded index and pass the existing exact output gate.

The CPU screen selects minimum suffix three using only first-half gain and
row efficiency. It reproduces all 198 control boundaries and adds 16 matched
tokens for 18 extra target rows across independent opportunities. The second
half contributes 9/9. This is not a throughput result or a new cycle count.
Next: one unchanged-control/candidate head replay with actual new boundaries,
sharing native weights and the existing 49 GiB complete incremental bound.

The head replay subsequently completes: control 198 calls / 1,242 rows, candidate
195 calls / 1,240 rows. Four candidate calls use M7; the existing row-dependent
M≤8 target lane and retained M8 allocation envelope cover that geometry.
The new index receives 32 MiB of additional host reserve. Proceed to one full
exact 16K/1K candidate with fresh admission and the original output gate;
head timing alone is not a speedup claim. See the dated consensus receipt.

The full candidate completes at 13.3997126089 TPS with all 1,024 native IDs exact,
195 target calls, and 109 slots at an 11.05 GB background. The prior 110-slot best
remains 13.4141517619 TPS. Different capacity prevents an isolated comparison;
the candidate is not promoted and receives no new regression tests.
