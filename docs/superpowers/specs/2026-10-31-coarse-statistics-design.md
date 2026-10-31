# Coarse contributed statistics kernels

User-authorized implementation: contribute larger kernels for the existing
stats/matvec-fused apps, share intermediates with ordinary commands, and measure
real speed without modifying the runtime core. Continue in the primary checkout
with accumulated work; no Git writes. Runtime.h/cpp and NativeRuntime.h/cpp are
frozen and hashed before work.

## Chosen implementation

The implementer's choice, supported by Astra recon, is factorized statistics.
For the existing width-n repeated slot layout, s=reduce(x), the variance proxy
n*reduce(x*x)-s*s equals reduce((n*x-s)^2/n). Reusing the ordinary command result s
saves one ciphertext tensor product and one relinearization/rescale per dataset.
This retains the exact application formula, output count and modulus depth.
Division by public n uses actual output scale n*(input_scale^2/dropped_prime).
The changed scale is explicit, not a relabeling to force incompatible joins.
Statistics kernels require input level>=2; the tested envelope is the default
N8192 Q50/40/40 profile, width64, |x|<=8 and supplied ordinary sum. This leaves
headroom for the larger encoded centered square. Rounding differs and must
satisfy the unchanged independent oracle tolerance.
No plaintext shortcut or client/secret access enters compute kernels.

Alternatives: reduce(x*(n*x-s)) also saves a tensor/switch, but a constant -8
regression exposed worse first-order cancellation error (1.66e-4 versus the
original 1.92e-5); it was rejected without relaxing the 1e-4 absolute threshold.
Grouping the old expression only removes boundary overhead;
delayed relinearization through rotations needs additional keys and can increase
work. General Shoup precomputation for evaluation keys is useful future work,
but not necessary to establish contribution/state sharing here.

Four additive native exports implement centered energy (n*x-s)^2/n, its reduced
result, packed dot product reduce(a*b), and partial rotate-add reduction. They
use existing private region arithmetic, transactional staging and context scratch.
They do not change Eval arithmetic, the opcode table, runtime dispatch or the
compiler. Input validation and all key checks precede public output publication.
These functions compute their stated expressions for the supplied operands;
metadata does not prove a caller-supplied s is the corresponding sum.

A separate contribution target registers named recipes using installed public
Runtime/NativeRuntime APIs. It obtains inputs through buffer(), charges a native
staging buffer through makeValue(), then publishes with importCopy(). The copy
is included in timing/accounting; no private native Value payload is forged.
Callbacks retain services/adapter/library ownership and use no client pointer.

## Evidence

- Native tests: full-slot encrypted results, ordinary-command compatibility,
  admitted statistics levels and all valid dot/reduce levels, width bounds, input/output aliasing, expired/missing keys,
  wrong context, exhausted scratch, and transactional failure.
- Contribution tests: runtime/core unchanged, large kernel consumes ordinary s,
  energy output consumed by ordinary rotation/add then contributed reduction,
  shared values retained/released, constrained budgets and no retry on failure.
- Existing stats oracle unchanged. Include constant vectors and cancellation
  cases, which test absolute accuracy near zero variance.
- Existing matvec-fused can consume the same contributed packed-dot kernel with
  its original shared vector and encrypted rows, if the measured gain justifies it.
- Time complete apps, including sum construction, staging, publication and
  dispatch. Report setup separately, alternate mode order, use identical
  ciphertext/key inputs, include all outputs/slots and all timing samples.
- Run Debug/sanitizers, Release/test-off and installed contribution consumer.
  Hash checks establish the four frozen runtime files did not change.
