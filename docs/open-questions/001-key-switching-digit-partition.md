# 001 — How should the modulus chain be partitioned into key-switching digits?

Status: **open**. Default in the meantime: partition by prime count.
Nothing is blocked on the answer.

## The question

Hybrid key switching splits the main modulus chain `q_0 … q_{L-1}` into
`dnum` digits, and raises each digit to `P · Q` before coming back down. `P`
must exceed the widest digit, so the partition decides how large `P` has to
be, and `P` is not free — every special prime costs work in ModUp/ModDown and
consumes budget that could have been a level.

Given `L` primes and a chosen `dnum`, **which primes go into which digit?**

We are not asking how to choose `dnum`. That is a parameter the caller sets
explicitly, on purpose — see the `dnum` decision in
[`../superpowers/plans/2026-07-28-builtin-ckks-backend.md`](../superpowers/plans/2026-07-28-builtin-ckks-backend.md).
This is only about the partition that follows from it.

## Why it is not obvious

The systems-obvious answer is to split by **prime count**: `ceil(L/dnum)`
primes per digit. It is one line, it indexes trivially, and every digit is
"the same size".

But digits are not the same size. What `P` has to cover is a digit's
**product**, and real CKKS chains do not use uniform primes — a typical
64-bit CPU chain has a wide bottom prime and narrower middle primes. Counting
primes and measuring bits are different things the moment the chain is not
uniform.

This project's own code already disagreed with itself about which one a digit
"size" means: `Params::create` checks that `P` is sufficient by summing
`log2` over a digit's primes, while `Params::digitRange` partitions by
count. The validator measures bits; the partitioner counts primes.

## What we measured

Chain `[2^59, 2^39 × 7]` — a standard shape, wide bottom prime and narrower
middle primes, `L = 8`. Partitions restricted to **contiguous** ranges, since
that is what the current API can express. "Bit-balanced" is the contiguous
partition minimising the widest digit, found by exhaustive search.

| `dnum` | widest digit, by count | widest digit, by bits | `P` wasted |
|---|---|---|---|
| 2 | 176.0 | 176.0 | — |
| 3 | 137.0 | 117.0 | **20 bits** |
| 4 | 98.0 | 98.0 | — |
| 5 | 98.0 | 78.0 | **20 bits** |
| 6 | 98.0 | 78.0 | **20 bits** |

At `dnum = 3` the count partition is `[3,3,2]` primes → `[137, 117, 78]`
bits; the bit partition is `[2,3,3]` primes → `[98, 117, 117]` bits. Same
`dnum`, same chain, contiguous in both cases — 20 bits less `P` required,
roughly half a special prime.

On a **uniform** chain the two coincide exactly. We also checked all
`(L ≤ 12, dnum)` pairs on a uniform chain: the widest digit is identical for
every one of the 78, so nothing here argues against count-balancing when the
primes really are the same size.

## What we did NOT measure

This is the part we would like help with. Everything above is a statement
about `P` sizing. None of it is a statement about noise, and noise is the
thing that actually constrains a parameter set.

- **Noise growth as a function of the partition.** Our understanding is that
  hybrid key-switching error grows with `sqrt(dnum)` and with the widest
  digit, so minimising the widest digit should help noise as well as `P`.
  We have not verified this, and we would not notice if the real bound
  depended on, say, the *sum* over digits rather than the max.
- **Whether the widest digit is even the right objective.** Minimising the
  max is what `P ≥ widest digit` suggests. If the actual cost is dominated by
  something else, the whole table above is measuring the wrong quantity.
- **Whether digits must be contiguous.** We assumed yes because it is what
  the API expresses, not because we found an argument for it. If a
  non-contiguous partition is legal, the search space is larger and the
  bit-balanced result improves further.
- **Whether unequal digits are ever deliberately wanted.** We treated
  "balanced" as obviously good. A reason to want one digit deliberately
  narrow — noise, the level it sits at, the bottom prime being special —
  would invert the conclusion.
- **Whether a `dnum` that does not divide `L` is a real use case at all.**
  If practitioners always pick `dnum | L`, this question is academic and the
  simplest partition wins by default.

## Current default, and why

**Partition by prime count**, which is what the code does today.

It is the incumbent rather than a choice: it was written before the question
was noticed. It is correct — the partition covers every prime exactly once —
and it is safe, because `Params::create` independently verifies that `P`
exceeds the widest digit *in bits* and rejects the parameter set otherwise.
A count-partitioned parameter set is therefore never silently under-sized;
at worst it is rejected for a `P` that a better partition would have accepted.

One known wart of the count partition, recorded so it is not mistaken for
intent: it can leave a trailing digit empty (`L = 4, dnum = 3` gives sizes
`[2, 2, 0]`), and it makes distinct `dnum` values behave identically
(`L = 12` makes `dnum` 6 through 11 the same partition). Balancing — by
either measure — removes both.

## What would settle this

Any one of these closes it:

1. A reference or derivation for the hybrid key-switching noise bound in
   terms of the digit partition, showing what the objective should be.
2. A counterexample where minimising the widest digit is worse.
3. Practitioner experience that `dnum` is always chosen to divide `L`, which
   makes the whole question moot.
4. Confirmation that the objective is the widest digit and that contiguity is
   required — in which case we switch the default to contiguous bit-balanced
   and this closes as answered.

## Reproducing the table

No dependency on this codebase; the numbers come from the chain shape alone.

```python
import math, itertools
def bits(ps): return sum(math.log2(p) for p in ps)
def by_count(L, d):
    b, r = divmod(L, d); return [b + 1] * r + [b] * (d - r)
def contiguous_minimax(ps, d):
    L, best = len(ps), None
    for cuts in itertools.combinations(range(1, L), d - 1):
        idx = (0,) + cuts + (L,)
        parts = [ps[idx[i]:idx[i + 1]] for i in range(d)]
        if any(not p for p in parts): continue
        m = max(bits(p) for p in parts)
        if best is None or m < best: best = m
    return best

chain = [1 << 59] + [1 << 39] * 7
for d in range(2, 7):
    sizes, i, parts = by_count(len(chain), d), 0, []
    for s in sizes:
        parts.append(chain[i:i + s]); i += s
    print(d, round(max(bits(p) for p in parts), 1), round(contiguous_minimax(chain, d), 1))
```
