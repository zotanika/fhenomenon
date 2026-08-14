# Open Questions

Questions this project has hit where the systems answer and the cryptographic
answer might not be the same one, and where we would rather be corrected than
be quietly wrong.

Each file states one question, the evidence we actually gathered, **the
evidence we did not gather**, and the default the code uses in the meantime.
Nothing here blocks work: every question has a stated default, so the answer
can change later without anything waiting on it.

## Why this exists

A systems engineer partitioning a modulus chain reaches for the partition
that is easiest to index. A cryptographer looking at the same code may see a
cost function the systems engineer never measured. Question 001 is exactly
that: the obvious partition is by prime count, the quantity that actually
costs something is bit-width, and on a standard parameter chain those differ
by 20 bits of special prime.

We found that only because writing the question down for someone else forced
it into a form where the gap was visible. That is the first reason these
files exist, and it does not depend on anyone showing up.

The second reason is the repository's standing rule that a claim is either
measured or not made. A question we cannot answer yet should be *recorded* as
unanswered, with the evidence attached, rather than settled by whoever
happened to write the code.

## If you want to answer one

You do not need to know this codebase. Each question is written to be
readable on its own and ends with **What would settle this** — the specific
evidence that would close it. A reference to the literature, a counterexample,
or "you measured the wrong quantity, here is the right one" are all useful
answers. So is "the default is fine, and here is why."

Open an issue referencing the question number, or comment on the issue that
tracks it.

## Index

| | Question | Status | Default in the meantime |
|---|---|---|---|
| [001](001-key-switching-digit-partition.md) | How should the modulus chain be partitioned into key-switching digits? | open | Partition by prime count |

## What is deliberately not here

Decisions that are ours to make and have been made — scheme choice, the
layering, key switching being parameterised by `dnum` rather than picked —
live in `../superpowers/plans/` and `../superpowers/specs/` with their
reasoning. This directory is only for questions where an outside answer would
genuinely change what we build.
