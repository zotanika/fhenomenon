# Language-neutral runtime behavior

`runtime-v1.json` is consumed by both `cpp/test/LanguageConformanceTest.cpp`
and `rust/fhenomenon/tests/conformance.rs`. Its expected values are written by
hand and are not generated from either implementation.

Each case contains integer imports, commands and exported symbol indices.
Imports occupy the first indices. Each command appends its declared output
symbols, including unused outputs. `add` returns one sum, `split` returns
`[x, -x]`, and `observe` accepts one input and returns no outputs. `missing`
passes the metadata contract but has no recipe; `invalid` fails inference.
Each retained integer resource is charged eight bytes.

The `stats` object records the expected Scope progress even on failure. Repeated
input edges and exported aliases count separately; repeated imports of the same
identity do not. A failed or declined span never counts as completed. Success
and failure both release unpublished intermediates when the scope closes.

Optional fields configure resource/capture caps, kernel selection or an invalid
provider output. `two_adds` is a small coarse kernel that returns precisely the
span's live outputs; `decline` never accepts. Lookahead shortens a view, whereas
exhausting the matching-attempt budget fails without fallback.

These fixtures cover observable common behavior. Language-specific ownership,
panic/exception handling, ABI layout, native arithmetic and large-graph memory
checks remain in focused tests. New contracts should add a failing case here
when their behavior is shared by both implementations; API syntax need not be
identical.
