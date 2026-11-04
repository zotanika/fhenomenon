# Private spectrum monitoring

Build and run with the native CKKS plugin for your platform. This example is
built even with `BUILD_TEST=OFF`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DOCUMENTATION=OFF
cmake --build build --target fhn-private-spectrum
build/bin/fhn-private-spectrum build/lib/libckks_fhn.dylib --frames 4 --compare
```

Use `libckks_fhn.so` on Linux. `--help` also works without a plugin path.

| Option | Effect |
| --- | --- |
| `<plugin>` | Path to the native CKKS shared library. |
| `--frames N` | Integer from 2 to 16, default 4. One finite accumulation window; more frames do not increase multiplication depth, but do increase magnitude and error. |
| `--no-scope-kernels` | Keep scope recording, shared preparations and ordinary pair recipes; skip span matching. Default is pair-tile selection. |
| `--compare` | Run pair tiles, ordinary pair recipes, and native operations on the same deterministic fixture. Every path checks all bins; the native path also shares one reference conjugate per frame. |
| `-h`, `--help` | Explain options without loading the plugin. |

Follow [Client.cpp](Client.cpp) for owner-only encryption/decryption,
[Workflow.cpp](Workflow.cpp) for the two `Session::run` scopes and encrypted
state, and [the contribution](../../contributions/spectral/README.md) to replace
pair or pair-tile implementations through public interfaces.

The fixture represents selected, normalized local FFT bins from four sensors.
Each frame carries eight bins per sensor, one complex phasor per CKKS slot.
Phasors have nonzero imaginary parts and include magnitudes above one, bounded
by two. The fixture supplies these phasors directly. Sensor 0 is the reference,
sensor 1 has a stable phase offset, sensor 2 varies its phase across frames,
and sensor 3 has a stable offset with a varying amplitude.

These are synthetic post-FFT inputs, not a recording or an implementation of
FFT, sensor synchronization or calibration. The phase convention is
`sensor phase - reference phase`; sensor 1's bin-zero offset is -0.65 radians.
The eight logical slots repeat across the native profile's 4096 physical slots.
This small example demonstrates composition and reuse, not efficient full-slot
packing or production monitoring throughput.

A single trusted owner encrypts the frames, provisions multiplication and
conjugation evaluation keys, and decrypts the accumulated spectra. The
encrypted evaluator receives ciphertexts and evaluation services in the
same process. Real deployments would need a separate owner/evaluator
boundary. The example has no rotation keys because bins are independent.

For each frame the evaluator prepares the conjugated reference once.
The first `Session::run` scope records two adjacent spectral pairs, exports the reference
conjugate and preparation, and exports a cross product as an ordinary
ciphertext. The second scope imports that same preparation for two more
pairs and uses the earlier cross product in native accumulation. The
preparation belongs to one frame and is rebuilt for the next. Native
conjugation and multiplication compute the remaining auto-spectra, and
native additions replace the accumulated ciphertexts without retaining
prior frames. The native baseline also shares one reference conjugate
within each frame.

For each bin the client computes the signed phase `arg(C_i)` and
magnitude-squared coherence `|C_i|² / (P_i P_0)` from decrypted sums
`C_i = Σ X_i conj(X_0)` and `P_i = Σ |X_i|²`. The phase and coherence
are reported as unavailable if powers, denominator, or cross magnitude
fall at or below the diagnostic floor or if the calculation is nonfinite.
The client also treats materially imaginary power or coherence above one
as unavailable, with a small tolerance for CKKS error.
The CLI shows one selected bin and checks every complex cross and power
result against an independent cleartext oracle with an absolute tolerance
of 1e-5.

The output reports client key setup, encryption and decryption separately
from evaluator append time. Peak accounted bytes cover retained runtime
resources and exclude keys, scratch, and allocator overhead. Selected
recipes, matched span sizes, and frame-local preparation identities show
the dispatch and reuse. The pair tile groups recipe dispatch. Timing
comparisons are observations for this run and machine.
