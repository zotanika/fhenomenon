# Private spectrum monitoring

## Why combine sensor streams on a server?

PNNL's 2026 [ESAMS field-demonstration report (PNNL-39487)](https://www.pnnl.gov/publications/cloud-based-demonstration-eastern-interconnection-situational-awareness-monitoring)
describes a six-month AWS deployment receiving synchrophasor measurements from
ISO New England and PJM for wide-area oscillation detection and source
localization. It gives a concrete reason to combine measurements across sites
and organizations: a local device's DSP does not have that shared view.

This example explores encrypted cross-spectrum accumulation motivated by that
setting, using synthetic frequency bins. It does not reproduce ESAMS's
detection/localization algorithms or claim that the PNNL deployment used FHE.
The [report's security discussion](https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-39487.pdf#page=22)
explicitly identifies protection of shared data and derived results, controlled
access and auditing as requirements for long-term participation. It describes
conventional cloud controls, including encryption in transit and at rest.
This supports the need for data protection; whether operators would additionally
require the analysis service itself to remain unable to read their measurements
is a deployment question to validate with them.

## Shared insight without exposing the raw signal

An operator may want the benefits of a shared view while keeping control of its
own measurements. FHE offers an appealing way to separate those concerns:
encrypt the frequency bins, compute their cross-spectra on the server, and
decrypt the permitted results. The input need not be deliberately distorted
with privacy noise to keep it hidden from the evaluator. Phase and correlation
are computed with the same analysis formula; this sample measures CKKS's
approximation error against the cleartext result.

Differential privacy addresses a different disclosure boundary. Local DP
perturbs contributors' inputs, creating a signal-fidelity tradeoff; central DP
can instead perturb released results while trusting a data curator. FHE can
protect the computation, and DP can still be applied to the outputs if their
disclosure needs statistical protection. These distinctions and combinations
are described in [NIST SP 800-226, Section 4.2](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-226.pdf#page=65).
The story demonstrated here is encrypted signal analysis without added input
privacy noise. The intended application is shared anomaly detection: multiple
operators contribute secret measurements, and the monitor learns the agreed
joint verdict. The example runs normal, local-disturbance and shared-oscillation
scenarios through encrypted spectral analysis and a trusted final detector.
The four contributors are simulated under one key owner in one process.

## Run

Build and run with the native CKKS plugin for your platform. This example is
built even with `BUILD_TEST=OFF`.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DOCUMENTATION=OFF
cmake --build build --target fhn-private-spectrum
build/bin/fhn-private-spectrum build/lib/libckks_fhn.dylib --frames 4 --compare
# Run only the shared-oscillation case:
build/bin/fhn-private-spectrum build/lib/libckks_fhn.dylib --scenario shared --frames 4
```

Use `libckks_fhn.so` on Linux. `--help` also works without a plugin path.

| Option | Effect |
| --- | --- |
| `<plugin>` | Path to the native CKKS shared library. |
| `--scenario NAME` | `all` (default), `normal`, `local`, or `shared`. Each case has a baseline window and two observation windows. |
| `--frames N` | Integer from 2 to 16, default 4, per window. More frames do not increase multiplication depth, but do increase accumulated magnitude and error. |
| `--no-scope-kernels` | Keep scope recording, shared preparations and ordinary pair recipes; skip span matching. Default is pair-tile selection. |
| `--compare` | Run pair tiles, ordinary pair recipes, and native operations on the selected cases. Check every spectral bin and compare window verdicts; the native path also shares one reference conjugate per frame. |
| `-h`, `--help` | Explain options without loading the plugin. |

Follow `run()` in [Workflow.cpp](Workflow.cpp) for the scenario workflow and
`EncryptedMonitor::append()` for the two `Session::run` scopes and encrypted
state. [Client.cpp](Client.cpp) owns encryption, decryption, validation and the
trusted [detector](Detector.cpp). [The contribution](../../contributions/spectral/README.md)
shows how to replace pair or pair-tile implementations through public interfaces.

The fixture represents selected, normalized local FFT bins from four simulated
operators, with one sensor each.
Each frame carries eight bins per sensor, one complex phasor per CKKS slot.
Phasors have nonzero imaginary parts and include magnitudes above one, bounded
by 2.78 in these scenarios. The fixture supplies these phasors directly. Sensor 0 is the reference,
sensor 1 has a stable phase offset, sensor 2 varies its phase across frames,
and sensor 3 has a stable offset with a varying amplitude.

These are synthetic post-FFT inputs, not a recording or an implementation of
FFT, sensor synchronization or calibration. The phase convention is
`sensor phase - reference phase`; sensor 1's bin-zero offset is -0.65 radians.
The eight logical slots repeat across the native profile's 4096 physical slots.
This small example demonstrates composition and reuse, not efficient full-slot
packing or production monitoring throughput.

A single trusted owner encrypts the frames, provisions multiplication and
conjugation evaluation keys, and decrypts the accumulated spectra internally
to validate them and derive alarms. The
encrypted evaluator receives ciphertexts and evaluation services in the
same process. Real deployments would need a separate owner/evaluator
boundary. The example has no rotation keys because bins are independent.

## What the monitor detects

Each case begins with a baseline, then processes two observation windows.
The detector has no access to the scenario label. It scans all eight bins for
increased mean power and coherent observations across operators.

| Synthetic case | Observation | Expected monitor result |
| --- | --- | --- |
| `normal` | Baseline amplitudes continue, including already-correlated signals. | No shared alarm. |
| `local` | Operator 1's amplitude doubles at bin 3 in both windows. | No shared alarm: a local change lacks joint corroboration. |
| `shared` | Operators 0, 1 and 3 double their amplitudes at bin 3 in both windows. | No alarm after the first observation; a shared alarm after the second. |

Operator 1 sees the same local signal in the `local` and `shared` cases. The
other contributors' observations let the joint detector distinguish them.

The fixed illustrative rule requires the reference and at least two other
operators to have mean-power gains above 2.25 relative to their own baselines.
The other participating operators must also have magnitude-squared coherence
above 0.8 with the reference. The same bin must meet these conditions in two
consecutive observation windows. Power gains are normalized by each window's
frame count. High coherence alone is not an anomaly.

Comparison margins are 0.05 for power gain and 0.02 for coherence. Ambiguous or
invalid evidence does not count as a confirmed positive; an indeterminate
verdict represents uncertainty when it affects the decision. This fixed rule
and these thresholds illustrate the workflow, rather than a calibrated alarm
for a real power grid. It detects the specified reference-participating event;
it does not cover every subset of operators or establish the event's cause.

The monitor receives window verdicts, not decoded contributor spectra. The
trusted fixture checks all complex bins against a cleartext oracle and checks
the literal scenario outcomes. `--compare` additionally verifies that changing
kernel granularity leaves those outcomes unchanged.

## Ownership and result disclosure

The intended consumer of the anomaly verdict is the monitor. Contributors
supply measurements; returning decrypted spectra to each contributor is not
the application goal. The single-owner fixture decrypts spectra internally to
validate the computation and apply the detector; the application displays its
alarms and public execution statistics.

For this example's next steps, assume the monitor honestly executes an agreed
detector on the intended contributors and time window. Proofs against a
malicious evaluator or a general authorization protocol are not prerequisites
for extending the workload. A joint verdict generally leaves many possible
inputs consistent with the result. Nonlinearity alone does not establish that
property; exposing the internal per-sensor spectra would disclose more than
the monitor-facing verdict.

The input secret key must remain outside the evaluator if measurements are to
stay hidden from it. A trusted decryption role can deliver the agreed result
to the monitor; deployment key custody remains separate work. Giving the
monitor the input secret key would also let it decrypt accessible inputs.
The current same-process owner/evaluator split demonstrates API roles, not an
isolated security boundary or a multi-owner protocol.

Native CKKS already generates secret, public and evaluation keys and separates
the client and evaluator objects. The legacy [KeyManager](../../include/KeyManager.h)
facade records configured key paths; it does not provide cross-operator key
distribution or decryption authorization. Serialized ciphertext transport and
cross-process deployment are also outside this example.

## TODO: encrypted anomaly verdicts

The current detector runs in the trusted decryption role. Keep CKKS for
complex-valued analysis and accumulation; add a
CKKS-to-CGGI bridge for encrypted comparison and Boolean alarm logic when that
backend connection is available. Convert back to CKKS only when subsequent
approximate arithmetic needs it. Neither direction is implemented today.

The comparison contract must account for CKKS approximation and conversion
error near the threshold, with a stated margin or indeterminate outcome.
Until the bridge exists, any cleartext thresholding belongs explicitly to the
trusted decryption role. Scheme switching is a future kernel contribution
boundary; the lightweight runtime and reuse across scopes remain the basis
for composing the analysis.

## Scoped computation

For each frame the evaluator prepares the conjugated reference once.
The first `Session::run` scope records two adjacent spectral pairs, exports the reference
conjugate and preparation, and exports a cross product as an ordinary
ciphertext. The second scope imports that same preparation for two more
pairs and uses the earlier cross product in native accumulation. The
preparation belongs to one frame and is rebuilt for the next. Native
conjugation and multiplication compute the remaining auto-spectra, and
native additions replace the accumulated ciphertexts without retaining
prior frames. The native baseline also shares one reference conjugate
within each frame. `resetWindow()` releases the seven retained ciphertexts
between windows while preserving the session and registered recipes. Baseline
statistics and per-bin persistence live in the trusted detector, not in an
ever-growing encrypted execution history.

For each bin the diagnostic helper can compute the signed phase `arg(C_i)` and
magnitude-squared coherence `|C_i|² / (P_i P_0)` from decrypted sums
`C_i = Σ X_i conj(X_0)` and `P_i = Σ |X_i|²`. The phase and coherence
are reported as unavailable if powers, denominator, or cross magnitude
fall at or below the diagnostic floor or if the calculation is nonfinite.
The helper also treats materially imaginary power or coherence above one
as unavailable, with a small tolerance for CKKS error.
These decoded diagnostics remain available for low-level validation; the
application reports only joint alarms. Every complex cross and power result
is checked against an independent cleartext oracle with an absolute tolerance
of 1e-5.

The output reports client key setup, encryption and decryption separately
from evaluator append time. Peak accounted bytes cover retained runtime
resources and exclude keys, scratch, and allocator overhead. Selected
recipes, matched span sizes, and frame-local preparation identities show
the dispatch and reuse. The pair tile groups recipe dispatch. Timing
comparisons are observations for this run and machine.
