# Scoped OpenVLA LoRA training

This example captures the OpenVLA image encoders, projector, language decoder,
loss, full activation backward graph and LoRA parameter updates. All secret
computation is recorded in one Fhenomenon Scope. Frozen base weights are public;
images, dense instruction/action encodings and adapters are encrypted. Only final
outputs are decrypted by the client after reduction to the minimum level.

The model definition is pinned to [OpenVLA-7B revision 47a0ec7](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f).
The numerical `micro` and `tiny` profiles reduce dimensions and initialize random
weights while keeping all model stages. `micro` has a hidden width of 4 and is
not a 7B execution. `full7b_meta` captures full-size shapes without loading weights
or performing encrypted computation. These fixtures establish execution coverage,
not model utility, convergence or deployment performance.

## Setup

From the repository root, install CPU PyTorch 2.7.1 and torchvision 0.22.1 from an
appropriate PyTorch index, then install the remaining example dependencies:

```sh
python -m pip install -r examples/openvla_training/requirements.txt
python -m pip install './python[ml]'
python -m examples.openvla_training.source
```

The last command downloads only `config.json`, `configuration_prismatic.py` and
`modeling_prismatic.py` into `build/openvla-model`. No checkpoint weights are
needed. Capture loads this trusted model Python locally. `--snapshot` or
`FHN_OPENVLA_SNAPSHOT` selects another prepared directory. Capture itself uses
local files only and makes no model download request.

Build the [native provider](../../docs/kernel-providers.md) with CUDA support
before the encrypted GPU example. Python uses CPU tensors for trusted capture;
CUDA encrypted work runs in the native provider.

## Reproduce

```sh
python -m examples.openvla_training.run_graph --profile micro \
  --output build/openvla-graph.json
python -m examples.openvla_training.calibration --profile micro \
  --output build/openvla-ranges.json
FHN_LIBRARY_DIR="$PWD/build-cuda/lib" \
  python -m examples.openvla_training.run_training \
  --backend runtime-cuda --crypto-profile NativeTest --profile micro \
  --data-seed 2029 --public-range-contract build/openvla-ranges.json \
  --max-seconds 1800 --output build/openvla-result.json
```

**NativeTest is an insecure numerical fixture.** Public calibration uses separate
synthetic examples and binds the ranges to graph/public-weight identities. It is
an empirical requirement, not a proof for arbitrary private data. The chosen
ranges can fail on another distribution; no hidden recalibration or plaintext
fallback occurs during encrypted execution.

Use `runtime-cpu` and a CPU provider directory for the same execution contract.
`--decomposition-only` disables semantic tensor contributions for a controlled
comparison. `reference-exact` and `reference-polynomial` are noncryptographic
oracles. Successful reports include output errors, whole-Scope command counts,
contribution coverage, resource accounting and measurement-bound kernel requests.
Setup, capture, encrypted execution and terminal checks have separate timing.

```sh
python -m unittest examples.openvla_training.test_graph \
  examples.openvla_training.test_calibration examples.openvla_training.test_runner
```

## Source map

| File | Responsibility |
| --- | --- |
| `graph.py` | Pinned architecture, reduced/full-size profiles, one-hot lowering and joint training capture |
| `calibration.py` | Public empirical range profile and identity checks |
| `interpreter.py` | Example-specific range policy over the shared model interpreter |
| `run_training.py` | Shared compiler/registry, encrypted binding, one Scope and output validation |
| `run_graph.py` | Capture-only report |
| `source.py` | Explicit pinned model-definition download |

Generic lowering, contracts, kernel selection and costs live in
`python/fhenomenon/ml`; the native integration is in `python/fhenomenon/backends`.
Provider arithmetic is in `cpp/providers/ckks_tensor`. This example does not
supply a second framework or a model-specific cryptographic engine.

## Validation scope

The reference CUDA provider has completed the `micro` fixture on GB10 with
278,003 scoped commands, 53 `tensor.public_linear` contributions and no
intermediate decryption. All loss, adapter-gradient and update outputs met the
runner's numerical checks. This used `NativeTest`, random reduced weights and
the supplied empirical range contract. It is execution evidence for that
fixture, not a full-size training or security result. The commands above rerun
the checks on the selected implementation rather than reusing stored results.
