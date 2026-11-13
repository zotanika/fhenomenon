"""Turn measured semantic contracts into reviewable kernel engineering issues."""

import copy
import hashlib
import json
import re
from urllib.parse import urlsplit

from .costs import price_kernel_requests

_CONTRACT_FIELDS = (
    "operation",
    "input_contracts",
    "keyword_contracts",
    "output_contract",
    "acceptance",
)
_CONTEXT_FIELDS = (
    "workload",
    "hardware",
    "limitations",
    "numerical_acceptance",
    "reproduce",
    "source_url",
    "evidence_url",
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _block(value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    fence = "`" * max(3, 1 + max((len(x) for x in re.findall(r"`+", text)), default=0))
    return f"{fence}json\n{text}\n{fence}"


def derive_kernel_issues(request_artifact, measurement, *, context=None, limit=1):
    """Rank complete measured signatures and return local, metadata-only drafts.

    No network calls. The application must review architecture disclosure before
    publication. Only compiler-exported contracts are supported; this function is
    not a redactor for arbitrary dictionaries. Context and numerical acceptance
    are application-supplied statements, not independently verified evidence.

    Stable IDs include semantic contracts, execution requirements and numerical
    acceptance. They exclude graph/node names, timings and implementation/catalog
    revisions, so another run of the same request reuses its issue. This is
    conservative equality, not a proof that different contracts are equivalent.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if request_artifact.get("contains_model_weights_or_training_values") is not False:
        raise ValueError("requires a metadata-only compiler request artifact")
    if measurement is None:
        raise ValueError("matching scope measurement is required")
    if context is not None and not isinstance(context, dict):
        raise ValueError("context must be an object")
    context = {
        k: copy.deepcopy(v) for k, v in (context or {}).items() if k in _CONTEXT_FIELDS
    }
    _json(context)
    for key in ("source_url", "evidence_url"):
        if key in context:
            value = context[key]
            if not isinstance(value, str):
                raise ValueError(f"{key} must be an HTTPS URL")
            parsed = urlsplit(value)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.username
                or re.search(r"[\s<>]", value)
            ):
                raise ValueError(f"{key} must be an HTTPS URL without credentials")

    # An explicit zero-cost offer gives an optimistic bound, not a forecast.
    prices = price_kernel_requests(
        request_artifact,
        measurement,
        [
            {
                "signature": request["signature"],
                "assumed_seconds_per_invocation": 0.0,
                "setup_seconds": 0.0,
                "transfer_seconds": 0.0,
            }
            for request in request_artifact["requests"]
        ],
    )
    groups = {r["signature"]: r for r in request_artifact["requests"]}
    ranked = sorted(
        (
            row
            for row in prices["offers"]
            if row["status"] == "conditional_estimate"
            and row["measured_covered_seconds"] > 0
        ),
        key=lambda row: (-row["measured_covered_seconds"], row["signature"]),
    )
    environment = copy.deepcopy(request_artifact["environment"])
    target = copy.deepcopy(environment)
    target["runtime"].pop("implementation_identity", None)
    target["runtime"].pop("kernel_catalog", None)
    drafts = []
    for row in ranked[:limit]:
        request = groups[row["signature"]]
        missing = [key for key in _CONTRACT_FIELDS if key not in request]
        if missing:
            raise ValueError(f"engineering request lacks {', '.join(missing)}")
        contract = {key: copy.deepcopy(request[key]) for key in _CONTRACT_FIELDS}
        operation = contract["operation"]
        if not isinstance(operation, str) or not re.fullmatch(
            r"[A-Za-z0-9_.:]{1,120}", operation
        ):
            raise ValueError("operation must be a bounded semantic operation name")
        identity = {
            "schema_version": 1,
            "contract": contract,
            "target": target,
            "numerical_acceptance": context.get("numerical_acceptance"),
        }
        request_id = hashlib.sha256(_json(identity).encode()).hexdigest()
        marker = f"<!-- fhenomenon-kernel-request:v1:{request_id} -->"
        baseline = prices["measured_baseline_seconds"]
        covered = row["measured_covered_seconds"]
        fraction = covered / baseline
        runtime = environment["runtime"]
        profile = runtime["profile"]
        toy = isinstance(profile, dict) and (
            profile.get("test_only")
            or profile.get("toy_profile")
            or profile.get("name") == "NativeTest"
        )
        qualification = (
            "NativeTest / test-only profile: insecure numerical fixture; no production security or full-model performance claim."
            if toy
            else "This report does not independently establish cryptographic security or application accuracy."
        )
        current = {
            "implementations": sorted(
                set(request.get("current_implementations", {}).values())
            )
            or [request.get("current_implementation")],
            "selected_requirements": request.get("selected_requirements"),
            "declined_candidates": request.get("declined_candidates", []),
        }
        links = "\n".join(
            f"{name}: <{context[key]}>"
            for key, name in (
                ("source_url", "Source"),
                ("evidence_url", "Evidence and reproduction artifacts"),
            )
            if key in context
        )
        body = f"""{marker}
## Derived opportunity

`{operation}` has {row["occurrences"]} measured invocation(s), accounting for
**{covered:.6f} s / {baseline:.6f} s ({fraction:.2%})** of the completed Scope.
Priority is measured exclusive wall cost among fully measured signatures.
This is a semantic tensor operation; its implementation may fuse multiple CKKS operations.

| Measurement | Value |
| --- | --- |
| Samples in the measured cohort | {prices["sample_count"]} |
| Mean cost per invocation | {covered / row["occurrences"]:.6f} s |
| Timing boundary | `synchronized_scope_execution` |
| Attribution | `exclusive_synchronized_wall` |
| Signature | `{row["signature"]}` |
| Evidence fingerprint | `{prices["request_fingerprint"]}` |

Removing this entire cost with **zero replacement, setup and transfer time** would
leave {row["conditional_whole_scope_seconds"]:.6f} s. This is an optimistic substitution
bound, **not a measured speedup** or a delivery target. Actual offers must supply all
three costs; unchanged surrounding work, layout, levels and accuracy are assumptions.
Capture, input encryption and final decryption are outside this timing boundary.
{prices["qualification"]}

{qualification}

## Required semantic contract

{_block(contract)}

## Current realization

{_block(current)}

## Execution requirements and measured implementation

{_block(environment)}

## Acceptance before substitution

- [ ] Match input/output shapes, visibility, dtype, operation arguments and all live-outs.
- [ ] Declare representation, packing, scale/level transitions and required evaluation keys.
- [ ] Declare workspace, retained memory, transfer, setup and synchronization boundaries.
- [ ] Keep secret work inside the Scope, with zero intermediate decryption or private host fallback.
- [ ] Supply input-domain/range requirements and application-approved numerical tolerances; unresolved tolerances prevent acceptance.
- [ ] Validate final outputs and gradients, then measure the replacement under the same Scope boundary and disclose all overheads.

## Application-supplied context

{_block(context)}

{links}

Generated from compiler contracts and bound execution measurements. The request is
an engineering interface, not a correctness proof. Contract geometry can disclose
model architecture; review this draft before sharing it outside the model owner.
"""
        if len(body) > 60000:
            raise ValueError(
                "issue exceeds 60000 characters; use a smaller reviewed request context"
            )
        drafts.append(
            {
                "schema_version": 1,
                "request_id": request_id,
                "title": f"Kernel request: {operation} [{request_id[:12]}]",
                "body": body,
                "operation": operation,
                "measured_seconds": covered,
                "scope_fraction": fraction,
            }
        )
    return drafts
