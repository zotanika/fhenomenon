"""Conditional kernel replacement costs, separate from measurement collection.

This module consumes metadata-only engineering requests and caller-supplied
timing evidence. It does not benchmark providers, certify instrumentation or
infer an accelerator speedup from a CPU/reference implementation.
"""

# Malformed serialized record values, including JSON types, use one error type.
# ruff: noqa: TRY004

import copy
import hashlib
import json
import math

_BOUNDARY = "synchronized_scope_execution"
_ATTRIBUTION = "exclusive_synchronized_wall"


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _version(value, name):
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError(f"{name} requires schema_version 1")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _count(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seconds(value, name, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds finite numeric range") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return result


def _sum(values, name):
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds finite numeric range") from exc
    return _seconds(result, name)


def _json(value, name):
    """Compare JSON metadata exactly, including extra fields and number types."""

    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{name} must contain string object keys")
            for nested in item.values():
                check(nested)
        elif isinstance(item, list):
            for nested in item:
                check(nested)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError(f"{name} must contain JSON metadata only")

    check(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite JSON metadata") from exc


def _requests(artifact):
    _version(_object(artifact, "request artifact"), "request artifact")
    environment = _object(artifact.get("environment"), "request environment")
    serialized = _json(artifact, "request artifact")
    fingerprint = "sha256:" + hashlib.sha256(serialized.encode()).hexdigest()
    if "runtime" in environment:
        _object(environment["runtime"], "environment.runtime")
    entries = artifact.get("requests")
    if not isinstance(entries, list):
        raise ValueError("requests must be a list")
    groups, nodes = {}, {}
    for entry in entries:
        entry = _object(entry, "request")
        signature = _text(entry.get("signature"), "request signature")
        if signature in groups:
            raise ValueError("duplicate request signature")
        coverage = entry.get("nodes")
        if not isinstance(coverage, list) or not coverage:
            raise ValueError("request node coverage must be a nonempty list")
        if _count(entry.get("occurrences"), "occurrences") != len(coverage):
            raise ValueError("occurrences must equal explicit node coverage")
        for node in coverage:
            _text(node, "node name")
        implementations = entry.get("current_implementations")
        if implementations is not None:
            _object(implementations, "current_implementations")
            if set(implementations) != set(coverage):
                raise ValueError(
                    "implementation mapping must match node coverage exactly"
                )
        for node in coverage:
            if node in nodes:
                raise ValueError("duplicate or overlapping request node coverage")
            implementation = _text(
                implementations[node]
                if implementations is not None
                else entry.get("current_implementation"),
                "current implementation",
            )
            if implementation == "mixed":
                raise ValueError(
                    "mixed implementation requires a per-node implementation mapping"
                )
            nodes[node] = (signature, implementation)
        groups[signature] = entry
    return environment, groups, nodes, fingerprint


def _measure(evidence, environment, nodes, fingerprint):
    _version(_object(evidence, "measurement"), "measurement")
    measured_environment = _object(
        evidence.get("environment"), "measurement environment"
    )
    if _json(environment, "request environment") != _json(
        measured_environment, "measurement environment"
    ):
        raise ValueError("measurement environment differs from request environment")
    if evidence.get("request_fingerprint") != fingerprint:
        raise ValueError(
            "measurement request fingerprint differs from the complete request artifact"
        )
    runtime = _object(environment.get("runtime"), "environment.runtime")
    _text(runtime.get("backend_kind"), "environment.runtime.backend_kind")
    for key in ("device", "profile"):
        if not _object(runtime.get(key), f"environment.runtime.{key}"):
            raise ValueError(
                f"environment.runtime.{key} must identify the measured configuration"
            )
    if evidence.get("timing_boundary") != _BOUNDARY:
        raise ValueError(
            f"timing boundary must be {_BOUNDARY}, excluding capture/client work"
        )
    if evidence.get("attribution_kind") != _ATTRIBUTION:
        raise ValueError(
            f"attribution kind must be {_ATTRIBUTION}, not overlapping GPU event durations"
        )
    sample_count = _count(evidence.get("sample_count"), "sample_count")
    baseline = _seconds(
        evidence.get("whole_scope_wall_seconds"),
        "whole_scope_wall_seconds",
        positive=True,
    )
    entries = evidence.get("node_measurements")
    if not isinstance(entries, list):
        raise ValueError("node_measurements must be a list")
    times = {}
    for entry in entries:
        entry = _object(entry, "node measurement")
        node = _text(entry.get("node"), "measurement node")
        if node in times:
            raise ValueError("duplicate or overlapping measured node coverage")
        if node not in nodes:
            raise ValueError("measured node is outside request coverage")
        signature, implementation = nodes[node]
        if entry.get("signature") != signature:
            raise ValueError("measured signature differs from request node signature")
        if entry.get("current_implementation") != implementation:
            raise ValueError(
                "measured implementation differs from current node implementation"
            )
        if implementation == "unsupported":
            raise ValueError(
                "unsupported implementation cannot have an executed node measurement"
            )
        times[node] = _seconds(entry.get("seconds"), "node measured seconds")
    total = _sum(times.values(), "total attributable seconds")
    if total > baseline and not math.isclose(
        total, baseline, rel_tol=1e-12, abs_tol=0.0
    ):
        raise ValueError("exclusive node attributions exceed whole-scope wall time")
    return baseline, sample_count, times


def _projection(baseline, covered, replacement, overhead):
    duration = _sum(
        (max(0.0, baseline - covered), replacement, overhead), "projected scope seconds"
    )
    speedup = baseline / duration if duration else None
    # A zero or subnormal assumed duration is an idealized bound, not a
    # finite speedup claim suitable for a JSON report.
    if speedup is not None and not math.isfinite(speedup):
        speedup = None
    return {
        "measured_covered_seconds": covered,
        "assumed_kernel_seconds": replacement,
        "assumed_overhead_seconds": overhead,
        "conditional_whole_scope_seconds": duration,
        "conditional_speedup": speedup,
    }


def price_kernel_requests(request_artifact, measurement, offers):
    """Price explicit kernel offers against compatible whole-scope evidence.

    ``request_artifact`` is the version-1 output of ``kernel_requests()``. Each
    request binds a unique signature to explicit, disjoint node names, its
    occurrence count, and ``current_implementations`` (node -> implementation).
    Older artifacts may use one homogeneous ``current_implementation`` string.

    ``measurement`` is None or a version-1 object containing:

    * ``request_fingerprint``: the fingerprint returned by
      ``price_kernel_requests(request_artifact, None, [])``. It binds the entire
      request artifact, including graph identity and unmeasured node coverage,
      so partial timing rows cannot accidentally price another scope's baseline.
    * ``environment``: exactly the request's JSON metadata, including the
      runtime's nonempty ``backend_kind``, ``device`` and ``profile``. Callers
      must identify actual parameters, binary revisions and hardware here;
      the estimator cannot establish that provenance itself.
    * ``timing_boundary="synchronized_scope_execution"`` and
      ``attribution_kind="exclusive_synchronized_wall"``. Times exclude capture,
      input encryption and terminal decryption. Per-node wall costs must be
      nonoverlapping and additive; raw concurrent GPU event durations do not
      qualify. Fused shared spans need a defensible exclusive attribution or
      must be left unpriced.
    * Positive ``sample_count`` and ``whole_scope_wall_seconds``: measured
      arithmetic means from the same completed sample cohort.
    * ``node_measurements``: rows with ``node``, ``signature``,
      ``current_implementation`` and measured mean ``seconds``. Partial coverage
      is allowed, but an offer needs every node of its signature measured.

    ``offers`` is a list of objects with a request ``signature`` and finite,
    nonnegative ``assumed_seconds_per_invocation``, ``setup_seconds`` and
    ``transfer_seconds``. All three costs are required; a claimed zero must be
    explicit. Setup and transfer are total additional costs per entire scope
    for that offer, charged once, not implicitly amortized. Offers must cover
    the same computation, packing, accuracy and intermediate live-outs as their
    request; this function does not certify numerical or cryptographic validity.

    Each estimate is baseline - covered measured cost + assumed invocation
    latency * occurrences + setup + transfer. Unoffered/unmeasured work retains
    its measured contribution in the baseline. The combined estimate lists
    exactly the priced signatures; unpriced offers are never filled with zero
    or extrapolated from other nodes. Invalid identities, duplicated coverage,
    timing provenance and numeric costs raise ValueError, even on unpriced
    offers. Zero estimated duration yields no finite speedup ratio.

    Returned numbers are conditional estimates, never measured improvements.
    Reference-provider measurements apply only to that reference provider and
    cannot establish an encrypted speedup. Inputs are not modified.
    """
    environment, groups, nodes, fingerprint = _requests(request_artifact)
    if not isinstance(offers, list):
        raise ValueError("offers must be a list")
    proposed, seen = [], set()
    for entry in offers:
        entry = _object(entry, "offer")
        signature = _text(entry.get("signature"), "offer signature")
        if signature not in groups:
            raise ValueError("offer signature is outside request coverage")
        if signature in seen:
            raise ValueError("duplicate offer signature would overlap node coverage")
        seen.add(signature)
        costs = {
            name: _seconds(entry.get(name), name)
            for name in (
                "assumed_seconds_per_invocation",
                "setup_seconds",
                "transfer_seconds",
            )
        }
        proposed.append({"signature": signature, **costs})
    baseline = sample_count = None
    times = {}
    if measurement is not None:
        baseline, sample_count, times = _measure(
            measurement, environment, nodes, fingerprint
        )
    priced, rows = [], []
    for entry in proposed:
        signature = entry["signature"]
        group = groups[signature]
        row = {
            **entry,
            "nodes": list(group["nodes"]),
            "occurrences": len(group["nodes"]),
        }
        missing = [node for node in group["nodes"] if node not in times]
        if baseline is None or missing:
            row.update(
                status="unpriced",
                reason="missing_measurement"
                if baseline is None
                else "missing_node_measurements",
                missing_nodes=missing,
            )
        else:
            covered = _sum(
                (times[node] for node in group["nodes"]), "covered measured seconds"
            )
            replacement = _seconds(
                entry["assumed_seconds_per_invocation"] * row["occurrences"],
                "total assumed kernel seconds",
            )
            overhead = _sum(
                (entry["setup_seconds"], entry["transfer_seconds"]),
                "assumed overhead seconds",
            )
            row.update(
                status="conditional_estimate",
                **_projection(baseline, covered, replacement, overhead),
            )
            priced.append(row)
        rows.append(row)
    combined = None
    if priced:
        combined = {
            "status": "conditional_estimate",
            "priced_signatures": [row["signature"] for row in priced],
            **_projection(
                baseline,
                _sum(
                    (row["measured_covered_seconds"] for row in priced),
                    "combined covered seconds",
                ),
                _sum(
                    (row["assumed_kernel_seconds"] for row in priced),
                    "combined assumed kernel seconds",
                ),
                _sum(
                    (row["assumed_overhead_seconds"] for row in priced),
                    "combined assumed overhead seconds",
                ),
            ),
        }
    qualification = (
        "Conditional substitution under the declared environment and unchanged remaining work; "
        "not a measured speedup. Timing evidence and offer equivalence require independent validation."
    )
    backend_kind = environment.get("runtime", {}).get("backend_kind")
    if backend_kind == "noncryptographic_reference":
        qualification += " Noncryptographic reference measurements do not predict encrypted execution."
    return {
        "schema_version": 1,
        "request_fingerprint": fingerprint,
        "status": "unpriced"
        if not priced
        else (
            "conditional_estimate" if len(priced) == len(rows) else "partially_priced"
        ),
        "environment": copy.deepcopy(environment),
        "measurement_backend_kind": backend_kind,
        "measured_baseline_seconds": baseline,
        "sample_count": sample_count,
        "timing_boundary": _BOUNDARY if measurement is not None else None,
        "offers": rows,
        "combined": combined,
        "unoffered_signatures": [
            signature for signature in groups if signature not in seen
        ],
        "qualification": qualification,
    }
