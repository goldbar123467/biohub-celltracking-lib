"""Strict validation evidence for Biohub campaign decisions.

The organizer scorer produces per-sample raw counts and derived metrics.  This
module keeps those per-clip results as the source of truth, validates their
provenance, and recomputes every aggregate used for a campaign decision.

It intentionally has no dependency on the optional organizer packages.  The
matching itself remains the responsibility of the pinned scorer; this module
only applies the pinned ``per_sample_metrics`` and ``summarise`` conventions to
trusted raw scorer outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

SCHEMA_VERSION = 1
PINNED_SCORER_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
PINNED_SCORER_SHA256 = "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444"
ADJUSTMENT_ALPHA = 0.1
DIVISION_SCORE_WEIGHT = 0.1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")

_NO_ANNOTATED_NODES = "annotated_nodes_zero"
_NO_ESTIMATED_NODES = "estimated_nodes_not_positive"
_NO_EDGE_UNION = "edge_union_zero"
_NO_DIVISION_UNION = "division_union_zero"
_NO_MATCHED_NODES = "matched_annotated_nodes_zero"


class EvidenceValidationError(ValueError):
    """Raised when evidence cannot support the requested claim."""


class EvidenceClass(str, Enum):
    DEVELOPMENT = "development"
    PREVIOUSLY_INSPECTED_DIAGNOSTIC = "previously_inspected_diagnostic"
    LOCKED_EVALUATION = "locked_evaluation"
    OVERLAP_UNKNOWN = "overlap_unknown"


class TrainingMembership(str, Enum):
    EXCLUDED = "excluded"
    INCLUDED = "included"
    UNKNOWN = "unknown"


class EvaluationStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


@dataclass(frozen=True)
class MetricValue:
    """A finite metric or an explicit reason that it is undefined."""

    value: float | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.value is None:
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise EvidenceValidationError(
                    "an undefined metric must use null plus a non-empty reason"
                )
            return
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise EvidenceValidationError("metric value must be a finite number or null")
        if not math.isfinite(float(self.value)):
            raise EvidenceValidationError("metric values must be finite; use null plus reason")
        if self.reason is not None:
            raise EvidenceValidationError("a defined metric cannot also have an undefined reason")
        object.__setattr__(self, "value", float(self.value))

    @classmethod
    def defined(cls, value: float) -> MetricValue:
        return cls(float(value), None)

    @classmethod
    def undefined(cls, reason: str) -> MetricValue:
        return cls(None, reason)

    @classmethod
    def from_dict(cls, raw: Any, *, field: str) -> MetricValue:
        data = _require_mapping(raw, field)
        _require_exact_keys(data, {"value", "reason"}, field)
        try:
            return cls(data["value"], data["reason"])
        except EvidenceValidationError as exc:
            raise EvidenceValidationError(f"{field}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "reason": self.reason}


@dataclass(frozen=True)
class ClipProvenance:
    candidate_sha256: str
    git_commit: str
    source_bundle_sha256: str
    dependency_manifest_sha256: str
    effective_config_sha256: str
    model_weight_sha256: str | None
    model_weight_sha256_reason: str | None
    input_manifest_sha256: str
    split_manifest_sha256: str
    scorer_commit: str
    scorer_sha256: str
    training_membership: TrainingMembership
    training_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    used_for_development_selection: bool
    previously_inspected: bool

    _KEYS: ClassVar[set[str]] = {
        "candidate_sha256",
        "git_commit",
        "source_bundle_sha256",
        "dependency_manifest_sha256",
        "effective_config_sha256",
        "model_weight_sha256",
        "model_weight_sha256_reason",
        "input_manifest_sha256",
        "split_manifest_sha256",
        "scorer_commit",
        "scorer_sha256",
        "training_membership",
        "training_ids",
        "development_ids",
        "used_for_development_selection",
        "previously_inspected",
    }

    @classmethod
    def from_dict(cls, raw: Any) -> ClipProvenance:
        data = _require_mapping(raw, "provenance")
        _require_exact_keys(data, cls._KEYS, "provenance")
        membership = _parse_enum(
            TrainingMembership, data["training_membership"], "provenance.training_membership"
        )
        value = cls(
            candidate_sha256=_require_sha256(
                data["candidate_sha256"], "provenance.candidate_sha256"
            ),
            git_commit=_require_git(data["git_commit"], "provenance.git_commit"),
            source_bundle_sha256=_require_sha256(
                data["source_bundle_sha256"], "provenance.source_bundle_sha256"
            ),
            dependency_manifest_sha256=_require_sha256(
                data["dependency_manifest_sha256"],
                "provenance.dependency_manifest_sha256",
            ),
            effective_config_sha256=_require_sha256(
                data["effective_config_sha256"], "provenance.effective_config_sha256"
            ),
            model_weight_sha256=_optional_sha256(
                data["model_weight_sha256"], "provenance.model_weight_sha256"
            ),
            model_weight_sha256_reason=_optional_nonempty_str(
                data["model_weight_sha256_reason"],
                "provenance.model_weight_sha256_reason",
            ),
            input_manifest_sha256=_require_sha256(
                data["input_manifest_sha256"], "provenance.input_manifest_sha256"
            ),
            split_manifest_sha256=_require_sha256(
                data["split_manifest_sha256"], "provenance.split_manifest_sha256"
            ),
            scorer_commit=_require_git(data["scorer_commit"], "provenance.scorer_commit"),
            scorer_sha256=_require_sha256(data["scorer_sha256"], "provenance.scorer_sha256"),
            training_membership=membership,
            training_ids=_string_tuple(data["training_ids"], "provenance.training_ids"),
            development_ids=_string_tuple(data["development_ids"], "provenance.development_ids"),
            used_for_development_selection=_require_bool(
                data["used_for_development_selection"],
                "provenance.used_for_development_selection",
            ),
            previously_inspected=_require_bool(
                data["previously_inspected"], "provenance.previously_inspected"
            ),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.model_weight_sha256 is None:
            if not self.model_weight_sha256_reason:
                raise EvidenceValidationError(
                    "provenance.model_weight_sha256 requires a digest or a reason"
                )
        elif self.model_weight_sha256_reason is not None:
            raise EvidenceValidationError(
                "provenance.model_weight_sha256_reason must be null when a digest exists"
            )
        if self.scorer_commit != PINNED_SCORER_COMMIT:
            raise EvidenceValidationError(
                f"scorer commit is not the pinned organizer revision {PINNED_SCORER_COMMIT}"
            )
        if self.scorer_sha256 != PINNED_SCORER_SHA256:
            raise EvidenceValidationError(
                f"scorer source is not the pinned normalized source {PINNED_SCORER_SHA256}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "git_commit": self.git_commit,
            "source_bundle_sha256": self.source_bundle_sha256,
            "dependency_manifest_sha256": self.dependency_manifest_sha256,
            "effective_config_sha256": self.effective_config_sha256,
            "model_weight_sha256": self.model_weight_sha256,
            "model_weight_sha256_reason": self.model_weight_sha256_reason,
            "input_manifest_sha256": self.input_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "scorer_commit": self.scorer_commit,
            "scorer_sha256": self.scorer_sha256,
            "training_membership": self.training_membership.value,
            "training_ids": list(self.training_ids),
            "development_ids": list(self.development_ids),
            "used_for_development_selection": self.used_for_development_selection,
            "previously_inspected": self.previously_inspected,
        }


@dataclass(frozen=True)
class PerClipEvidence:
    schema_version: int
    run_id: str
    candidate_id: str
    clip_id: str
    embryo: str
    direction: str
    frame_ids: tuple[int, ...]
    evidence_class: EvidenceClass
    provenance: ClipProvenance
    predicted_nodes: int
    estimated_nodes: int
    annotated_nodes: int
    matched_annotated_nodes: int
    annotated_recall: MetricValue
    predicted_to_estimated_ratio: MetricValue
    localization_um_quantiles: Mapping[str, MetricValue]
    pre_cap_candidates: int
    capped_frames: int
    total_frames: int
    edge_tp: int
    edge_fp: int
    edge_fn: int
    raw_edge_jaccard: MetricValue
    adjusted_edge_jaccard: MetricValue
    division_tp: int
    division_fp: int
    division_fn: int
    division_jaccard: MetricValue
    predicted_forks_after_solver: int
    predicted_forks_after_postprocess: int
    runtime_seconds_by_stage: Mapping[str, float]
    peak_ram_bytes: int
    peak_vram_bytes: int
    fallback_count: int
    graph_path: str
    graph_sha256: str
    evaluation_complete: bool

    _KEYS: ClassVar[set[str]] = {
        "schema_version",
        "run_id",
        "candidate_id",
        "clip_id",
        "embryo",
        "direction",
        "frame_ids",
        "evidence_class",
        "provenance",
        "predicted_nodes",
        "estimated_nodes",
        "annotated_nodes",
        "matched_annotated_nodes",
        "annotated_recall",
        "predicted_to_estimated_ratio",
        "localization_um_quantiles",
        "pre_cap_candidates",
        "capped_frames",
        "total_frames",
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "raw_edge_jaccard",
        "adjusted_edge_jaccard",
        "division_tp",
        "division_fp",
        "division_fn",
        "division_jaccard",
        "predicted_forks_after_solver",
        "predicted_forks_after_postprocess",
        "runtime_seconds_by_stage",
        "peak_ram_bytes",
        "peak_vram_bytes",
        "fallback_count",
        "graph_path",
        "graph_sha256",
        "evaluation_complete",
    }

    @classmethod
    def from_dict(cls, raw: Any) -> PerClipEvidence:
        data = _require_mapping(raw, "per_clip_record")
        _require_exact_keys(data, cls._KEYS, "per_clip_record")
        localization = _require_mapping(
            data["localization_um_quantiles"], "localization_um_quantiles"
        )
        _require_exact_keys(localization, {"p50", "p90", "p95"}, "localization_um_quantiles")
        runtime = _require_mapping(data["runtime_seconds_by_stage"], "runtime_seconds_by_stage")
        if not runtime:
            raise EvidenceValidationError("runtime_seconds_by_stage cannot be empty")
        parsed_runtime = {
            _require_nonempty_str(k, "runtime stage"): _require_nonnegative_float(
                v, f"runtime_seconds_by_stage.{k}"
            )
            for k, v in runtime.items()
        }
        frame_ids = _integer_tuple(data["frame_ids"], "frame_ids")
        value = cls(
            schema_version=_require_int(data["schema_version"], "schema_version", minimum=1),
            run_id=_require_nonempty_str(data["run_id"], "run_id"),
            candidate_id=_require_nonempty_str(data["candidate_id"], "candidate_id"),
            clip_id=_require_nonempty_str(data["clip_id"], "clip_id"),
            embryo=_require_nonempty_str(data["embryo"], "embryo"),
            direction=_require_nonempty_str(data["direction"], "direction"),
            frame_ids=frame_ids,
            evidence_class=_parse_enum(EvidenceClass, data["evidence_class"], "evidence_class"),
            provenance=ClipProvenance.from_dict(data["provenance"]),
            predicted_nodes=_require_int(data["predicted_nodes"], "predicted_nodes"),
            estimated_nodes=_require_int(data["estimated_nodes"], "estimated_nodes"),
            annotated_nodes=_require_int(data["annotated_nodes"], "annotated_nodes"),
            matched_annotated_nodes=_require_int(
                data["matched_annotated_nodes"], "matched_annotated_nodes"
            ),
            annotated_recall=MetricValue.from_dict(
                data["annotated_recall"], field="annotated_recall"
            ),
            predicted_to_estimated_ratio=MetricValue.from_dict(
                data["predicted_to_estimated_ratio"],
                field="predicted_to_estimated_ratio",
            ),
            localization_um_quantiles={
                key: MetricValue.from_dict(localization[key], field=f"localization.{key}")
                for key in ("p50", "p90", "p95")
            },
            pre_cap_candidates=_require_int(data["pre_cap_candidates"], "pre_cap_candidates"),
            capped_frames=_require_int(data["capped_frames"], "capped_frames"),
            total_frames=_require_int(data["total_frames"], "total_frames"),
            edge_tp=_require_int(data["edge_tp"], "edge_tp"),
            edge_fp=_require_int(data["edge_fp"], "edge_fp"),
            edge_fn=_require_int(data["edge_fn"], "edge_fn"),
            raw_edge_jaccard=MetricValue.from_dict(
                data["raw_edge_jaccard"], field="raw_edge_jaccard"
            ),
            adjusted_edge_jaccard=MetricValue.from_dict(
                data["adjusted_edge_jaccard"], field="adjusted_edge_jaccard"
            ),
            division_tp=_require_int(data["division_tp"], "division_tp"),
            division_fp=_require_int(data["division_fp"], "division_fp"),
            division_fn=_require_int(data["division_fn"], "division_fn"),
            division_jaccard=MetricValue.from_dict(
                data["division_jaccard"], field="division_jaccard"
            ),
            predicted_forks_after_solver=_require_int(
                data["predicted_forks_after_solver"], "predicted_forks_after_solver"
            ),
            predicted_forks_after_postprocess=_require_int(
                data["predicted_forks_after_postprocess"],
                "predicted_forks_after_postprocess",
            ),
            runtime_seconds_by_stage=parsed_runtime,
            peak_ram_bytes=_require_int(data["peak_ram_bytes"], "peak_ram_bytes"),
            peak_vram_bytes=_require_int(data["peak_vram_bytes"], "peak_vram_bytes"),
            fallback_count=_require_int(data["fallback_count"], "fallback_count"),
            graph_path=_require_nonempty_str(data["graph_path"], "graph_path"),
            graph_sha256=_require_sha256(data["graph_sha256"], "graph_sha256"),
            evaluation_complete=_require_bool(data["evaluation_complete"], "evaluation_complete"),
        )
        value.validate()
        return value

    def validate(self) -> None:
        self.provenance.validate()
        if self.schema_version != SCHEMA_VERSION:
            raise EvidenceValidationError(
                f"unsupported per-clip schema_version {self.schema_version}"
            )
        if not self.frame_ids:
            raise EvidenceValidationError("frame_ids cannot be empty")
        if tuple(sorted(set(self.frame_ids))) != self.frame_ids:
            raise EvidenceValidationError("frame_ids must be unique and strictly increasing")
        if self.total_frames != len(self.frame_ids):
            raise EvidenceValidationError("total_frames must equal len(frame_ids)")
        if self.capped_frames > self.total_frames:
            raise EvidenceValidationError("capped_frames cannot exceed total_frames")
        if self.pre_cap_candidates < self.predicted_nodes:
            raise EvidenceValidationError(
                "pre_cap_candidates cannot be smaller than predicted_nodes"
            )
        if self.matched_annotated_nodes > self.annotated_nodes:
            raise EvidenceValidationError("matched_annotated_nodes cannot exceed annotated_nodes")
        if self.clip_id != self.embryo and not self.clip_id.startswith(self.embryo + "_"):
            raise EvidenceValidationError("clip_id does not belong to the declared embryo")
        self._validate_evidence_class()
        expected = _derived_metrics(
            predicted_nodes=self.predicted_nodes,
            estimated_nodes=self.estimated_nodes,
            annotated_nodes=self.annotated_nodes,
            matched_annotated_nodes=self.matched_annotated_nodes,
            edge_tp=self.edge_tp,
            edge_fp=self.edge_fp,
            edge_fn=self.edge_fn,
            division_tp=self.division_tp,
            division_fp=self.division_fp,
            division_fn=self.division_fn,
        )
        for name in (
            "annotated_recall",
            "predicted_to_estimated_ratio",
            "raw_edge_jaccard",
            "adjusted_edge_jaccard",
            "division_jaccard",
        ):
            _require_metric_equal(getattr(self, name), expected[name], name)
        localization = [self.localization_um_quantiles[k] for k in ("p50", "p90", "p95")]
        if self.matched_annotated_nodes == 0:
            for key, metric in zip(("p50", "p90", "p95"), localization):
                _require_metric_equal(
                    metric, MetricValue.undefined(_NO_MATCHED_NODES), f"localization.{key}"
                )
        else:
            if any(metric.value is None for metric in localization):
                raise EvidenceValidationError(
                    "localization quantiles are required when matched nodes exist"
                )
            values = [metric.value for metric in localization]
            assert all(v is not None for v in values)
            if values != sorted(values) or values[0] < 0:
                raise EvidenceValidationError(
                    "localization p50/p90/p95 must be finite, nonnegative, and ordered"
                )

    def _validate_evidence_class(self) -> None:
        provenance = self.provenance
        membership = provenance.training_membership
        in_training_ids = self.clip_id in provenance.training_ids
        if membership is TrainingMembership.EXCLUDED and in_training_ids:
            raise EvidenceValidationError(
                "training_membership is excluded but clip_id occurs in training_ids"
            )
        if membership is TrainingMembership.INCLUDED and not in_training_ids:
            raise EvidenceValidationError(
                "training_membership is included but clip_id is absent from training_ids"
            )
        if membership is TrainingMembership.UNKNOWN:
            if self.evidence_class is not EvidenceClass.OVERLAP_UNKNOWN:
                raise EvidenceValidationError(
                    "unknown training membership must be labeled overlap_unknown; "
                    "it cannot be labeled as a clean/locked evaluation"
                )
        elif self.evidence_class is EvidenceClass.OVERLAP_UNKNOWN:
            raise EvidenceValidationError("overlap_unknown requires unknown training membership")
        if self.evidence_class is EvidenceClass.LOCKED_EVALUATION:
            if membership is not TrainingMembership.EXCLUDED:
                raise EvidenceValidationError(
                    "locked_evaluation requires known exclusion from training"
                )
            if provenance.used_for_development_selection or provenance.previously_inspected:
                raise EvidenceValidationError(
                    "locked_evaluation cannot have selection use or prior inspection"
                )
        if (
            self.evidence_class is EvidenceClass.DEVELOPMENT
            and not provenance.used_for_development_selection
        ):
            raise EvidenceValidationError(
                "development evidence must record development-selection use"
            )
        if (
            self.evidence_class is EvidenceClass.PREVIOUSLY_INSPECTED_DIAGNOSTIC
            and not provenance.previously_inspected
        ):
            raise EvidenceValidationError(
                "previously_inspected_diagnostic must record prior inspection"
            )

    @property
    def edge_union(self) -> int:
        return self.edge_tp + self.edge_fp + self.edge_fn

    @property
    def division_union(self) -> int:
        return self.division_tp + self.division_fp + self.division_fn

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "clip_id": self.clip_id,
            "embryo": self.embryo,
            "direction": self.direction,
            "frame_ids": list(self.frame_ids),
            "evidence_class": self.evidence_class.value,
            "provenance": self.provenance.to_dict(),
            "predicted_nodes": self.predicted_nodes,
            "estimated_nodes": self.estimated_nodes,
            "annotated_nodes": self.annotated_nodes,
            "matched_annotated_nodes": self.matched_annotated_nodes,
            "annotated_recall": self.annotated_recall.to_dict(),
            "predicted_to_estimated_ratio": self.predicted_to_estimated_ratio.to_dict(),
            "localization_um_quantiles": {
                key: self.localization_um_quantiles[key].to_dict() for key in ("p50", "p90", "p95")
            },
            "pre_cap_candidates": self.pre_cap_candidates,
            "capped_frames": self.capped_frames,
            "total_frames": self.total_frames,
            "edge_tp": self.edge_tp,
            "edge_fp": self.edge_fp,
            "edge_fn": self.edge_fn,
            "raw_edge_jaccard": self.raw_edge_jaccard.to_dict(),
            "adjusted_edge_jaccard": self.adjusted_edge_jaccard.to_dict(),
            "division_tp": self.division_tp,
            "division_fp": self.division_fp,
            "division_fn": self.division_fn,
            "division_jaccard": self.division_jaccard.to_dict(),
            "predicted_forks_after_solver": self.predicted_forks_after_solver,
            "predicted_forks_after_postprocess": self.predicted_forks_after_postprocess,
            "runtime_seconds_by_stage": dict(self.runtime_seconds_by_stage),
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
            "fallback_count": self.fallback_count,
            "graph_path": self.graph_path,
            "graph_sha256": self.graph_sha256,
            "evaluation_complete": self.evaluation_complete,
        }


def build_per_clip_evidence(raw: Mapping[str, Any]) -> PerClipEvidence:
    """Build a strict record while deriving the five scorer/count metrics.

    ``raw`` must contain every :class:`PerClipEvidence` field except the five
    derived metric fields.  Derived values supplied by a worker are rejected so
    this boundary, rather than the worker assertion, determines their values.
    """

    data = dict(raw)
    derived_names = {
        "annotated_recall",
        "predicted_to_estimated_ratio",
        "raw_edge_jaccard",
        "adjusted_edge_jaccard",
        "division_jaccard",
    }
    present = derived_names.intersection(data)
    if present:
        raise EvidenceValidationError(
            f"build_per_clip_evidence derives metric fields; remove supplied {sorted(present)}"
        )
    required = PerClipEvidence._KEYS - derived_names
    _require_exact_keys(data, required, "raw_per_clip_record")
    counts = _derived_metrics(
        predicted_nodes=_require_int(data["predicted_nodes"], "predicted_nodes"),
        estimated_nodes=_require_int(data["estimated_nodes"], "estimated_nodes"),
        annotated_nodes=_require_int(data["annotated_nodes"], "annotated_nodes"),
        matched_annotated_nodes=_require_int(
            data["matched_annotated_nodes"], "matched_annotated_nodes"
        ),
        edge_tp=_require_int(data["edge_tp"], "edge_tp"),
        edge_fp=_require_int(data["edge_fp"], "edge_fp"),
        edge_fn=_require_int(data["edge_fn"], "edge_fn"),
        division_tp=_require_int(data["division_tp"], "division_tp"),
        division_fp=_require_int(data["division_fp"], "division_fp"),
        division_fn=_require_int(data["division_fn"], "division_fn"),
    )
    data.update({name: metric.to_dict() for name, metric in counts.items()})
    return PerClipEvidence.from_dict(data)


def aggregate_evidence(
    records: Iterable[PerClipEvidence | Mapping[str, Any]],
    planned_clip_ids_by_direction: Mapping[str, Sequence[str]],
    *,
    status: EvaluationStatus | str,
    artifact_root: str | Path | None = None,
    verify_artifacts: bool | None = None,
) -> dict[str, Any]:
    """Validate coverage and recompute official and diagnostic aggregates.

    A ``COMPLETE`` result requires exact equality between the planned and
    evaluated clip sets in every direction and verified graph artifact hashes.
    ``PARTIAL`` permits missing planned clips but never duplicates, unexpected
    clips, incomplete per-clip records, or malformed measurements.
    """

    parsed_status = _parse_enum(EvaluationStatus, status, "status")
    plan = _normalize_plan(planned_clip_ids_by_direction)
    parsed = [
        PerClipEvidence.from_dict(r.to_dict())
        if isinstance(r, PerClipEvidence)
        else PerClipEvidence.from_dict(r)
        for r in records
    ]
    if not parsed:
        raise EvidenceValidationError("at least one evaluated per-clip record is required")
    for record in parsed:
        record.validate()
        if not record.evaluation_complete:
            raise EvidenceValidationError(
                f"{record.clip_id}: incomplete evaluation cannot be an evaluated record"
            )
    by_clip: dict[str, PerClipEvidence] = {}
    for record in parsed:
        if record.clip_id in by_clip:
            raise EvidenceValidationError(f"duplicate evaluated clip_id {record.clip_id}")
        by_clip[record.clip_id] = record

    planned_direction_for = {
        clip_id: direction for direction, clip_ids in plan.items() for clip_id in clip_ids
    }
    unexpected = sorted(set(by_clip) - set(planned_direction_for))
    if unexpected:
        raise EvidenceValidationError(f"evaluated clip IDs are outside the plan: {unexpected}")
    wrong_direction = sorted(
        record.clip_id
        for record in parsed
        if planned_direction_for[record.clip_id] != record.direction
    )
    if wrong_direction:
        raise EvidenceValidationError(
            f"records use a direction different from the plan: {wrong_direction}"
        )

    evaluated_by_direction = {
        direction: sorted(r.clip_id for r in parsed if r.direction == direction)
        for direction in plan
    }
    missing_by_direction = {
        direction: sorted(set(plan[direction]) - set(evaluated_by_direction[direction]))
        for direction in plan
    }
    missing = sorted(clip for clips in missing_by_direction.values() for clip in clips)
    if parsed_status is EvaluationStatus.COMPLETE and missing:
        raise EvidenceValidationError(
            f"COMPLETE requires exact planned/evaluated clip equality; missing clip IDs: {missing}"
        )

    candidate_ids = {r.candidate_id for r in parsed}
    candidate_hashes = {r.provenance.candidate_sha256 for r in parsed}
    if len(candidate_ids) != 1 or len(candidate_hashes) != 1:
        raise EvidenceValidationError(
            "one aggregate must contain exactly one candidate_id and candidate_sha256"
        )
    _validate_identity_within_directions(parsed)

    if verify_artifacts is None:
        verify_artifacts = parsed_status is EvaluationStatus.COMPLETE
    if parsed_status is EvaluationStatus.COMPLETE and not verify_artifacts:
        raise EvidenceValidationError("COMPLETE requires graph artifact hash verification")
    verified: list[dict[str, str]] = []
    if verify_artifacts:
        for record in parsed:
            resolved = _resolve_artifact_path(record.graph_path, artifact_root)
            digest = verify_artifact_sha256(resolved, record.graph_sha256)
            verified.append({"clip_id": record.clip_id, "path": str(resolved), "sha256": digest})

    directions: dict[str, Any] = {}
    for direction, planned_ids in plan.items():
        direction_records = [r for r in parsed if r.direction == direction]
        directions[direction] = {
            "planned_clip_ids": list(planned_ids),
            "evaluated_clip_ids": evaluated_by_direction[direction],
            "missing_clip_ids": missing_by_direction[direction],
            "coverage_complete": not missing_by_direction[direction],
            "provenance": _direction_identity(direction_records),
            "aggregate": _aggregate_group(direction_records),
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": parsed_status.value,
        "candidate_id": next(iter(candidate_ids)),
        "candidate_sha256": next(iter(candidate_hashes)),
        "run_ids": sorted({r.run_id for r in parsed}),
        "planned_clip_ids_by_direction": {k: list(v) for k, v in plan.items()},
        "evaluated_clip_ids_by_direction": evaluated_by_direction,
        "planned_clip_ids": sorted(planned_direction_for),
        "evaluated_clip_ids": sorted(by_clip),
        "missing_clip_ids": missing,
        "unexpected_clip_ids": [],
        "coverage_complete": not missing,
        "evidence_classes": sorted({r.evidence_class.value for r in parsed}),
        "official_scorer": {
            "commit": PINNED_SCORER_COMMIT,
            "normalized_source_sha256": PINNED_SCORER_SHA256,
            "aggregation": "tracking_cellmot.metrics.summarise",
        },
        "artifact_verification": {
            "required": parsed_status is EvaluationStatus.COMPLETE,
            "performed": bool(verify_artifacts),
            "verified_count": len(verified),
            "artifacts": verified,
        },
        "overall": _aggregate_group(parsed),
        "directions": directions,
        "record_sha256_by_clip": {
            record.clip_id: canonical_sha256(record.to_dict()) for record in parsed
        },
    }
    strict_json_dumps(result)
    return result


def compare_paired_evidence(
    control_records: Iterable[PerClipEvidence | Mapping[str, Any]],
    candidate_records: Iterable[PerClipEvidence | Mapping[str, Any]],
    planned_clip_ids_by_direction: Mapping[str, Sequence[str]],
    *,
    status: EvaluationStatus | str = EvaluationStatus.COMPLETE,
    control_artifact_root: str | Path | None = None,
    candidate_artifact_root: str | Path | None = None,
    verify_artifacts: bool | None = None,
) -> dict[str, Any]:
    """Reject unpaired results and compare separately aggregated candidates.

    Model/source/config identities may differ between control and candidate.
    Each side must be internally bound to its own identity, while data, split,
    scorer, coverage, frame population, evidence class, and training membership
    must match clip by clip.  Each candidate's adjusted-edge weights are always
    derived from that candidate's own TP/FP/FN counts.
    """

    control = [
        PerClipEvidence.from_dict(r.to_dict())
        if isinstance(r, PerClipEvidence)
        else PerClipEvidence.from_dict(r)
        for r in control_records
    ]
    candidate = [
        PerClipEvidence.from_dict(r.to_dict())
        if isinstance(r, PerClipEvidence)
        else PerClipEvidence.from_dict(r)
        for r in candidate_records
    ]
    control_by_id = _unique_record_map(control, "control")
    candidate_by_id = _unique_record_map(candidate, "candidate")
    if set(control_by_id) != set(candidate_by_id):
        missing_in_candidate = sorted(set(control_by_id) - set(candidate_by_id))
        missing_in_control = sorted(set(candidate_by_id) - set(control_by_id))
        raise EvidenceValidationError(
            "paired coverage differs; "
            f"missing in candidate={missing_in_candidate}, missing in control={missing_in_control}"
        )
    problems: list[str] = []
    for clip_id in sorted(control_by_id):
        left = control_by_id[clip_id]
        right = candidate_by_id[clip_id]
        comparable = {
            "embryo": (left.embryo, right.embryo),
            "direction": (left.direction, right.direction),
            "frame_ids": (left.frame_ids, right.frame_ids),
            "estimated_nodes": (left.estimated_nodes, right.estimated_nodes),
            "annotated_nodes": (left.annotated_nodes, right.annotated_nodes),
            "input_manifest_sha256": (
                left.provenance.input_manifest_sha256,
                right.provenance.input_manifest_sha256,
            ),
            "split_manifest_sha256": (
                left.provenance.split_manifest_sha256,
                right.provenance.split_manifest_sha256,
            ),
            "scorer_commit": (
                left.provenance.scorer_commit,
                right.provenance.scorer_commit,
            ),
            "scorer_sha256": (
                left.provenance.scorer_sha256,
                right.provenance.scorer_sha256,
            ),
            "evidence_class": (left.evidence_class, right.evidence_class),
            "training_membership": (
                left.provenance.training_membership,
                right.provenance.training_membership,
            ),
            "training_ids": (
                left.provenance.training_ids,
                right.provenance.training_ids,
            ),
            "development_ids": (
                left.provenance.development_ids,
                right.provenance.development_ids,
            ),
        }
        for field, (left_value, right_value) in comparable.items():
            if left_value != right_value:
                problems.append(f"{clip_id}.{field}")
    if problems:
        raise EvidenceValidationError(
            "paired evidence is not comparable; mismatched fields: " + ", ".join(problems)
        )

    left_candidate_ids = {r.candidate_id for r in control}
    right_candidate_ids = {r.candidate_id for r in candidate}
    if left_candidate_ids == right_candidate_ids:
        raise EvidenceValidationError(
            "control and candidate must have distinct candidate_id values"
        )

    control_aggregate = aggregate_evidence(
        control,
        planned_clip_ids_by_direction,
        status=status,
        artifact_root=control_artifact_root,
        verify_artifacts=verify_artifacts,
    )
    candidate_aggregate = aggregate_evidence(
        candidate,
        planned_clip_ids_by_direction,
        status=status,
        artifact_root=candidate_artifact_root,
        verify_artifacts=verify_artifacts,
    )
    metric_deltas = {}
    for name, path in {
        "adjusted_edge_jaccard": ("official", "adjusted_edge_jaccard"),
        "raw_edge_jaccard": ("official", "raw_edge_jaccard"),
        "division_jaccard": ("official", "division_jaccard"),
        "score": ("official", "score"),
        "mean_annotated_recall": ("recall", "mean_per_clip"),
        "pooled_annotated_recall": ("recall", "pooled"),
    }.items():
        left_metric = MetricValue.from_dict(
            control_aggregate["overall"][path[0]][path[1]], field=f"control.{name}"
        )
        right_metric = MetricValue.from_dict(
            candidate_aggregate["overall"][path[0]][path[1]], field=f"candidate.{name}"
        )
        if left_metric.value is None or right_metric.value is None:
            delta = MetricValue.undefined(f"{name}_undefined_for_one_or_both_candidates")
        else:
            delta = MetricValue.defined(right_metric.value - left_metric.value)
        metric_deltas[name] = delta.to_dict()
    result = {
        "schema_version": SCHEMA_VERSION,
        "paired_clip_ids": sorted(control_by_id),
        "planned_clip_ids_by_direction": {
            k: list(v) for k, v in _normalize_plan(planned_clip_ids_by_direction).items()
        },
        "comparability": "PASS",
        "control": control_aggregate,
        "candidate": candidate_aggregate,
        "candidate_minus_control": metric_deltas,
        "adjusted_weight_policy": ("each candidate uses its own per-clip edge union TP+FP+FN"),
    }
    strict_json_dumps(result)
    return result


def audit_evidence_bundle(
    raw: Mapping[str, Any], *, artifact_root: str | Path | None = None
) -> dict[str, Any]:
    """Audit a bundle and optionally verify its claimed aggregate."""

    data = _require_mapping(raw, "evidence_bundle")
    allowed = {
        "schema_version",
        "status",
        "planned_clip_ids_by_direction",
        "records",
        "aggregate",
    }
    required = allowed - {"aggregate"}
    missing = required - set(data)
    extra = set(data) - allowed
    if missing or extra:
        raise EvidenceValidationError(
            f"evidence_bundle schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if _require_int(data["schema_version"], "schema_version", minimum=1) != SCHEMA_VERSION:
        raise EvidenceValidationError("unsupported evidence bundle schema_version")
    records = _require_sequence(data["records"], "records")
    result = aggregate_evidence(
        records,
        _require_mapping(data["planned_clip_ids_by_direction"], "planned_clip_ids_by_direction"),
        status=data["status"],
        artifact_root=artifact_root,
        verify_artifacts=artifact_root is not None or data["status"] == "COMPLETE",
    )
    if "aggregate" in data:
        claimed = data["aggregate"]
        if canonical_sha256(claimed) != canonical_sha256(result):
            raise EvidenceValidationError(
                "claimed aggregate differs from recomputation over raw per-clip records"
            )
    return result


def audit_historical_outer_progress(
    outer_progress_path: str | Path,
    split_manifest_path: str | Path,
    *,
    fold: str = "fold0",
) -> dict[str, Any]:
    """Audit the legacy paired outer report without upgrading its evidence class."""

    outer_path = Path(outer_progress_path)
    split_path = Path(split_manifest_path)
    raw = load_json(outer_path)
    split_manifest = load_json(split_path)
    outer = _require_mapping(raw, "historical_outer_progress")
    _require_exact_keys(outer, {"classical", "complete", "learned"}, "historical_outer_progress")
    if fold not in split_manifest:
        raise EvidenceValidationError(f"split manifest does not contain {fold}")
    fold_data = _require_mapping(split_manifest[fold], f"split_manifest.{fold}")
    if "train" not in fold_data or "val" not in fold_data:
        raise EvidenceValidationError(f"split_manifest.{fold} requires train and val")
    training_ids = list(_string_tuple(fold_data["train"], f"split_manifest.{fold}.train"))
    planned_ids = list(_string_tuple(fold_data["val"], f"split_manifest.{fold}.val"))
    training_embryos = sorted({_embryo_from_clip(v) for v in training_ids})
    evaluation_embryos = sorted({_embryo_from_clip(v) for v in planned_ids})
    if len(training_embryos) != 1 or len(evaluation_embryos) != 1:
        raise EvidenceValidationError(
            "historical whole-embryo audit requires one training and one evaluation embryo"
        )
    direction = f"fit_{training_embryos[0]}_eval_{evaluation_embryos[0]}"
    summaries: dict[str, Any] = {}
    ids_by_candidate: dict[str, list[str]] = {}
    for candidate_name in ("classical", "learned"):
        rows = _require_sequence(outer[candidate_name], f"historical.{candidate_name}")
        summary, clip_ids = _aggregate_historical_rows(rows, candidate_name)
        summaries[candidate_name] = summary
        ids_by_candidate[candidate_name] = clip_ids
    pair_missing = {
        "missing_in_learned": sorted(
            set(ids_by_candidate["classical"]) - set(ids_by_candidate["learned"])
        ),
        "missing_in_classical": sorted(
            set(ids_by_candidate["learned"]) - set(ids_by_candidate["classical"])
        ),
    }
    evaluated_ids = sorted(
        set(ids_by_candidate["classical"]).intersection(ids_by_candidate["learned"])
    )
    unexpected = sorted(set(evaluated_ids) - set(planned_ids))
    if unexpected:
        raise EvidenceValidationError(f"historical report contains unexpected IDs: {unexpected}")
    missing = sorted(set(planned_ids) - set(evaluated_ids))
    diagnostic_gaps = [
        "estimated_nodes",
        "annotated_nodes",
        "matched_annotated_nodes",
        "localization_um_quantiles",
        "pre_cap_candidates",
        "total_frames",
        "predicted_forks_after_solver",
        "predicted_forks_after_postprocess",
        "runtime_seconds_by_stage",
        "peak_ram_bytes",
        "peak_vram_bytes",
        "split_and_model_identity_bound_per_clip",
        "graph_artifact_paths_for_hash_readback",
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": EvaluationStatus.PARTIAL.value,
        "evidence_class": EvidenceClass.PREVIOUSLY_INSPECTED_DIAGNOSTIC.value,
        "new_contract_eligible": False,
        "historical_complete_flag": _require_bool(outer["complete"], "historical.complete"),
        "direction": direction,
        "planned_clip_ids": planned_ids,
        "evaluated_clip_ids": evaluated_ids,
        "missing_clip_ids": missing,
        "planned_clip_count": len(planned_ids),
        "evaluated_clip_count": len(evaluated_ids),
        "pair_coverage": {
            **pair_missing,
            "comparable_on_shared_raw_fields": not any(pair_missing.values()),
        },
        "original_dataset_membership": {
            "split_fold": fold,
            "training_embryos": training_embryos,
            "evaluation_embryos": evaluation_embryos,
            "training_ids": training_ids,
            "planned_evaluation_ids": planned_ids,
            "learned_candidate_membership": (
                "trained_on_original_training_ids; evaluated IDs excluded by the split manifest"
            ),
            "classical_candidate_membership": "not_applicable_nonlearned_comparator",
            "checkpoint_to_split_binding": (
                "reported historically; not independently re-established by this JSON audit"
            ),
            "split_manifest_path": str(split_path),
            "split_manifest_sha256": sha256_file(split_path),
        },
        "official_scorer": {
            "commit": PINNED_SCORER_COMMIT,
            "normalized_source_sha256": PINNED_SCORER_SHA256,
            "aggregation": "recomputed from legacy raw count and adjusted-score fields",
        },
        "candidates": summaries,
        "missing_new_contract_diagnostics": diagnostic_gaps,
        "claims_not_supported": [
            "COMPLETE",
            "two_direction_evaluation",
            "out_of_fold_evaluation",
            "locked_evaluation_after_prior_inspection",
            "full_new_validation_contract",
        ],
        "source_artifact": {
            "path": str(outer_path),
            "sha256": sha256_file(outer_path),
        },
    }
    strict_json_dumps(result)
    return result


def verify_artifact_sha256(path: str | Path, expected_sha256: str) -> str:
    """Read an artifact and fail if its current bytes differ from the manifest."""

    expected = _require_sha256(expected_sha256, "expected_sha256")
    artifact = Path(path)
    if not artifact.is_file():
        raise EvidenceValidationError(f"artifact does not exist as a file: {artifact}")
    actual = sha256_file(artifact)
    if actual != expected:
        raise EvidenceValidationError(
            f"artifact SHA256 mismatch for {artifact}: expected {expected}, got {actual}"
        )
    return actual


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def strict_json_dumps(value: Any, *, indent: int | None = None, sort_keys: bool = True) -> str:
    """Serialize strict JSON after rejecting non-finite numbers recursively."""

    _reject_nonfinite(value, "$")
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
        separators=(",", ":") if indent is None else None,
    )


def load_json(path: str | Path) -> Any:
    """Load strict JSON, rejecting duplicate keys and NaN/Infinity tokens."""

    def reject_constant(token: str) -> None:
        raise EvidenceValidationError(f"non-finite JSON token is forbidden: {token}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceValidationError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise EvidenceValidationError(f"invalid JSON in {path}: {exc}") from exc


def _derived_metrics(**counts: int) -> dict[str, MetricValue]:
    predicted = counts["predicted_nodes"]
    estimated = counts["estimated_nodes"]
    annotated = counts["annotated_nodes"]
    matched = counts["matched_annotated_nodes"]
    edge_union = counts["edge_tp"] + counts["edge_fp"] + counts["edge_fn"]
    division_union = counts["division_tp"] + counts["division_fp"] + counts["division_fn"]
    recall = (
        MetricValue.defined(matched / annotated)
        if annotated > 0
        else MetricValue.undefined(_NO_ANNOTATED_NODES)
    )
    ratio = (
        MetricValue.defined(predicted / estimated)
        if estimated > 0
        else MetricValue.undefined(_NO_ESTIMATED_NODES)
    )
    edge = (
        MetricValue.defined(counts["edge_tp"] / edge_union)
        if edge_union > 0
        else MetricValue.undefined(_NO_EDGE_UNION)
    )
    if edge.value is None:
        adjusted = MetricValue.undefined(_NO_EDGE_UNION)
    elif estimated <= 0:
        adjusted = MetricValue.undefined(_NO_ESTIMATED_NODES)
    else:
        total_node_ratio = (predicted - estimated) / estimated
        adjusted = MetricValue.defined(
            max(0.0, edge.value * (1.0 - ADJUSTMENT_ALPHA * total_node_ratio))
        )
    division = (
        MetricValue.defined(counts["division_tp"] / division_union)
        if division_union > 0
        else MetricValue.undefined(_NO_DIVISION_UNION)
    )
    return {
        "annotated_recall": recall,
        "predicted_to_estimated_ratio": ratio,
        "raw_edge_jaccard": edge,
        "adjusted_edge_jaccard": adjusted,
        "division_jaccard": division,
    }


def _aggregate_group(records: Sequence[PerClipEvidence]) -> dict[str, Any]:
    edge_tp = sum(r.edge_tp for r in records)
    edge_fp = sum(r.edge_fp for r in records)
    edge_fn = sum(r.edge_fn for r in records)
    edge_union = edge_tp + edge_fp + edge_fn
    raw_edge = (
        MetricValue.defined(edge_tp / edge_union)
        if edge_union > 0
        else MetricValue.undefined(_NO_EDGE_UNION)
    )
    adjusted_rows = [r for r in records if r.adjusted_edge_jaccard.value is not None]
    adjusted_weight = sum(r.edge_union for r in adjusted_rows)
    adjusted = (
        MetricValue.defined(
            sum(r.edge_union * float(r.adjusted_edge_jaccard.value) for r in adjusted_rows)
            / adjusted_weight
        )
        if adjusted_weight > 0
        else MetricValue.undefined("no_positive_edge_union_with_defined_adjustment")
    )

    division_all = _division_aggregate(records, subset_definition="all_evaluated_clips")
    division_bearing_records = [r for r in records if r.division_tp + r.division_fn > 0]
    division_bearing = _division_aggregate(
        division_bearing_records,
        subset_definition="ground_truth_division_bearing_clips",
    )
    division_metric = MetricValue.from_dict(
        division_all["division_jaccard"], field="division_all.division_jaccard"
    )
    if adjusted.value is None:
        score = MetricValue.undefined("adjusted_edge_jaccard_undefined")
    elif division_metric.value is None:
        score = MetricValue.defined(adjusted.value)
    else:
        score = MetricValue.defined(adjusted.value + DIVISION_SCORE_WEIGHT * division_metric.value)

    recall_values = [r.annotated_recall.value for r in records]
    if records and all(value is not None for value in recall_values):
        mean_recall = MetricValue.defined(
            sum(float(value) for value in recall_values) / len(records)
        )
    elif records:
        mean_recall = MetricValue.undefined("one_or_more_clips_have_zero_annotated_nodes")
    else:
        mean_recall = MetricValue.undefined("no_evaluated_clips")
    total_annotated = sum(r.annotated_nodes for r in records)
    total_matched = sum(r.matched_annotated_nodes for r in records)
    pooled_recall = (
        MetricValue.defined(total_matched / total_annotated)
        if total_annotated > 0
        else MetricValue.undefined(_NO_ANNOTATED_NODES)
    )

    ratio_records = [r for r in records if r.predicted_to_estimated_ratio.value is not None]
    ratio_values = [float(r.predicted_to_estimated_ratio.value) for r in ratio_records]
    count_ratios = {
        "definition": "predicted_nodes / estimated_nodes per clip",
        "official_use": "per-clip count penalty only; aggregate summaries are diagnostic",
        "defined_clip_count": len(ratio_records),
        "undefined_clip_count": len(records) - len(ratio_records),
        "mean": _mean_metric(ratio_values, "no_positive_estimated_node_counts").to_dict(),
        "min": _order_metric(ratio_values, 0.0, "no_positive_estimated_node_counts").to_dict(),
        "p50": _order_metric(ratio_values, 0.5, "no_positive_estimated_node_counts").to_dict(),
        "p90": _order_metric(ratio_values, 0.9, "no_positive_estimated_node_counts").to_dict(),
        "p95": _order_metric(ratio_values, 0.95, "no_positive_estimated_node_counts").to_dict(),
        "max": _order_metric(ratio_values, 1.0, "no_positive_estimated_node_counts").to_dict(),
        "pooled_predicted_nodes": sum(r.predicted_nodes for r in ratio_records),
        "pooled_estimated_nodes_denominator": sum(r.estimated_nodes for r in ratio_records),
        "pooled_ratio": (
            MetricValue.defined(
                sum(r.predicted_nodes for r in ratio_records)
                / sum(r.estimated_nodes for r in ratio_records)
            )
            if sum(r.estimated_nodes for r in ratio_records) > 0
            else MetricValue.undefined("no_positive_estimated_node_counts")
        ).to_dict(),
        "values_by_clip": {r.clip_id: r.predicted_to_estimated_ratio.to_dict() for r in records},
        "estimated_node_denominator_by_clip": {r.clip_id: r.estimated_nodes for r in records},
    }

    total_frames = sum(r.total_frames for r in records)
    capped_frames = sum(r.capped_frames for r in records)
    capped_clips = sum(r.capped_frames > 0 for r in records)
    stage_names = sorted({name for r in records for name in r.runtime_seconds_by_stage})
    return {
        "evaluated_clip_count": len(records),
        "official": {
            "aggregation_source": "pinned tracking_cellmot.metrics.summarise conventions",
            "edge_tp": edge_tp,
            "edge_fp": edge_fp,
            "edge_fn": edge_fn,
            "edge_union_weight": edge_union,
            "raw_edge_jaccard": raw_edge.to_dict(),
            "adjusted_edge_jaccard": adjusted.to_dict(),
            "adjusted_edge_weight_denominator": adjusted_weight,
            "adjusted_defined_clip_count": len(adjusted_rows),
            "division_tp": division_all["division_tp"],
            "division_fp": division_all["division_fp"],
            "division_fn": division_all["division_fn"],
            "division_jaccard": division_all["division_jaccard"],
            "division_term_applied": division_metric.value is not None,
            "score": score.to_dict(),
        },
        "recall": {
            "mean_per_clip": mean_recall.to_dict(),
            "mean_per_clip_denominator_clips": len(records),
            "pooled": pooled_recall.to_dict(),
            "pooled_numerator_matched_annotated_nodes": total_matched,
            "pooled_denominator_annotated_nodes": total_annotated,
        },
        "count_ratio_distribution": count_ratios,
        "divisions": {
            "all_clips": division_all,
            "division_bearing_subset": division_bearing,
        },
        "detection_and_operations": {
            "predicted_nodes_total": sum(r.predicted_nodes for r in records),
            "pre_cap_candidates_total": sum(r.pre_cap_candidates for r in records),
            "capped_frames": capped_frames,
            "capped_frames_denominator_total_frames": total_frames,
            "capped_frame_fraction": (
                MetricValue.defined(capped_frames / total_frames)
                if total_frames > 0
                else MetricValue.undefined("total_frames_zero")
            ).to_dict(),
            "capped_clips": capped_clips,
            "capped_clips_denominator_evaluated_clips": len(records),
            "capped_clip_fraction": (
                MetricValue.defined(capped_clips / len(records))
                if records
                else MetricValue.undefined("no_evaluated_clips")
            ).to_dict(),
            "predicted_forks_after_solver": sum(r.predicted_forks_after_solver for r in records),
            "predicted_forks_after_postprocess": sum(
                r.predicted_forks_after_postprocess for r in records
            ),
            "runtime_seconds_by_stage": {
                name: sum(r.runtime_seconds_by_stage.get(name, 0.0) for r in records)
                for name in stage_names
            },
            "peak_ram_bytes_max": max((r.peak_ram_bytes for r in records), default=0),
            "peak_vram_bytes_max": max((r.peak_vram_bytes for r in records), default=0),
            "fallback_count": sum(r.fallback_count for r in records),
            "localization_quantiles": {
                "aggregation": "per_clip_only; quantiles cannot be pooled from quantiles",
                "values_by_clip": {
                    r.clip_id: {
                        key: r.localization_um_quantiles[key].to_dict()
                        for key in ("p50", "p90", "p95")
                    }
                    for r in records
                },
            },
        },
    }


def _division_aggregate(
    records: Sequence[PerClipEvidence], *, subset_definition: str
) -> dict[str, Any]:
    tp = sum(r.division_tp for r in records)
    fp = sum(r.division_fp for r in records)
    fn = sum(r.division_fn for r in records)
    union = tp + fp + fn
    metric = (
        MetricValue.defined(tp / union) if union > 0 else MetricValue.undefined(_NO_DIVISION_UNION)
    )
    return {
        "subset_definition": subset_definition,
        "clip_ids": sorted(r.clip_id for r in records),
        "clip_count": len(records),
        "division_tp": tp,
        "division_fp": fp,
        "division_fn": fn,
        "division_union_denominator": union,
        "division_jaccard": metric.to_dict(),
    }


def _aggregate_historical_rows(
    rows: Sequence[Any], candidate_name: str
) -> tuple[dict[str, Any], list[str]]:
    required = {
        "dataset",
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
        "node_recall",
        "node_count_ratio",
        "total_node_ratio",
        "edge_jaccard",
        "adj_edge_jaccard",
        "diagnostics",
    }
    parsed: list[Mapping[str, Any]] = []
    clip_ids: list[str] = []
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"historical.{candidate_name}[{index}]")
        missing = required - set(row)
        if missing:
            raise EvidenceValidationError(
                f"historical.{candidate_name}[{index}] lacks raw fields {sorted(missing)}"
            )
        clip_id = _require_nonempty_str(row["dataset"], f"{candidate_name}[{index}].dataset")
        if clip_id in clip_ids:
            raise EvidenceValidationError(f"duplicate historical clip ID {clip_id}")
        clip_ids.append(clip_id)
        counts = {
            key: _require_int(row[key], f"{candidate_name}.{clip_id}.{key}")
            for key in (
                "edge_tp",
                "edge_fp",
                "edge_fn",
                "division_tp",
                "division_fp",
                "division_fn",
            )
        }
        edge_union = counts["edge_tp"] + counts["edge_fp"] + counts["edge_fn"]
        raw_edge = _require_finite_float(
            row["edge_jaccard"], f"{candidate_name}.{clip_id}.edge_jaccard"
        )
        if edge_union <= 0 or not math.isclose(
            raw_edge, counts["edge_tp"] / edge_union, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise EvidenceValidationError(
                f"{candidate_name}.{clip_id}: edge_jaccard disagrees with raw counts"
            )
        node_ratio = _require_finite_float(
            row["node_count_ratio"], f"{candidate_name}.{clip_id}.node_count_ratio"
        )
        total_node_ratio = _require_finite_float(
            row["total_node_ratio"], f"{candidate_name}.{clip_id}.total_node_ratio"
        )
        if not math.isclose(node_ratio, total_node_ratio + 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise EvidenceValidationError(
                f"{candidate_name}.{clip_id}: node ratio fields are inconsistent"
            )
        expected_adjusted = max(0.0, raw_edge * (1.0 - ADJUSTMENT_ALPHA * total_node_ratio))
        adjusted = _require_finite_float(
            row["adj_edge_jaccard"], f"{candidate_name}.{clip_id}.adj_edge_jaccard"
        )
        if not math.isclose(adjusted, expected_adjusted, rel_tol=1e-12, abs_tol=1e-12):
            raise EvidenceValidationError(
                f"{candidate_name}.{clip_id}: adjusted edge score disagrees with pinned formula"
            )
        _require_finite_float(row["node_recall"], f"{candidate_name}.{clip_id}.node_recall")
        parsed.append(row)
    totals = {
        key: sum(int(row[key]) for row in parsed)
        for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
    }
    edge_union = totals["edge_tp"] + totals["edge_fp"] + totals["edge_fn"]
    own_weights = [int(r["edge_tp"]) + int(r["edge_fp"]) + int(r["edge_fn"]) for r in parsed]
    weight_total = sum(own_weights)
    adjusted = (
        MetricValue.defined(
            sum(w * float(r["adj_edge_jaccard"]) for w, r in zip(own_weights, parsed))
            / weight_total
        )
        if weight_total > 0
        else MetricValue.undefined(_NO_EDGE_UNION)
    )
    division_union = totals["division_tp"] + totals["division_fp"] + totals["division_fn"]
    division = (
        MetricValue.defined(totals["division_tp"] / division_union)
        if division_union > 0
        else MetricValue.undefined(_NO_DIVISION_UNION)
    )
    score = (
        MetricValue.undefined("adjusted_edge_jaccard_undefined")
        if adjusted.value is None
        else MetricValue.defined(
            adjusted.value
            + (DIVISION_SCORE_WEIGHT * division.value if division.value is not None else 0.0)
        )
    )
    capped_clip_count = sum(
        bool(_require_mapping(r["diagnostics"], "historical.diagnostics").get("capped_frames"))
        for r in parsed
    )
    bearing_rows = [r for r in parsed if int(r["division_tp"]) + int(r["division_fn"]) > 0]
    bearing_counts = {
        key: sum(int(row[key]) for row in bearing_rows)
        for key in ("division_tp", "division_fp", "division_fn")
    }
    bearing_union = sum(bearing_counts.values())
    bearing_division = (
        MetricValue.defined(bearing_counts["division_tp"] / bearing_union)
        if bearing_union > 0
        else MetricValue.undefined(_NO_DIVISION_UNION)
    )
    count_ratio_values = [float(r["node_count_ratio"]) for r in parsed]
    return (
        {
            "evaluated_clip_count": len(parsed),
            "edge_tp": totals["edge_tp"],
            "edge_fp": totals["edge_fp"],
            "edge_fn": totals["edge_fn"],
            "raw_edge_jaccard": (
                MetricValue.defined(totals["edge_tp"] / edge_union)
                if edge_union > 0
                else MetricValue.undefined(_NO_EDGE_UNION)
            ).to_dict(),
            "adjusted_edge_jaccard": adjusted.to_dict(),
            "adjusted_edge_weight_denominator_own_edge_union": weight_total,
            "mean_per_clip_annotated_recall": _mean_metric(
                [float(r["node_recall"]) for r in parsed], "no_evaluated_clips"
            ).to_dict(),
            "mean_per_clip_annotated_recall_denominator_clips": len(parsed),
            "pooled_annotated_recall": MetricValue.undefined(
                "legacy_rows_lack_matched_and_annotated_node_counts"
            ).to_dict(),
            "mean_per_clip_predicted_to_estimated_ratio": _mean_metric(
                count_ratio_values, "no_evaluated_clips"
            ).to_dict(),
            "mean_count_ratio_denominator_clips": len(parsed),
            "count_ratio_distribution": {
                "definition": "legacy predicted_nodes / estimated_nodes per clip",
                "denominator_clips": len(parsed),
                "min": _order_metric(count_ratio_values, 0.0, "no_evaluated_clips").to_dict(),
                "p50": _order_metric(count_ratio_values, 0.5, "no_evaluated_clips").to_dict(),
                "p90": _order_metric(count_ratio_values, 0.9, "no_evaluated_clips").to_dict(),
                "p95": _order_metric(count_ratio_values, 0.95, "no_evaluated_clips").to_dict(),
                "max": _order_metric(count_ratio_values, 1.0, "no_evaluated_clips").to_dict(),
                "pooled_ratio": MetricValue.undefined(
                    "legacy_rows_lack_predicted_and_estimated_count_denominators"
                ).to_dict(),
            },
            "division_tp": totals["division_tp"],
            "division_fp": totals["division_fp"],
            "division_fn": totals["division_fn"],
            "division_jaccard_all_clips": division.to_dict(),
            "division_bearing_subset": {
                "subset_definition": "ground_truth_division_bearing_clips",
                "clip_ids": sorted(str(r["dataset"]) for r in bearing_rows),
                "clip_count": len(bearing_rows),
                **bearing_counts,
                "division_union_denominator": bearing_union,
                "division_jaccard": bearing_division.to_dict(),
            },
            "score": score.to_dict(),
            "capped_clip_count": capped_clip_count,
            "capped_frame_fraction": MetricValue.undefined(
                "legacy rows lack total_frames denominators"
            ).to_dict(),
        },
        clip_ids,
    )


def _validate_identity_within_directions(records: Sequence[PerClipEvidence]) -> None:
    for direction in sorted({r.direction for r in records}):
        direction_records = [r for r in records if r.direction == direction]
        identities = {
            (
                r.run_id,
                r.candidate_id,
                r.provenance.candidate_sha256,
                r.provenance.git_commit,
                r.provenance.source_bundle_sha256,
                r.provenance.dependency_manifest_sha256,
                r.provenance.effective_config_sha256,
                r.provenance.model_weight_sha256,
                r.provenance.model_weight_sha256_reason,
                r.provenance.input_manifest_sha256,
                r.provenance.split_manifest_sha256,
                r.provenance.scorer_commit,
                r.provenance.scorer_sha256,
                r.provenance.training_membership,
                r.provenance.training_ids,
                r.provenance.development_ids,
                r.provenance.used_for_development_selection,
                r.provenance.previously_inspected,
                r.evidence_class,
            )
            for r in direction_records
        }
        if len(identities) != 1:
            raise EvidenceValidationError(
                f"{direction}: per-clip candidate/provenance identity is not uniform"
            )


def _direction_identity(records: Sequence[PerClipEvidence]) -> dict[str, Any] | None:
    if not records:
        return None
    record = records[0]
    return {
        "run_id": record.run_id,
        "candidate_id": record.candidate_id,
        "evidence_class": record.evidence_class.value,
        **record.provenance.to_dict(),
    }


def _unique_record_map(
    records: Sequence[PerClipEvidence], label: str
) -> dict[str, PerClipEvidence]:
    result: dict[str, PerClipEvidence] = {}
    for record in records:
        record.validate()
        if record.clip_id in result:
            raise EvidenceValidationError(f"duplicate {label} clip_id {record.clip_id}")
        result[record.clip_id] = record
    return result


def _normalize_plan(raw: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    plan = _require_mapping(raw, "planned_clip_ids_by_direction")
    if not plan:
        raise EvidenceValidationError("planned_clip_ids_by_direction cannot be empty")
    normalized: dict[str, tuple[str, ...]] = {}
    all_ids: list[str] = []
    for raw_direction, raw_ids in plan.items():
        direction = _require_nonempty_str(raw_direction, "planned direction")
        clip_ids = _string_tuple(raw_ids, f"planned.{direction}")
        if not clip_ids:
            raise EvidenceValidationError(f"planned direction {direction} cannot be empty")
        normalized[direction] = tuple(sorted(clip_ids))
        all_ids.extend(clip_ids)
    if len(all_ids) != len(set(all_ids)):
        duplicates = sorted({v for v in all_ids if all_ids.count(v) > 1})
        raise EvidenceValidationError(f"duplicate planned clip IDs: {duplicates}")
    return dict(sorted(normalized.items()))


def _resolve_artifact_path(path: str, artifact_root: str | Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    if artifact_root is None:
        raise EvidenceValidationError(
            f"relative graph_path requires artifact_root for verification: {path}"
        )
    root = Path(artifact_root).resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvidenceValidationError(f"artifact path escapes artifact_root: {path}") from exc
    return resolved


def _mean_metric(values: Sequence[float], empty_reason: str) -> MetricValue:
    if not values:
        return MetricValue.undefined(empty_reason)
    return MetricValue.defined(sum(values) / len(values))


def _order_metric(values: Sequence[float], q: float, empty_reason: str) -> MetricValue:
    if not values:
        return MetricValue.undefined(empty_reason)
    ordered = sorted(values)
    if len(ordered) == 1:
        return MetricValue.defined(ordered[0])
    index = q * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return MetricValue.defined(ordered[low])
    fraction = index - low
    return MetricValue.defined(ordered[low] * (1 - fraction) + ordered[high] * fraction)


def _require_metric_equal(actual: MetricValue, expected: MetricValue, field: str) -> None:
    if actual.value is None or expected.value is None:
        if actual != expected:
            raise EvidenceValidationError(
                f"{field} must be {expected.to_dict()}, got {actual.to_dict()}"
            )
        return
    if actual.reason != expected.reason or not math.isclose(
        actual.value, expected.value, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise EvidenceValidationError(
            f"{field} disagrees with raw-count recomputation: "
            f"expected {expected.value}, got {actual.value}"
        )


def _reject_nonfinite(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceValidationError(
            f"{path} contains a non-finite number; use null plus an explicit reason"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceValidationError(f"{field} must be an object")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvidenceValidationError(f"{field} must be an array")
    return value


def _require_exact_keys(data: Mapping[str, Any], keys: set[str], field: str) -> None:
    missing = keys - set(data)
    extra = set(data) - keys
    if missing or extra:
        raise EvidenceValidationError(
            f"{field} schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _require_nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{field} must be a non-empty string")
    return value


def _optional_nonempty_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_str(value, field)


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise EvidenceValidationError(f"{field} must be boolean")
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceValidationError(f"{field} must be an integer >= {minimum}")
    return value


def _require_nonnegative_float(value: Any, field: str) -> float:
    result = _require_finite_float(value, field)
    if result < 0:
        raise EvidenceValidationError(f"{field} must be nonnegative")
    return result


def _require_finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceValidationError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceValidationError(f"{field} must be finite")
    return result


def _require_sha256(value: Any, field: str) -> str:
    text = _require_nonempty_str(value, field)
    if not _SHA256_RE.fullmatch(text):
        raise EvidenceValidationError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _optional_sha256(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, field)


def _require_git(value: Any, field: str) -> str:
    text = _require_nonempty_str(value, field)
    if not _GIT_RE.fullmatch(text):
        raise EvidenceValidationError(f"{field} must be a lowercase 40- or 64-hex digest")
    return text


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    sequence = _require_sequence(value, field)
    result = tuple(_require_nonempty_str(v, f"{field}[]") for v in sequence)
    if len(result) != len(set(result)):
        raise EvidenceValidationError(f"{field} contains duplicates")
    return result


def _integer_tuple(value: Any, field: str) -> tuple[int, ...]:
    sequence = _require_sequence(value, field)
    return tuple(_require_int(v, f"{field}[]") for v in sequence)


def _parse_enum(enum_type: type[Enum], value: Any, field: str):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = [entry.value for entry in enum_type]
        raise EvidenceValidationError(f"{field} must be one of {allowed}") from exc


def _embryo_from_clip(clip_id: str) -> str:
    return clip_id.split("_", 1)[0]


__all__ = [
    "ADJUSTMENT_ALPHA",
    "DIVISION_SCORE_WEIGHT",
    "PINNED_SCORER_COMMIT",
    "PINNED_SCORER_SHA256",
    "EvaluationStatus",
    "EvidenceClass",
    "EvidenceValidationError",
    "MetricValue",
    "PerClipEvidence",
    "TrainingMembership",
    "aggregate_evidence",
    "audit_evidence_bundle",
    "audit_historical_outer_progress",
    "build_per_clip_evidence",
    "canonical_sha256",
    "load_json",
    "sha256_file",
    "strict_json_dumps",
    "verify_artifact_sha256",
]
