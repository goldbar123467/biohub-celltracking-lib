"""Durable transactional state for one campaign controller.

SQLite is the authority. JSON exports are review artifacts and never accepted as
authorization. All external mutations must first acquire a persisted intent.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from biohub_ct.campaign.budget import (
    BudgetExceeded,
    BudgetSnapshot,
    ReservationError,
    UnknownBudgetError,
    decimal_text,
)
from biohub_ct.campaign.contracts import (
    ContractError,
    IntentState,
    JobState,
    ReleaseState,
    candidate_digest,
    candidate_document,
    canonical_json,
    intent_digest,
    require_resolved,
    run_spec_digest,
    run_spec_preimage,
    validate_candidate,
    validate_candidate_identity,
    validate_identifier,
    validate_job_transition,
    validate_release_transition,
    validate_run_spec,
)


class CampaignStateError(RuntimeError):
    """Base class for state-store failures."""


class ApprovalError(CampaignStateError):
    """An approval is absent, stale, altered, or already consumed."""


class DuplicateIntentError(CampaignStateError):
    """A second external intent conflicts with a persisted one."""


class DispatchError(CampaignStateError):
    """A worker or submission dispatch does not match its durable intent."""


class ReviewLockBusy(CampaignStateError):
    """Another process owns the controller review lock."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO meta(key, value) VALUES ('state_version', '0');
INSERT OR IGNORE INTO meta(key, value) VALUES ('review_fencing_token', '0');
INSERT OR IGNORE INTO meta(key, value) VALUES ('review_owner', 'null');

CREATE TABLE IF NOT EXISTS campaign (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    campaign_id TEXT NOT NULL UNIQUE,
    document_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    run_id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS releases (
    candidate_id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    decision_id TEXT PRIMARY KEY,
    subject_kind TEXT NOT NULL CHECK(subject_kind IN ('run','release')),
    subject_id TEXT NOT NULL,
    subject_digest TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('LAUNCH','SUBMIT')),
    reviewer TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    consumed_at TEXT,
    superseded_by TEXT,
    operational_approval INTEGER NOT NULL CHECK(operational_approval IN (0,1)),
    quality_class TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    reservation_ids_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS approvals_subject ON approvals(subject_kind, subject_id, action);

CREATE TABLE IF NOT EXISTS intents (
    intent_id TEXT PRIMARY KEY,
    intent_kind TEXT NOT NULL CHECK(intent_kind IN ('launch','submission')),
    subject_kind TEXT NOT NULL CHECK(subject_kind IN ('run','release')),
    subject_id TEXT NOT NULL,
    subject_digest TEXT NOT NULL,
    intent_digest TEXT NOT NULL UNIQUE,
    request_id TEXT NOT NULL,
    description_tag TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    external_id TEXT,
    receipt_json TEXT,
    reservation_ids_json TEXT NOT NULL,
    fencing_token INTEGER NOT NULL,
    approval_id TEXT NOT NULL REFERENCES approvals(decision_id),
    UNIQUE(intent_kind, subject_id),
    UNIQUE(intent_kind, request_id)
);

CREATE TABLE IF NOT EXISTS ledgers (
    ledger_id TEXT PRIMARY KEY,
    resource_scope TEXT NOT NULL,
    unit TEXT NOT NULL,
    currency TEXT,
    authorized_total TEXT,
    confirmed_spend TEXT NOT NULL,
    unreconciled_spend TEXT NOT NULL,
    protected_reserve TEXT NOT NULL,
    observation_time TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reservations (
    reservation_id TEXT PRIMARY KEY,
    ledger_id TEXT NOT NULL REFERENCES ledgers(ledger_id),
    subject_kind TEXT NOT NULL CHECK(subject_kind IN ('run','release')),
    subject_id TEXT NOT NULL,
    amount TEXT NOT NULL,
    allow_protected INTEGER NOT NULL CHECK(allow_protected IN (0,1)),
    status TEXT NOT NULL CHECK(status IN ('ACTIVE','SETTLED','RELEASED')),
    actual_amount TEXT,
    intent_id TEXT,
    note TEXT,
    created_at TEXT NOT NULL,
    settled_at TEXT
);
CREATE INDEX IF NOT EXISTS reservations_ledger ON reservations(ledger_id, status);

CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    state_version_before INTEGER NOT NULL,
    state_version_after INTEGER NOT NULL,
    subject_kind TEXT,
    subject_id TEXT,
    request_id TEXT,
    payload_json TEXT NOT NULL,
    receipt_path TEXT,
    occurred_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update
BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete
BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
"""


def _now(value: datetime | None = None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return current.astimezone(UTC)


def _iso(value: datetime | None = None) -> str:
    return _now(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _json_load(value: str | None) -> Any:
    return None if value is None else json.loads(value)


class ReviewLease:
    """Cross-platform process lock paired with a monotonic database token."""

    def __init__(
        self,
        store: CampaignStore,
        owner: str,
        path: Path,
        *,
        blocking: bool = False,
    ) -> None:
        self.store = store
        self.owner = validate_identifier(owner, "review owner")
        self.path = path
        self.blocking = blocking
        self.fencing_token: int | None = None
        self._stream: Any = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                mode = msvcrt.LK_LOCK if self.blocking else msvcrt.LK_NBLCK
                msvcrt.locking(stream.fileno(), mode, 1)
            else:
                import fcntl

                flags = fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB)
                fcntl.flock(stream.fileno(), flags)
        except OSError as exc:
            stream.close()
            raise ReviewLockBusy(f"review lock is already owned: {self.path}") from exc
        self._stream = stream
        try:
            self.fencing_token = self.store._claim_review(self.owner)
        except BaseException:
            self._unlock()
            raise
        return self

    def _unlock(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self.fencing_token is not None:
                self.store._release_review(self.owner, self.fencing_token)
        finally:
            self._unlock()


class CampaignStore:
    """Transactional controller state with immutable identities and event history."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def review_lock(
        self,
        owner: str,
        *,
        path: str | os.PathLike[str] | None = None,
        blocking: bool = False,
    ) -> ReviewLease:
        lock_path = Path(path).resolve() if path is not None else self.path.with_suffix(".review.lock")
        return ReviewLease(self, owner, lock_path, blocking=blocking)

    def _event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        *,
        subject_kind: str | None = None,
        subject_id: str | None = None,
        request_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        receipt_path: str | None = None,
        occurred_at: str | None = None,
    ) -> int:
        before = int(connection.execute("SELECT value FROM meta WHERE key='state_version'").fetchone()[0])
        after = before + 1
        timestamp = occurred_at or _iso()
        event_id = f"event-{after:020d}"
        connection.execute("UPDATE meta SET value=? WHERE key='state_version'", (str(after),))
        connection.execute(
            """INSERT INTO events(event_id,event_type,state_version_before,state_version_after,
               subject_kind,subject_id,request_id,payload_json,receipt_path,occurred_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id,
                event_type,
                before,
                after,
                subject_kind,
                subject_id,
                request_id,
                canonical_json(payload or {}),
                receipt_path,
                timestamp,
            ),
        )
        return after

    def _claim_review(self, owner: str) -> int:
        with self.transaction() as connection:
            token = int(
                connection.execute("SELECT value FROM meta WHERE key='review_fencing_token'").fetchone()[0]
            ) + 1
            connection.execute(
                "UPDATE meta SET value=? WHERE key='review_fencing_token'", (str(token),)
            )
            connection.execute(
                "UPDATE meta SET value=? WHERE key='review_owner'",
                (canonical_json({"owner": owner, "fencing_token": token, "acquired_at": _iso()}),),
            )
            self._event(
                connection,
                "REVIEW_LOCK_ACQUIRED",
                subject_kind="controller",
                subject_id=owner,
                payload={"fencing_token": token},
            )
            return token

    def _release_review(self, owner: str, token: int) -> None:
        with self.transaction() as connection:
            current = _json_load(
                connection.execute("SELECT value FROM meta WHERE key='review_owner'").fetchone()[0]
            )
            if (
                not isinstance(current, dict)
                or current.get("owner") != owner
                or current.get("fencing_token") != token
            ):
                raise ReviewLockBusy("review lock ownership changed before release")
            connection.execute("UPDATE meta SET value='null' WHERE key='review_owner'")
            self._event(
                connection,
                "REVIEW_LOCK_RELEASED",
                subject_kind="controller",
                subject_id=owner,
                payload={"fencing_token": token},
            )

    def _assert_review_token(self, connection: sqlite3.Connection, token: int) -> None:
        if isinstance(token, bool) or not isinstance(token, int) or token <= 0:
            raise ApprovalError("mutation requires a positive integer review fencing token")
        current = _json_load(
            connection.execute("SELECT value FROM meta WHERE key='review_owner'").fetchone()[0]
        )
        if not isinstance(current, dict) or current.get("fencing_token") != token:
            raise ApprovalError("mutation requires the current review fencing token")

    @staticmethod
    def _validate_campaign_document(document: Mapping[str, Any]) -> None:
        if document.get("schema_version") != 1:
            raise ContractError("campaign schema_version must be 1")
        validate_identifier(document.get("campaign_id"), "campaign_id")
        require_resolved(document.get("competition"), "competition")
        if document.get("status") not in {
            "INVENTORY_REQUIRED", "ACTIVE", "PAUSED", "STOPPED", "COMPLETE"
        }:
            raise ContractError("campaign status is invalid")
        authorization = document.get("authorization")
        if not isinstance(authorization, Mapping):
            raise ContractError("campaign authorization must be an object")
        for field in (
            "routine_runs_and_submissions",
            "existing_allocation_authorized",
            "new_rental_purchase_authorized",
            "rental_extension_authorized",
        ):
            if not isinstance(authorization.get(field), bool):
                raise ContractError(f"authorization.{field} must be an explicit boolean")
        if (
            authorization["routine_runs_and_submissions"]
            or authorization["existing_allocation_authorized"]
        ):
            require_resolved(authorization.get("source"), "authorization.source")
        references = document.get("prior_campaign_refs")
        if not isinstance(references, list):
            raise ContractError("prior_campaign_refs must be a list")
        seen_references: set[str] = set()
        for index, reference in enumerate(references):
            if not isinstance(reference, Mapping):
                raise ContractError(f"prior_campaign_refs[{index}] must be an object")
            prior_id = validate_identifier(
                reference.get("campaign_id"), f"prior_campaign_refs[{index}].campaign_id"
            )
            if prior_id in seen_references:
                raise ContractError("prior_campaign_refs contains a duplicate campaign")
            seen_references.add(prior_id)
            ledger_status = reference.get("ledger_status")
            legacy_status = reference.get("status")
            if ledger_status is None and legacy_status is None:
                raise ContractError(
                    f"prior_campaign_refs[{index}] requires ledger_status or legacy status"
                )
            if ledger_status is not None:
                require_resolved(
                    ledger_status, f"prior_campaign_refs[{index}].ledger_status"
                )
            if legacy_status is not None:
                require_resolved(legacy_status, f"prior_campaign_refs[{index}].status")
        expires_at = authorization.get("expires_at")
        if expires_at is not None:
            try:
                _parse_time(expires_at)
            except (TypeError, ValueError) as exc:
                raise ContractError("authorization.expires_at must be ISO 8601 UTC") from exc
        canonical_json(document)

    def initialize_campaign(
        self,
        campaign_id: str,
        competition: str,
        *,
        authorization: Mapping[str, Any] | None = None,
        prior_campaign_refs: Sequence[Mapping[str, Any]] = (),
        status: str = "INVENTORY_REQUIRED",
        fields: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a fresh campaign without importing or resetting historical spend."""
        base_authorization = {
            "routine_runs_and_submissions": False,
            "existing_allocation_authorized": False,
            "new_rental_purchase_authorized": False,
            "rental_extension_authorized": False,
            "source": None,
            "expires_at": None,
        }
        if authorization is not None:
            base_authorization.update(dict(authorization))
        document = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "status": status,
            "competition": competition,
            "authorization": base_authorization,
            "prior_campaign_refs": [dict(reference) for reference in prior_campaign_refs],
            **dict(fields or {}),
        }
        # Fixed identity fields cannot be overwritten via the extension mapping.
        document.update(schema_version=1, campaign_id=campaign_id, status=status,
                        competition=competition,
                        authorization=base_authorization,
                        prior_campaign_refs=[dict(reference) for reference in prior_campaign_refs])
        self._validate_campaign_document(document)
        timestamp = _iso()
        with self.transaction() as connection:
            existing = connection.execute("SELECT * FROM campaign WHERE singleton=1").fetchone()
            if existing is not None:
                saved = _json_load(existing["document_json"])
                if saved == document:
                    return saved
                raise ContractError(
                    "campaign store already has an identity; create a new database for a fresh campaign"
                )
            connection.execute(
                "INSERT INTO campaign VALUES(1,?,?,?,?)",
                (campaign_id, canonical_json(document), timestamp, timestamp),
            )
            self._event(connection, "CAMPAIGN_INITIALIZED", subject_kind="campaign",
                        subject_id=campaign_id, request_id=request_id,
                        payload={"status": status,
                                 "authorization": base_authorization,
                                 "prior_campaign_refs": document["prior_campaign_refs"]})
        return self.get_campaign()

    def get_campaign(self) -> dict[str, Any]:
        with self.transaction(immediate=False) as connection:
            row = connection.execute("SELECT * FROM campaign WHERE singleton=1").fetchone()
            if row is None:
                raise KeyError("campaign is not initialized")
            return _json_load(row["document_json"])

    def update_campaign(
        self,
        changes: Mapping[str, Any],
        *,
        reviewer: str,
        reason: str,
        request_id: str | None = None,
        fencing_token: int | None = None,
    ) -> dict[str, Any]:
        """Audit a recursive patch while preserving the campaign's identity and history refs."""
        validate_identifier(reviewer, "reviewer")
        require_resolved(reason, "reason")

        def merge(original: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
            result = dict(original)
            for key, value in patch.items():
                if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = value
            return result

        if "authorization" in changes and fencing_token is None:
            raise ApprovalError("authorization changes require the current review fencing token")
        with self.transaction() as connection:
            if fencing_token is not None:
                self._assert_review_token(connection, fencing_token)
            row = connection.execute("SELECT * FROM campaign WHERE singleton=1").fetchone()
            if row is None:
                raise KeyError("campaign is not initialized")
            prior = _json_load(row["document_json"])
            updated = merge(prior, changes)
            if updated.get("campaign_id") != prior["campaign_id"]:
                raise ContractError("campaign_id cannot change")
            if updated.get("competition") != prior["competition"]:
                raise ContractError("campaign competition cannot change")
            if updated.get("prior_campaign_refs") != prior["prior_campaign_refs"]:
                raise ContractError("prior campaign references are append-only and cannot be reset")
            self._validate_campaign_document(updated)
            connection.execute("UPDATE campaign SET document_json=?,updated_at=? WHERE singleton=1",
                               (canonical_json(updated), _iso()))
            self._event(connection, "CAMPAIGN_UPDATED", subject_kind="campaign",
                        subject_id=prior["campaign_id"], request_id=request_id,
                        payload={"reviewer": reviewer, "reason": reason,
                                 "changes": dict(changes)})
        return self.get_campaign()

    def _assert_campaign_authorized(
        self, connection: sqlite3.Connection, *, action: str, now: datetime | None = None
    ) -> None:
        row = connection.execute("SELECT document_json FROM campaign WHERE singleton=1").fetchone()
        if row is None:
            raise ApprovalError("campaign authorization is not initialized")
        campaign = _json_load(row["document_json"])
        if campaign.get("status") != "ACTIVE":
            raise ApprovalError(f"campaign is not ACTIVE: {campaign.get('status')}")
        authorization = campaign.get("authorization", {})
        if action == "launch":
            authorized = (
                authorization.get("existing_allocation_authorized") is True
                or authorization.get("routine_runs_and_submissions") is True
            )
            message = "launches on the existing allocation are not authorized"
        elif action == "submission":
            authorized = authorization.get("routine_runs_and_submissions") is True
            message = "routine submissions are not authorized"
        else:
            raise ValueError(f"unsupported campaign authorization action: {action}")
        if not authorized:
            raise ApprovalError(message)
        require_resolved(authorization.get("source"), "authorization.source")
        expires_at = authorization.get("expires_at")
        if expires_at is not None and _now(now) >= _parse_time(expires_at):
            raise ApprovalError("campaign authorization expired")

    def register_run(self, spec: Mapping[str, Any], *, request_id: str | None = None) -> dict[str, Any]:
        validate_run_spec(spec)
        immutable = run_spec_preimage(spec)
        digest = run_spec_digest(immutable)
        run_id = str(immutable["run_id"])
        timestamp = _iso()
        with self.transaction() as connection:
            existing = connection.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone()
            if existing is not None:
                if existing["digest"] != digest:
                    raise ContractError(f"run_id {run_id!r} already has a different specification")
                return self._job_record(existing)
            connection.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?)",
                (run_id, canonical_json(immutable), digest, JobState.PROPOSED.value, timestamp, timestamp),
            )
            self._event(
                connection,
                "RUN_REGISTERED",
                subject_kind="run",
                subject_id=run_id,
                request_id=request_id,
                payload={"run_spec_sha256": digest},
            )
        return self.get_run(run_id)

    def update_run_spec(self, run_id: str, spec: Mapping[str, Any]) -> dict[str, Any]:
        validate_identifier(run_id, "run_id")
        validate_run_spec(spec)
        immutable = run_spec_preimage(spec)
        if immutable.get("run_id") != run_id:
            raise ContractError("run_id cannot change")
        digest = run_spec_digest(immutable)
        with self.transaction() as connection:
            row = self._require_row(connection, "jobs", "run_id", run_id)
            if row["state"] != JobState.PROPOSED.value:
                raise ContractError("run specification is immutable after approval")
            connection.execute(
                "UPDATE jobs SET spec_json=?,digest=?,updated_at=? WHERE run_id=?",
                (canonical_json(immutable), digest, _iso(), run_id),
            )
            self._event(connection, "RUN_SPEC_UPDATED", subject_kind="run", subject_id=run_id,
                        payload={"old_digest": row["digest"], "run_spec_sha256": digest})
        return self.get_run(run_id)

    def register_candidate(
        self, candidate: Mapping[str, Any], *, request_id: str | None = None
    ) -> dict[str, Any]:
        validate_candidate(candidate)
        immutable = candidate_document(candidate)
        digest = candidate_digest(immutable)
        candidate_id = str(immutable["candidate_id"])
        timestamp = _iso()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM releases WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            if existing is not None:
                if existing["digest"] != digest:
                    raise ContractError(f"candidate_id {candidate_id!r} already has a different release")
                return self._release_record(existing)
            connection.execute(
                "INSERT INTO releases VALUES(?,?,?,?,?,?)",
                (candidate_id, canonical_json(immutable), digest, ReleaseState.DRAFT.value,
                 timestamp, timestamp),
            )
            self._event(connection, "CANDIDATE_REGISTERED", subject_kind="release",
                        subject_id=candidate_id, request_id=request_id,
                        payload={"candidate_sha256": digest})
        return self.get_candidate(candidate_id)

    def update_candidate_spec(self, candidate_id: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
        validate_candidate(candidate)
        immutable = candidate_document(candidate)
        if immutable.get("candidate_id") != candidate_id:
            raise ContractError("candidate_id cannot change")
        digest = candidate_digest(immutable)
        with self.transaction() as connection:
            row = self._require_row(connection, "releases", "candidate_id", candidate_id)
            if row["state"] != ReleaseState.DRAFT.value:
                raise ContractError("candidate identity is immutable after freeze")
            connection.execute(
                "UPDATE releases SET spec_json=?,digest=?,updated_at=? WHERE candidate_id=?",
                (canonical_json(immutable), digest, _iso(), candidate_id),
            )
            self._event(connection, "CANDIDATE_SPEC_UPDATED", subject_kind="release",
                        subject_id=candidate_id,
                        payload={"old_digest": row["digest"], "candidate_sha256": digest})
        return self.get_candidate(candidate_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.transaction(immediate=False) as connection:
            return self._job_record(self._require_row(connection, "jobs", "run_id", run_id))

    def list_runs(self, *, states: Sequence[str | JobState] | None = None) -> list[dict[str, Any]]:
        state_values = None if states is None else [JobState(state).value for state in states]
        with self.transaction(immediate=False) as connection:
            if state_values is None:
                rows = connection.execute("SELECT * FROM jobs ORDER BY created_at,run_id").fetchall()
            elif not state_values:
                return []
            else:
                placeholders = ",".join("?" for _ in state_values)
                rows = connection.execute(
                    f"SELECT * FROM jobs WHERE state IN ({placeholders}) ORDER BY created_at,run_id",
                    state_values,
                ).fetchall()
            return [self._job_record(row) for row in rows]

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self.transaction(immediate=False) as connection:
            return self._release_record(
                self._require_row(connection, "releases", "candidate_id", candidate_id)
            )

    def list_candidates(
        self, *, states: Sequence[str | ReleaseState] | None = None
    ) -> list[dict[str, Any]]:
        state_values = None if states is None else [ReleaseState(state).value for state in states]
        with self.transaction(immediate=False) as connection:
            if state_values is None:
                rows = connection.execute(
                    "SELECT * FROM releases ORDER BY created_at,candidate_id"
                ).fetchall()
            elif not state_values:
                return []
            else:
                placeholders = ",".join("?" for _ in state_values)
                rows = connection.execute(
                    f"SELECT * FROM releases WHERE state IN ({placeholders}) "
                    "ORDER BY created_at,candidate_id",
                    state_values,
                ).fetchall()
            return [self._release_record(row) for row in rows]

    @staticmethod
    def _require_row(
        connection: sqlite3.Connection, table: str, key: str, value: str
    ) -> sqlite3.Row:
        row = connection.execute(f"SELECT * FROM {table} WHERE {key}=?", (value,)).fetchone()
        if row is None:
            raise KeyError(value)
        return row

    @staticmethod
    def _job_record(row: sqlite3.Row) -> dict[str, Any]:
        return {"run_id": row["run_id"], "spec": _json_load(row["spec_json"]),
                "run_spec_sha256": row["digest"], "state": row["state"],
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    @staticmethod
    def _release_record(row: sqlite3.Row) -> dict[str, Any]:
        return {"candidate_id": row["candidate_id"], "candidate": _json_load(row["spec_json"]),
                "candidate_sha256": row["digest"], "state": row["state"],
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def transition_job(
        self, run_id: str, target: str | JobState, *, evidence: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        target_state = JobState(target)
        if target_state in {JobState.APPROVED, JobState.LAUNCH_INTENT, JobState.RUNNING,
                            JobState.LAUNCH_UNKNOWN}:
            raise DispatchError(f"{target_state.value} requires its approval or intent API")
        with self.transaction() as connection:
            row = self._require_row(connection, "jobs", "run_id", run_id)
            validate_job_transition(row["state"], target_state)
            self._set_job_state(connection, run_id, target_state)
            self._event(connection, "JOB_TRANSITION", subject_kind="run", subject_id=run_id,
                        payload={"from": row["state"], "to": target_state.value,
                                 "evidence": evidence or {}})
        return self.get_run(run_id)

    def list_intent_reservations(self, intent_id: str) -> list[dict[str, Any]]:
        """Return the resource records bound to one immutable external intent."""
        with self.transaction(immediate=False) as connection:
            intent = self._require_row(connection, "intents", "intent_id", intent_id)
            reservation_ids = _json_load(intent["reservation_ids_json"])
            records = []
            for reservation_id in reservation_ids:
                row = self._require_row(
                    connection, "reservations", "reservation_id", reservation_id
                )
                ledger = self._require_row(
                    connection, "ledgers", "ledger_id", row["ledger_id"]
                )
                record = dict(row)
                record["unit"] = ledger["unit"]
                record["currency"] = ledger["currency"]
                records.append(record)
            return records

    def finalize_run(
        self,
        run_id: str,
        target: str | JobState,
        *,
        evidence: Mapping[str, Any],
        settlements: Sequence[Mapping[str, Any]] | None = None,
        receipt_path: str,
    ) -> dict[str, Any]:
        """Atomically record an operational terminal result and known resource charges.

        Passing ``settlements=None`` deliberately leaves active reservations in place
        for later billing reconciliation. Passing a sequence requires an exact
        settlement for every still-active reservation on the launch intent.
        """
        target_state = JobState(target)
        terminal = {
            JobState.COMPLETE,
            JobState.PARTIAL,
            JobState.FAILED,
            JobState.STOPPED,
        }
        if target_state not in terminal:
            raise DispatchError("run finalization requires an operational terminal state")
        if not isinstance(evidence, Mapping) or not evidence:
            raise ContractError("run finalization requires immutable completion evidence")
        if target_state == JobState.COMPLETE and evidence.get("quality_review_required") is not True:
            raise ContractError("operational COMPLETE must remain pending separate quality review")
        require_resolved(receipt_path, "receipt_path")
        canonical_json(evidence)
        normalized: dict[str, str] | None = None
        if settlements is not None:
            if isinstance(settlements, (str, bytes)) or not isinstance(settlements, Sequence):
                raise ReservationError("settlements must be a sequence of reservation records")
            normalized = {}
            for item in settlements:
                if not isinstance(item, Mapping):
                    raise ReservationError("each settlement must be an object")
                reservation_id = validate_identifier(
                    item.get("reservation_id"), "reservation_id"
                )
                if reservation_id in normalized:
                    raise ReservationError("a finalization cannot settle one reservation twice")
                normalized[reservation_id] = decimal_text(
                    item.get("actual_amount"), "actual_amount"
                )

        with self.transaction() as connection:
            job = self._require_row(connection, "jobs", "run_id", run_id)
            intents = connection.execute(
                "SELECT * FROM intents WHERE intent_kind='launch' AND subject_id=?",
                (run_id,),
            ).fetchall()
            if len(intents) != 1 or intents[0]["state"] != IntentState.CONFIRMED.value:
                raise DispatchError("run finalization requires one confirmed launch intent")
            intent = intents[0]
            event_evidence = dict(evidence)
            if job["state"] != JobState.RUNNING.value:
                prior = connection.execute(
                    """SELECT payload_json,receipt_path FROM events
                       WHERE event_type='RUN_FINALIZED' AND subject_id=?
                       ORDER BY sequence DESC LIMIT 1""",
                    (run_id,),
                ).fetchone()
                if prior is not None:
                    payload = _json_load(prior["payload_json"])
                    if (
                        job["state"] == target_state.value
                        and payload.get("evidence") == event_evidence
                        and prior["receipt_path"] == str(receipt_path)
                    ):
                        retained = payload.get("retained_reservations", [])
                        return {
                            "run": self._job_record(job),
                            "settlement_needed": bool(retained),
                            "retained_reservations": retained,
                            "idempotent": True,
                        }
                raise DispatchError(f"run cannot be finalized from {job['state']}")
            validate_job_transition(job["state"], target_state)
            reservation_ids = _json_load(intent["reservation_ids_json"])
            active: dict[str, sqlite3.Row] = {}
            for reservation_id in reservation_ids:
                row = self._require_row(
                    connection, "reservations", "reservation_id", reservation_id
                )
                if row["intent_id"] != intent["intent_id"]:
                    raise ReservationError("run reservation is not bound to its launch intent")
                if row["status"] == "ACTIVE":
                    active[reservation_id] = row
            if normalized is not None and set(normalized) != set(active):
                raise ReservationError(
                    "known finalization settlements must cover every active run reservation"
                )
            settled = []
            if normalized is not None:
                for reservation_id, actual_amount in normalized.items():
                    row = active[reservation_id]
                    self._settle(connection, reservation_id, actual_amount)
                    settled.append(
                        {
                            "reservation_id": reservation_id,
                            "ledger_id": row["ledger_id"],
                            "reserved_amount": row["amount"],
                            "actual_amount": actual_amount,
                        }
                    )
            retained = []
            if normalized is None:
                for reservation_id, row in active.items():
                    ledger = self._require_row(
                        connection, "ledgers", "ledger_id", row["ledger_id"]
                    )
                    retained.append(
                        {
                            "reservation_id": reservation_id,
                            "ledger_id": row["ledger_id"],
                            "unit": ledger["unit"],
                            "reserved_amount": row["amount"],
                        }
                    )
            self._set_job_state(connection, run_id, target_state)
            self._event(
                connection,
                "RUN_FINALIZED",
                subject_kind="run",
                subject_id=run_id,
                request_id=intent["request_id"],
                receipt_path=str(receipt_path),
                payload={
                    "from": job["state"],
                    "to": target_state.value,
                    "evidence": event_evidence,
                    "settlements": settled,
                    "retained_reservations": retained,
                },
            )
            result = {
                "run": self._job_record(
                    self._require_row(connection, "jobs", "run_id", run_id)
                ),
                "settlement_needed": bool(retained),
                "retained_reservations": retained,
                "idempotent": False,
            }
        return result

    def transition_release(
        self, candidate_id: str, target: str | ReleaseState,
        *, evidence: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        target_state = ReleaseState(target)
        if target_state in {ReleaseState.APPROVED, ReleaseState.SUBMIT_INTENT,
                            ReleaseState.SUBMISSION_UNKNOWN, ReleaseState.ACCEPTED}:
            raise DispatchError(f"{target_state.value} requires its approval or intent API")
        with self.transaction() as connection:
            row = self._require_row(connection, "releases", "candidate_id", candidate_id)
            validate_release_transition(row["state"], target_state)
            if target_state == ReleaseState.FROZEN:
                validate_candidate_identity(_json_load(row["spec_json"]))
            self._set_release_state(connection, candidate_id, target_state)
            self._event(connection, "RELEASE_TRANSITION", subject_kind="release",
                        subject_id=candidate_id,
                        payload={"from": row["state"], "to": target_state.value,
                                 "evidence": evidence or {}})
        return self.get_candidate(candidate_id)

    @staticmethod
    def _set_job_state(connection: sqlite3.Connection, run_id: str, state: JobState) -> None:
        connection.execute("UPDATE jobs SET state=?,updated_at=? WHERE run_id=?",
                           (state.value, _iso(), run_id))

    @staticmethod
    def _set_release_state(
        connection: sqlite3.Connection, candidate_id: str, state: ReleaseState
    ) -> None:
        connection.execute("UPDATE releases SET state=?,updated_at=? WHERE candidate_id=?",
                           (state.value, _iso(), candidate_id))

    # ----- finite resource ledgers -----

    def create_ledger(
        self, ledger_id: str, *, resource_scope: str, unit: str,
        authorized_total: Any | None, currency: str | None = None,
        confirmed_spend: Any = 0, unreconciled_spend: Any = 0,
        protected_reserve: Any = 0, observation_time: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> BudgetSnapshot:
        validate_identifier(ledger_id, "ledger_id")
        require_resolved(resource_scope, "resource_scope")
        require_resolved(unit, "unit")
        total = None if authorized_total is None else decimal_text(authorized_total, "authorized_total")
        confirmed = decimal_text(confirmed_spend, "confirmed_spend")
        unreconciled = decimal_text(unreconciled_spend, "unreconciled_spend")
        protected = decimal_text(protected_reserve, "protected_reserve")
        if total is not None and Decimal(confirmed) + Decimal(unreconciled) + Decimal(protected) > Decimal(total):
            raise BudgetExceeded("initial charges and protected reserve exceed authorization")
        timestamp = _iso()
        with self.transaction() as connection:
            if connection.execute("SELECT 1 FROM ledgers WHERE ledger_id=?", (ledger_id,)).fetchone():
                raise ContractError(f"ledger_id {ledger_id!r} already exists")
            connection.execute(
                """INSERT INTO ledgers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ledger_id, str(resource_scope), str(unit), currency, total, confirmed,
                 unreconciled, protected, observation_time, canonical_json(metadata or {}),
                 timestamp, timestamp),
            )
            self._event(connection, "LEDGER_CREATED", subject_kind="ledger", subject_id=ledger_id,
                        payload={"resource_scope": resource_scope, "unit": unit,
                                 "authorized_total": total})
        return self.budget_snapshot(ledger_id)

    def _budget_snapshot(self, connection: sqlite3.Connection, ledger_id: str) -> BudgetSnapshot:
        row = self._require_row(connection, "ledgers", "ledger_id", ledger_id)
        active = sum(
            (Decimal(item[0]) for item in connection.execute(
                "SELECT amount FROM reservations WHERE ledger_id=? AND status='ACTIVE'", (ledger_id,)
            )), Decimal(0))
        total = None if row["authorized_total"] is None else Decimal(row["authorized_total"])
        confirmed = Decimal(row["confirmed_spend"])
        unreconciled = Decimal(row["unreconciled_spend"])
        protected = Decimal(row["protected_reserve"])
        available = None if total is None else total - confirmed - unreconciled - active - protected
        return BudgetSnapshot(ledger_id, row["resource_scope"], row["unit"], row["currency"],
                              total, confirmed, unreconciled, protected, active, available,
                              row["observation_time"])

    def budget_snapshot(self, ledger_id: str) -> BudgetSnapshot:
        with self.transaction(immediate=False) as connection:
            return self._budget_snapshot(connection, ledger_id)

    def _reserve(
        self, connection: sqlite3.Connection, ledger_id: str, reservation_id: str, *,
        subject_kind: str, subject_id: str, amount: Any, allow_protected: bool,
        note: str | None = None,
    ) -> dict[str, Any]:
        validate_identifier(reservation_id, "reservation_id")
        if subject_kind not in {"run", "release"}:
            raise ContractError("reservation subject_kind must be run or release")
        amount_text = decimal_text(amount, "reservation amount", allow_zero=False)
        existing = connection.execute(
            "SELECT * FROM reservations WHERE reservation_id=?", (reservation_id,)
        ).fetchone()
        if existing is not None:
            same = (existing["ledger_id"] == ledger_id and existing["subject_kind"] == subject_kind
                    and existing["subject_id"] == subject_id and existing["amount"] == amount_text
                    and existing["status"] == "ACTIVE")
            if same:
                return dict(existing)
            raise ReservationError(f"reservation_id {reservation_id!r} conflicts with existing record")
        snapshot = self._budget_snapshot(connection, ledger_id)
        if snapshot.authorized_total is None:
            raise UnknownBudgetError(f"ledger {ledger_id!r} has no numerical authorization")
        headroom = (snapshot.authorized_total - snapshot.confirmed_spend
                    - snapshot.unreconciled_spend - snapshot.outstanding_reservations)
        if not allow_protected:
            headroom -= snapshot.protected_reserve
        if Decimal(amount_text) > headroom:
            raise BudgetExceeded(
                f"reservation {amount_text} {snapshot.unit} exceeds available {headroom}"
            )
        timestamp = _iso()
        connection.execute(
            """INSERT INTO reservations(reservation_id,ledger_id,subject_kind,subject_id,amount,
               allow_protected,status,actual_amount,intent_id,note,created_at,settled_at)
               VALUES(?,?,?,?,?,?, 'ACTIVE',NULL,NULL,?,?,NULL)""",
            (reservation_id, ledger_id, subject_kind, subject_id, amount_text,
             int(allow_protected), note, timestamp),
        )
        return dict(self._require_row(connection, "reservations", "reservation_id", reservation_id))

    def reserve_budget(
        self, ledger_id: str, reservation_id: str, *, subject_kind: str,
        subject_id: str, amount: Any, allow_protected: bool = False,
        note: str | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            record = self._reserve(connection, ledger_id, reservation_id,
                                   subject_kind=subject_kind, subject_id=subject_id,
                                   amount=amount, allow_protected=allow_protected, note=note)
            self._event(connection, "BUDGET_RESERVED", subject_kind=subject_kind,
                        subject_id=subject_id,
                        payload={"ledger_id": ledger_id, "reservation_id": reservation_id,
                                 "amount": record["amount"]})
            return record

    def _settle(self, connection: sqlite3.Connection, reservation_id: str, actual_amount: Any) -> None:
        row = self._require_row(connection, "reservations", "reservation_id", reservation_id)
        if row["status"] != "ACTIVE":
            raise ReservationError("only an active reservation can be settled")
        actual = decimal_text(actual_amount, "actual_amount")
        ledger = self._require_row(connection, "ledgers", "ledger_id", row["ledger_id"])
        new_spend = Decimal(ledger["confirmed_spend"]) + Decimal(actual)
        timestamp = _iso()
        connection.execute("UPDATE ledgers SET confirmed_spend=?,updated_at=? WHERE ledger_id=?",
                           (decimal_text(new_spend, "confirmed_spend"), timestamp, row["ledger_id"]))
        connection.execute(
            "UPDATE reservations SET status='SETTLED',actual_amount=?,settled_at=? WHERE reservation_id=?",
            (actual, timestamp, reservation_id),
        )

    def settle_reservation(self, reservation_id: str, actual_amount: Any) -> dict[str, Any]:
        with self.transaction() as connection:
            self._settle(connection, reservation_id, actual_amount)
            row = self._require_row(connection, "reservations", "reservation_id", reservation_id)
            self._event(connection, "RESERVATION_SETTLED", subject_kind=row["subject_kind"],
                        subject_id=row["subject_id"],
                        payload={"reservation_id": reservation_id,
                                 "reserved": row["amount"], "actual": row["actual_amount"]})
            return dict(row)

    def _release_reservation(
        self, connection: sqlite3.Connection, reservation_id: str, reason: str
    ) -> None:
        row = self._require_row(connection, "reservations", "reservation_id", reservation_id)
        if row["status"] != "ACTIVE":
            raise ReservationError("only an active reservation can be released")
        connection.execute(
            "UPDATE reservations SET status='RELEASED',note=?,settled_at=? WHERE reservation_id=?",
            (reason, _iso(), reservation_id),
        )

    def release_reservation(self, reservation_id: str, *, reason: str) -> dict[str, Any]:
        require_resolved(reason, "reason")
        with self.transaction() as connection:
            self._release_reservation(connection, reservation_id, reason)
            row = self._require_row(connection, "reservations", "reservation_id", reservation_id)
            self._event(connection, "RESERVATION_RELEASED", subject_kind=row["subject_kind"],
                        subject_id=row["subject_id"],
                        payload={"reservation_id": reservation_id, "reason": reason})
            return dict(row)

    def set_unreconciled_spend(self, ledger_id: str, amount: Any, *, evidence: str) -> BudgetSnapshot:
        value = decimal_text(amount, "unreconciled_spend")
        require_resolved(evidence, "evidence")
        with self.transaction() as connection:
            row = self._require_row(connection, "ledgers", "ledger_id", ledger_id)
            connection.execute("UPDATE ledgers SET unreconciled_spend=?,updated_at=? WHERE ledger_id=?",
                               (value, _iso(), ledger_id))
            self._event(connection, "UNRECONCILED_SPEND_UPDATED", subject_kind="ledger",
                        subject_id=ledger_id,
                        payload={"prior": row["unreconciled_spend"], "current": value,
                                 "evidence": evidence})
        return self.budget_snapshot(ledger_id)

    # ----- approvals and intents -----

    def _reserve_many(
        self, connection: sqlite3.Connection, subject_kind: str, subject_id: str,
        reservations: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        if not reservations:
            raise UnknownBudgetError("an external mutation requires at least one finite reservation")
        ids: list[str] = []
        for request in reservations:
            reservation_id = str(require_resolved(request.get("reservation_id"), "reservation_id"))
            if reservation_id in ids:
                raise ReservationError("one approval cannot repeat a reservation_id")
            self._reserve(connection, str(require_resolved(request.get("ledger_id"), "ledger_id")),
                          reservation_id, subject_kind=subject_kind, subject_id=subject_id,
                          amount=request.get("amount"),
                          allow_protected=bool(request.get("allow_protected", False)),
                          note=request.get("note"))
            ids.append(reservation_id)
        return ids

    def _insert_approval(
        self, connection: sqlite3.Connection, *, decision_id: str, subject_kind: str,
        subject_id: str, subject_digest: str, action: str, reviewer: str,
        reservation_ids: Sequence[str], reason: str, quality_class: str,
        evidence_paths: Sequence[str], valid_for_minutes: int,
        now: datetime | None,
    ) -> dict[str, Any]:
        validate_identifier(decision_id, "decision_id")
        validate_identifier(reviewer, "reviewer")
        require_resolved(reason, "reason")
        if (
            isinstance(valid_for_minutes, bool)
            or not isinstance(valid_for_minutes, int)
            or not 0 < valid_for_minutes <= 60
        ):
            raise ApprovalError("approval validity must be between 1 and 60 minutes")
        reviewed = _now(now)
        reviewed_at, valid_until = _iso(reviewed), _iso(reviewed + timedelta(minutes=valid_for_minutes))
        connection.execute(
            """INSERT INTO approvals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (decision_id, subject_kind, subject_id, subject_digest, action, reviewer,
             reviewed_at, valid_until, None, None, 1, quality_class,
             canonical_json(list(evidence_paths)), reason, canonical_json(list(reservation_ids))),
        )
        return dict(self._require_row(connection, "approvals", "decision_id", decision_id))

    def approve_run(
        self, run_id: str, *, decision_id: str, reviewer: str,
        reservations: Sequence[Mapping[str, Any]], reason: str,
        quality_class: str = "unproven", evidence_paths: Sequence[str] = (),
        valid_for_minutes: int = 60, fencing_token: int, now: datetime | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            self._assert_review_token(connection, fencing_token)
            self._assert_campaign_authorized(connection, action="launch", now=now)
            row = self._require_row(connection, "jobs", "run_id", run_id)
            if row["state"] not in {JobState.PROPOSED.value, JobState.APPROVED.value}:
                raise ApprovalError(f"run is not approvable from {row['state']}")
            spec = _json_load(row["spec_json"])
            validate_run_spec(spec, for_approval=True)
            ids = self._reserve_many(connection, "run", run_id, reservations)
            approval = self._insert_approval(
                connection, decision_id=decision_id, subject_kind="run", subject_id=run_id,
                subject_digest=row["digest"], action="LAUNCH", reviewer=reviewer,
                reservation_ids=ids, reason=reason, quality_class=quality_class,
                evidence_paths=evidence_paths, valid_for_minutes=valid_for_minutes, now=now)
            if row["state"] == JobState.PROPOSED.value:
                self._set_job_state(connection, run_id, JobState.APPROVED)
            self._event(connection, "RUN_APPROVED", subject_kind="run", subject_id=run_id,
                        payload={"decision_id": decision_id, "digest": row["digest"],
                                 "reservation_ids": ids, "valid_until": approval["valid_until"]})
            return self._approval_record(approval)

    def approve_submission(
        self, candidate_id: str, *, decision_id: str, reviewer: str,
        reservations: Sequence[Mapping[str, Any]], reason: str,
        quality_class: str = "unproven", evidence_paths: Sequence[str] = (),
        valid_for_minutes: int = 60, fencing_token: int, now: datetime | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            self._assert_review_token(connection, fencing_token)
            self._assert_campaign_authorized(connection, action="submission", now=now)
            row = self._require_row(connection, "releases", "candidate_id", candidate_id)
            if row["state"] not in {ReleaseState.OUTPUT_VALIDATED.value, ReleaseState.APPROVED.value}:
                raise ApprovalError(f"release is not approvable from {row['state']}")
            candidate = _json_load(row["spec_json"])
            validate_candidate(candidate, for_approval=True)
            ids = self._reserve_many(connection, "release", candidate_id, reservations)
            approval = self._insert_approval(
                connection, decision_id=decision_id, subject_kind="release",
                subject_id=candidate_id, subject_digest=row["digest"], action="SUBMIT",
                reviewer=reviewer, reservation_ids=ids, reason=reason,
                quality_class=quality_class, evidence_paths=evidence_paths,
                valid_for_minutes=valid_for_minutes, now=now)
            if row["state"] == ReleaseState.OUTPUT_VALIDATED.value:
                self._set_release_state(connection, candidate_id, ReleaseState.APPROVED)
            self._event(connection, "SUBMISSION_APPROVED", subject_kind="release",
                        subject_id=candidate_id,
                        payload={"decision_id": decision_id, "digest": row["digest"],
                                 "reservation_ids": ids, "valid_until": approval["valid_until"]})
            return self._approval_record(approval)

    def _consume_approval(
        self, connection: sqlite3.Connection, decision_id: str, *, subject_kind: str,
        subject_id: str, subject_digest: str, action: str, now: datetime | None,
    ) -> tuple[sqlite3.Row, list[str], str]:
        row = self._require_row(connection, "approvals", "decision_id", decision_id)
        if (row["subject_kind"], row["subject_id"], row["subject_digest"], row["action"]) != (
            subject_kind, subject_id, subject_digest, action
        ):
            raise ApprovalError("approval is not bound to this exact subject digest and action")
        current = _now(now)
        if row["consumed_at"] is not None:
            raise ApprovalError("approval was already consumed")
        if row["superseded_by"] is not None:
            raise ApprovalError("approval was superseded")
        if current >= _parse_time(row["valid_until"]):
            raise ApprovalError("approval expired before it was consumed")
        reservation_ids = _json_load(row["reservation_ids_json"])
        if not reservation_ids:
            raise ApprovalError("approval has no resource reservation")
        for reservation_id in reservation_ids:
            reservation = self._require_row(
                connection, "reservations", "reservation_id", reservation_id
            )
            if reservation["status"] != "ACTIVE" or reservation["subject_id"] != subject_id:
                raise ApprovalError("approval reservation is no longer active and bound")
        consumed_at = _iso(current)
        connection.execute("UPDATE approvals SET consumed_at=? WHERE decision_id=?",
                           (consumed_at, decision_id))
        return row, reservation_ids, consumed_at

    @staticmethod
    def _approval_record(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["operational_approval"] = bool(result["operational_approval"])
        result["evidence_paths"] = _json_load(result.pop("evidence_json"))
        result["reservation_ids"] = _json_load(result.pop("reservation_ids_json"))
        return result

    def get_approval(self, decision_id: str) -> dict[str, Any]:
        with self.transaction(immediate=False) as connection:
            return self._approval_record(
                self._require_row(connection, "approvals", "decision_id", decision_id)
            )

    def expire_approvals(self, *, now: datetime | None = None) -> list[str]:
        """Invalidate due unconsumed approvals and release only their active reservations."""
        current = _now(now)
        expired: list[str] = []
        with self.transaction() as connection:
            rows = connection.execute(
                """SELECT * FROM approvals
                   WHERE consumed_at IS NULL AND superseded_by IS NULL
                   ORDER BY reviewed_at,decision_id"""
            ).fetchall()
            for row in rows:
                if current < _parse_time(row["valid_until"]):
                    continue
                for reservation_id in _json_load(row["reservation_ids_json"]):
                    reservation = self._require_row(
                        connection, "reservations", "reservation_id", reservation_id
                    )
                    if reservation["status"] == "ACTIVE":
                        self._release_reservation(
                            connection, reservation_id, "unconsumed approval expired"
                        )
                connection.execute(
                    "UPDATE approvals SET superseded_by='EXPIRED' WHERE decision_id=?",
                    (row["decision_id"],),
                )
                self._event(
                    connection,
                    "APPROVAL_EXPIRED",
                    subject_kind=row["subject_kind"],
                    subject_id=row["subject_id"],
                    payload={"decision_id": row["decision_id"], "expired_at": _iso(current)},
                )
                expired.append(row["decision_id"])
        return expired

    def _create_intent(
        self, connection: sqlite3.Connection, *, intent_id: str, intent_kind: str,
        subject_kind: str, subject_id: str, subject_digest: str, request_id: str,
        description_tag: str, payload: Mapping[str, Any], reservation_ids: Sequence[str],
        fencing_token: int, approval_id: str,
    ) -> dict[str, Any]:
        validate_identifier(intent_id, "intent_id")
        validate_identifier(request_id, "request_id")
        require_resolved(description_tag, "description_tag")
        digest = intent_digest(intent_kind, payload)
        existing = connection.execute(
            "SELECT * FROM intents WHERE intent_kind=? AND subject_id=?", (intent_kind, subject_id)
        ).fetchone()
        if existing is not None:
            if (existing["intent_id"] == intent_id and existing["intent_digest"] == digest
                    and existing["request_id"] == request_id):
                return self._intent_record(existing)
            raise DuplicateIntentError(f"{intent_kind} intent already exists for {subject_id}")
        timestamp = _iso()
        try:
            connection.execute(
                """INSERT INTO intents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (intent_id, intent_kind, subject_kind, subject_id, subject_digest, digest,
                 request_id, description_tag, canonical_json(payload), IntentState.PENDING.value,
                 timestamp, timestamp, None, None, canonical_json(list(reservation_ids)),
                 fencing_token, approval_id),
            )
        except sqlite3.IntegrityError as exc:
            raise DuplicateIntentError("intent ID, request ID, or digest was already used") from exc
        for reservation_id in reservation_ids:
            connection.execute("UPDATE reservations SET intent_id=? WHERE reservation_id=?",
                               (intent_id, reservation_id))
        return self._intent_record(
            self._require_row(connection, "intents", "intent_id", intent_id)
        )

    def create_launch_intent(
        self, run_id: str, *, decision_id: str, intent_id: str, request_id: str,
        description_tag: str, fencing_token: int, now: datetime | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            self._assert_review_token(connection, fencing_token)
            self._assert_campaign_authorized(connection, action="launch", now=now)
            run = self._require_row(connection, "jobs", "run_id", run_id)
            if run["state"] != JobState.APPROVED.value:
                raise DispatchError(f"run cannot create intent from {run['state']}")
            _, reservations, consumed_at = self._consume_approval(
                connection, decision_id, subject_kind="run", subject_id=run_id,
                subject_digest=run["digest"], action="LAUNCH", now=now)
            payload = {"run_id": run_id, "run_spec_sha256": run["digest"],
                       "unique_description_tag": description_tag}
            intent = self._create_intent(
                connection, intent_id=intent_id, intent_kind="launch", subject_kind="run",
                subject_id=run_id, subject_digest=run["digest"], request_id=request_id,
                description_tag=description_tag, payload=payload,
                reservation_ids=reservations, fencing_token=fencing_token,
                approval_id=decision_id)
            self._set_job_state(connection, run_id, JobState.LAUNCH_INTENT)
            self._event(connection, "LAUNCH_INTENT_CREATED", subject_kind="run", subject_id=run_id,
                        request_id=request_id,
                        payload={"intent_id": intent_id, "intent_digest": intent["intent_digest"],
                                 "decision_id": decision_id, "consumed_at": consumed_at,
                                 "fencing_token": fencing_token})
            return intent

    def create_submission_intent(
        self, candidate_id: str, *, decision_id: str, intent_id: str, request_id: str,
        description_tag: str, fencing_token: int, now: datetime | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            self._assert_review_token(connection, fencing_token)
            self._assert_campaign_authorized(connection, action="submission", now=now)
            release = self._require_row(connection, "releases", "candidate_id", candidate_id)
            if release["state"] != ReleaseState.APPROVED.value:
                raise DispatchError(f"release cannot create intent from {release['state']}")
            candidate = _json_load(release["spec_json"])
            _, reservations, consumed_at = self._consume_approval(
                connection, decision_id, subject_kind="release", subject_id=candidate_id,
                subject_digest=release["digest"], action="SUBMIT", now=now)
            payload = {"competition": candidate["competition"],
                       "notebook_slug": candidate["notebook_slug"],
                       "notebook_version": candidate["notebook_version"],
                       "output_filename": candidate["output_filename"],
                       "candidate_sha256": release["digest"]}
            intent = self._create_intent(
                connection, intent_id=intent_id, intent_kind="submission",
                subject_kind="release", subject_id=candidate_id,
                subject_digest=release["digest"], request_id=request_id,
                description_tag=description_tag, payload=payload,
                reservation_ids=reservations, fencing_token=fencing_token,
                approval_id=decision_id)
            self._set_release_state(connection, candidate_id, ReleaseState.SUBMIT_INTENT)
            self._event(connection, "SUBMISSION_INTENT_CREATED", subject_kind="release",
                        subject_id=candidate_id, request_id=request_id,
                        payload={"intent_id": intent_id, "intent_digest": intent["intent_digest"],
                                 "decision_id": decision_id, "consumed_at": consumed_at,
                                 "fencing_token": fencing_token})
            return intent

    def authorize_launch(self, run_id: str, **kwargs: Any) -> dict[str, Any]:
        """Atomically approve, reserve, consume, and persist a launch intent."""
        approval_keys = {"decision_id", "reviewer", "reservations", "reason", "quality_class",
                         "evidence_paths", "valid_for_minutes", "fencing_token", "now"}
        approve = {key: value for key, value in kwargs.items() if key in approval_keys}
        intent = {key: value for key, value in kwargs.items()
                  if key in {"decision_id", "intent_id", "request_id", "description_tag",
                             "fencing_token", "now"}}
        # One outer transaction is required, so use the dedicated implementation below.
        return self._authorize_external("run", run_id, approve, intent)

    def authorize_submission(self, candidate_id: str, **kwargs: Any) -> dict[str, Any]:
        """Atomically approve, reserve, consume, and persist a submission intent."""
        approval_keys = {"decision_id", "reviewer", "reservations", "reason", "quality_class",
                         "evidence_paths", "valid_for_minutes", "fencing_token", "now"}
        approve = {key: value for key, value in kwargs.items() if key in approval_keys}
        intent = {key: value for key, value in kwargs.items()
                  if key in {"decision_id", "intent_id", "request_id", "description_tag",
                             "fencing_token", "now"}}
        return self._authorize_external("release", candidate_id, approve, intent)

    def _authorize_external(
        self, subject_kind: str, subject_id: str, approve: Mapping[str, Any],
        intent_args: Mapping[str, Any]
    ) -> dict[str, Any]:
        required = {"decision_id", "reviewer", "reservations", "reason", "fencing_token"}
        missing = required - approve.keys()
        missing |= {"intent_id", "request_id", "description_tag"} - intent_args.keys()
        if missing:
            raise TypeError(f"missing authorization arguments: {sorted(missing)}")
        token = approve["fencing_token"]
        if isinstance(token, bool) or not isinstance(token, int) or token <= 0:
            raise ApprovalError("fencing_token must be a positive integer")
        now = approve.get("now")
        with self.transaction() as connection:
            self._assert_review_token(connection, token)
            self._assert_campaign_authorized(
                connection,
                action="launch" if subject_kind == "run" else "submission",
                now=now,
            )
            if subject_kind == "run":
                table, key, initial, action = "jobs", "run_id", JobState.PROPOSED.value, "LAUNCH"
            else:
                table, key, initial, action = (
                    "releases", "candidate_id", ReleaseState.OUTPUT_VALIDATED.value, "SUBMIT"
                )
            row = self._require_row(connection, table, key, subject_id)
            if row["state"] != initial:
                raise ApprovalError(f"{subject_kind} is not ready for authorization from {row['state']}")
            spec = _json_load(row["spec_json"])
            (validate_run_spec(spec, for_approval=True) if subject_kind == "run"
             else validate_candidate(spec, for_approval=True))
            reservations = self._reserve_many(
                connection, subject_kind, subject_id, approve["reservations"]
            )
            approval = self._insert_approval(
                connection, decision_id=str(approve["decision_id"]), subject_kind=subject_kind,
                subject_id=subject_id, subject_digest=row["digest"], action=action,
                reviewer=str(approve["reviewer"]), reservation_ids=reservations,
                reason=str(approve["reason"]), quality_class=str(approve.get("quality_class", "unproven")),
                evidence_paths=approve.get("evidence_paths", ()),
                valid_for_minutes=int(approve.get("valid_for_minutes", 60)), now=now)
            _, reservations, consumed_at = self._consume_approval(
                connection, approval["decision_id"], subject_kind=subject_kind,
                subject_id=subject_id, subject_digest=row["digest"], action=action, now=now)
            if subject_kind == "run":
                payload = {"run_id": subject_id, "run_spec_sha256": row["digest"],
                           "unique_description_tag": intent_args["description_tag"]}
                kind = "launch"
            else:
                payload = {"competition": spec["competition"],
                           "notebook_slug": spec["notebook_slug"],
                           "notebook_version": spec["notebook_version"],
                           "output_filename": spec["output_filename"],
                           "candidate_sha256": row["digest"]}
                kind = "submission"
            intent = self._create_intent(
                connection, intent_id=str(intent_args["intent_id"]), intent_kind=kind,
                subject_kind=subject_kind, subject_id=subject_id, subject_digest=row["digest"],
                request_id=str(intent_args["request_id"]),
                description_tag=str(intent_args["description_tag"]), payload=payload,
                reservation_ids=reservations, fencing_token=token,
                approval_id=approval["decision_id"])
            if subject_kind == "run":
                self._set_job_state(connection, subject_id, JobState.LAUNCH_INTENT)
            else:
                self._set_release_state(connection, subject_id, ReleaseState.SUBMIT_INTENT)
            self._event(connection, f"{kind.upper()}_AUTHORIZED", subject_kind=subject_kind,
                        subject_id=subject_id, request_id=str(intent_args["request_id"]),
                        payload={"decision_id": approval["decision_id"],
                                 "intent_id": intent["intent_id"],
                                 "subject_digest": row["digest"],
                                 "reservation_ids": reservations,
                                 "consumed_at": consumed_at, "fencing_token": token})
            return intent

    @staticmethod
    def _intent_record(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = _json_load(result.pop("payload_json"))
        result["receipt"] = _json_load(result.pop("receipt_json"))
        result["reservation_ids"] = _json_load(result.pop("reservation_ids_json"))
        return result

    def get_intent(self, intent_id: str) -> dict[str, Any]:
        with self.transaction(immediate=False) as connection:
            return self._intent_record(
                self._require_row(connection, "intents", "intent_id", intent_id)
            )

    def list_intents(
        self, *, kind: str | None = None, states: Sequence[str | IntentState] | None = None
    ) -> list[dict[str, Any]]:
        if kind is not None and kind not in {"launch", "submission"}:
            raise ValueError("intent kind must be launch or submission")
        state_values = None if states is None else [IntentState(state).value for state in states]
        clauses: list[str] = []
        values: list[str] = []
        if kind is not None:
            clauses.append("intent_kind=?")
            values.append(kind)
        if state_values is not None:
            if not state_values:
                return []
            clauses.append("state IN (" + ",".join("?" for _ in state_values) + ")")
            values.extend(state_values)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT * FROM intents" + where + " ORDER BY created_at,intent_id", values
            ).fetchall()
            return [self._intent_record(row) for row in rows]

    def _claim_dispatch(
        self, intent_id: str, *, kind: str, subject_id: str, subject_digest: str,
        fencing_token: int, now: datetime | None = None
    ) -> dict[str, Any]:
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int) or fencing_token <= 0:
            raise DispatchError("dispatch fencing token must be a positive integer")
        with self.transaction() as connection:
            row = self._require_row(connection, "intents", "intent_id", intent_id)
            if (row["intent_kind"], row["subject_id"], row["subject_digest"],
                row["fencing_token"]) != (kind, subject_id, subject_digest, fencing_token):
                raise DispatchError("dispatch identity or fencing token does not match intent")
            if row["state"] != IntentState.PENDING.value:
                raise DispatchError(f"intent cannot be dispatched from {row['state']}")
            latest_token = int(
                connection.execute(
                    "SELECT value FROM meta WHERE key='review_fencing_token'"
                ).fetchone()[0]
            )
            if fencing_token != latest_token:
                raise DispatchError("dispatch fencing token is stale after a newer review")
            approval = self._require_row(
                connection, "approvals", "decision_id", row["approval_id"]
            )
            current = _now(now)
            if current >= _parse_time(approval["valid_until"]):
                raise DispatchError("pending intent outlived its approval dispatch window")
            if kind == "launch":
                job = self._require_row(connection, "jobs", "run_id", subject_id)
                spec = _json_load(job["spec_json"])
                if current >= _parse_time(spec["execution"]["deadline_utc"]):
                    raise DispatchError("pending launch intent reached its absolute job deadline")
            for reservation_id in _json_load(row["reservation_ids_json"]):
                reservation = self._require_row(
                    connection, "reservations", "reservation_id", reservation_id
                )
                if reservation["status"] != "ACTIVE" or reservation["intent_id"] != intent_id:
                    raise DispatchError("dispatch reservation is absent, settled, or released")
            connection.execute("UPDATE intents SET state=?,updated_at=? WHERE intent_id=?",
                               (IntentState.DISPATCHED.value, _iso(), intent_id))
            self._event(connection, "INTENT_DISPATCH_CLAIMED", subject_kind=row["subject_kind"],
                        subject_id=subject_id, request_id=row["request_id"],
                        payload={"intent_id": intent_id, "fencing_token": fencing_token})
            return self._intent_record(
                self._require_row(connection, "intents", "intent_id", intent_id)
            )

    def validate_dispatch(
        self, intent_id: str, run_id: str, run_spec_sha256: str, fencing_token: int,
        *, now: datetime | None = None,
    ) -> dict[str, Any]:
        """Atomically claim the exact launch intent immediately before worker creation."""
        return self._claim_dispatch(intent_id, kind="launch", subject_id=run_id,
                                    subject_digest=run_spec_sha256, fencing_token=fencing_token,
                                    now=now)

    def validate_submission_dispatch(
        self, intent_id: str, candidate_id: str, candidate_sha256: str, fencing_token: int,
        *, now: datetime | None = None,
    ) -> dict[str, Any]:
        return self._claim_dispatch(intent_id, kind="submission", subject_id=candidate_id,
                                    subject_digest=candidate_sha256, fencing_token=fencing_token,
                                    now=now)

    def confirm_launch(
        self, intent_id: str, *, provider_job_id: str, receipt: Mapping[str, Any]
    ) -> dict[str, Any]:
        require_resolved(provider_job_id, "provider_job_id")
        with self.transaction() as connection:
            intent = self._require_row(connection, "intents", "intent_id", intent_id)
            if intent["intent_kind"] != "launch" or intent["state"] != IntentState.DISPATCHED.value:
                raise DispatchError("only a dispatched launch intent can be confirmed")
            job = self._require_row(connection, "jobs", "run_id", intent["subject_id"])
            validate_job_transition(job["state"], JobState.RUNNING)
            connection.execute(
                "UPDATE intents SET state=?,external_id=?,receipt_json=?,updated_at=? WHERE intent_id=?",
                (IntentState.CONFIRMED.value, provider_job_id, canonical_json(receipt), _iso(), intent_id),
            )
            self._set_job_state(connection, intent["subject_id"], JobState.RUNNING)
            self._event(connection, "LAUNCH_CONFIRMED", subject_kind="run",
                        subject_id=intent["subject_id"], request_id=intent["request_id"],
                        payload={"intent_id": intent_id, "provider_job_id": provider_job_id})
        return self.get_intent(intent_id)

    def mark_intent_unknown(
        self, intent_id: str, *, receipt: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            intent = self._require_row(connection, "intents", "intent_id", intent_id)
            if intent["state"] != IntentState.DISPATCHED.value:
                raise DispatchError("only a dispatched intent can become UNKNOWN")
            connection.execute("UPDATE intents SET state=?,receipt_json=?,updated_at=? WHERE intent_id=?",
                               (IntentState.UNKNOWN.value, canonical_json(receipt or {}), _iso(), intent_id))
            if intent["intent_kind"] == "launch":
                self._set_job_state(connection, intent["subject_id"], JobState.LAUNCH_UNKNOWN)
            else:
                self._set_release_state(connection, intent["subject_id"],
                                        ReleaseState.SUBMISSION_UNKNOWN)
            # Deliberately do not settle or release reservations.
            self._event(connection, "INTENT_UNKNOWN", subject_kind=intent["subject_kind"],
                        subject_id=intent["subject_id"], request_id=intent["request_id"],
                        payload={"intent_id": intent_id,
                                 "reservations_retained": _json_load(intent["reservation_ids_json"])})
        return self.get_intent(intent_id)

    def cancel_pending_intent(
        self,
        intent_id: str,
        *,
        fencing_token: int,
        reason: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Cancel an unclaimed stale intent; a dispatched intent must be reconciled instead."""
        require_resolved(reason, "reason")
        current = _now(now)
        with self.transaction() as connection:
            self._assert_review_token(connection, fencing_token)
            intent = self._require_row(connection, "intents", "intent_id", intent_id)
            if intent["state"] != IntentState.PENDING.value:
                raise DispatchError("only an unclaimed PENDING intent can be cancelled")
            approval = self._require_row(
                connection, "approvals", "decision_id", intent["approval_id"]
            )
            stale = intent["fencing_token"] != fencing_token or current >= _parse_time(
                approval["valid_until"]
            )
            if intent["intent_kind"] == "launch":
                run = self._require_row(connection, "jobs", "run_id", intent["subject_id"])
                spec = _json_load(run["spec_json"])
                stale = stale or current >= _parse_time(spec["execution"]["deadline_utc"])
            if not stale:
                raise DispatchError("current pending intent is still dispatchable")
            for reservation_id in _json_load(intent["reservation_ids_json"]):
                reservation = self._require_row(
                    connection, "reservations", "reservation_id", reservation_id
                )
                if reservation["status"] == "ACTIVE":
                    self._release_reservation(connection, reservation_id, reason)
            connection.execute(
                "UPDATE intents SET state=?,receipt_json=?,updated_at=? WHERE intent_id=?",
                (IntentState.ABSENT.value, canonical_json({"reason": reason}), _iso(current), intent_id),
            )
            if intent["intent_kind"] == "launch":
                self._set_job_state(connection, intent["subject_id"], JobState.FAILED)
            else:
                self._set_release_state(connection, intent["subject_id"], ReleaseState.FAILED)
            self._event(
                connection,
                "PENDING_INTENT_CANCELLED",
                subject_kind=intent["subject_kind"],
                subject_id=intent["subject_id"],
                request_id=intent["request_id"],
                payload={"intent_id": intent_id, "reason": reason},
            )
        return self.get_intent(intent_id)

    def reconcile_launch(
        self, intent_id: str, *, outcome: str, provider_job_id: str | None = None,
        receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        outcome = outcome.upper()
        if outcome not in {"RUNNING", "ABSENT", "FAILED"}:
            raise ValueError("launch reconciliation outcome must be RUNNING, ABSENT, or FAILED")
        with self.transaction() as connection:
            intent = self._require_row(connection, "intents", "intent_id", intent_id)
            if intent["intent_kind"] != "launch" or intent["state"] != IntentState.UNKNOWN.value:
                raise DispatchError("launch reconciliation requires LAUNCH_UNKNOWN")
            run_id = intent["subject_id"]
            if outcome == "RUNNING":
                require_resolved(provider_job_id, "provider_job_id")
                new_intent, new_job = IntentState.CONFIRMED, JobState.RUNNING
            else:
                new_intent, new_job = (IntentState.ABSENT if outcome == "ABSENT" else IntentState.FAILED,
                                       JobState.FAILED)
                if outcome == "ABSENT":
                    for reservation_id in _json_load(intent["reservation_ids_json"]):
                        self._release_reservation(connection, reservation_id,
                                                  "remote absence confirmed during reconciliation")
            connection.execute(
                "UPDATE intents SET state=?,external_id=?,receipt_json=?,updated_at=? WHERE intent_id=?",
                (new_intent.value, provider_job_id, canonical_json(receipt or {}), _iso(), intent_id),
            )
            self._set_job_state(connection, run_id, new_job)
            self._event(connection, "LAUNCH_RECONCILED", subject_kind="run", subject_id=run_id,
                        request_id=intent["request_id"],
                        payload={"intent_id": intent_id, "outcome": outcome,
                                 "provider_job_id": provider_job_id})
        return self.get_intent(intent_id)

    def confirm_submission(
        self, intent_id: str, *, submission_id: str | int, receipt: Mapping[str, Any]
    ) -> dict[str, Any]:
        parsed_submission_id = self._positive_submission_id(submission_id)
        with self.transaction() as connection:
            intent = self._require_row(connection, "intents", "intent_id", intent_id)
            if intent["intent_kind"] != "submission" or intent["state"] != IntentState.DISPATCHED.value:
                raise DispatchError("only a dispatched submission intent can be accepted")
            self._accept_submission(connection, intent, parsed_submission_id, receipt)
        return self.get_intent(intent_id)

    @staticmethod
    def _positive_submission_id(value: str | int) -> str:
        if isinstance(value, bool):
            raise DispatchError("submission_id must be a positive integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise DispatchError("submission_id must be a positive integer") from exc
        if parsed <= 0 or str(parsed) != str(value):
            raise DispatchError("submission_id must be a positive integer")
        return str(parsed)

    def _accept_submission(
        self, connection: sqlite3.Connection, intent: sqlite3.Row,
        submission_id: str, receipt: Mapping[str, Any]
    ) -> None:
        for reservation_id in _json_load(intent["reservation_ids_json"]):
            reservation = self._require_row(connection, "reservations", "reservation_id", reservation_id)
            self._settle(connection, reservation_id, reservation["amount"])
        connection.execute(
            "UPDATE intents SET state=?,external_id=?,receipt_json=?,updated_at=? WHERE intent_id=?",
            (IntentState.CONFIRMED.value, submission_id, canonical_json(receipt), _iso(),
             intent["intent_id"]),
        )
        self._set_release_state(connection, intent["subject_id"], ReleaseState.ACCEPTED)
        self._event(connection, "SUBMISSION_ACCEPTED", subject_kind="release",
                    subject_id=intent["subject_id"], request_id=intent["request_id"],
                    payload={"intent_id": intent["intent_id"], "submission_id": submission_id})

    def reconcile_submission(
        self, intent_id: str, *, outcome: str, submission_id: str | int | None = None,
        receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        outcome = outcome.upper()
        if outcome not in {"ACCEPTED", "ABSENT", "FAILED"}:
            raise ValueError("submission outcome must be ACCEPTED, ABSENT, or FAILED")
        with self.transaction() as connection:
            intent = self._require_row(connection, "intents", "intent_id", intent_id)
            if intent["intent_kind"] != "submission" or intent["state"] != IntentState.UNKNOWN.value:
                raise DispatchError("submission reconciliation requires SUBMISSION_UNKNOWN")
            if outcome == "ACCEPTED":
                if submission_id is None:
                    raise DispatchError("submission_id must be a positive integer")
                parsed_submission_id = self._positive_submission_id(submission_id)
                self._accept_submission(connection, intent, parsed_submission_id, receipt or {})
            else:
                # FAILED can consume a remote slot; keep reservations until an explicit settlement.
                if outcome == "ABSENT":
                    for reservation_id in _json_load(intent["reservation_ids_json"]):
                        self._release_reservation(connection, reservation_id,
                                                  "remote absence confirmed during reconciliation")
                connection.execute(
                    "UPDATE intents SET state=?,receipt_json=?,updated_at=? WHERE intent_id=?",
                    ((IntentState.ABSENT if outcome == "ABSENT" else IntentState.FAILED).value,
                     canonical_json(receipt or {}), _iso(), intent_id),
                )
                self._set_release_state(connection, intent["subject_id"], ReleaseState.FAILED)
                self._event(connection, "SUBMISSION_RECONCILED", subject_kind="release",
                            subject_id=intent["subject_id"], request_id=intent["request_id"],
                            payload={"intent_id": intent_id, "outcome": outcome})
        return self.get_intent(intent_id)

    def list_events(self, *, after_sequence: int = 0) -> list[dict[str, Any]]:
        with self.transaction(immediate=False) as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE sequence>? ORDER BY sequence", (after_sequence,)
            ).fetchall()
        result = []
        for row in rows:
            record = dict(row)
            record["payload"] = _json_load(record.pop("payload_json"))
            result.append(record)
        return result

    def export_json(self, path: str | os.PathLike[str]) -> Path:
        """Atomically export one transactionally consistent, non-authoritative review snapshot."""
        destination = Path(path).resolve()
        protected = {
            self.path,
            Path(str(self.path) + "-wal"),
            Path(str(self.path) + "-shm"),
            Path(str(self.path) + "-journal"),
            self.path.with_suffix(".review.lock"),
        }
        if destination in protected:
            raise CampaignStateError("JSON export cannot overwrite the canonical store or lock files")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction(immediate=False) as connection:
            version = int(connection.execute(
                "SELECT value FROM meta WHERE key='state_version'"
            ).fetchone()[0])
            payload = {
                "schema_version": 1,
                "state_version": version,
                "exported_at": _iso(),
                "campaign": (
                    _json_load(campaign_row["document_json"])
                    if (campaign_row := connection.execute(
                        "SELECT document_json FROM campaign WHERE singleton=1"
                    ).fetchone())
                    else None
                ),
                "jobs": [self._job_record(row) for row in connection.execute(
                    "SELECT * FROM jobs ORDER BY run_id")],
                "releases": [self._release_record(row) for row in connection.execute(
                    "SELECT * FROM releases ORDER BY candidate_id")],
                "approvals": [self._approval_record(row) for row in connection.execute(
                    "SELECT * FROM approvals ORDER BY reviewed_at,decision_id")],
                "intents": [self._intent_record(row) for row in connection.execute(
                    "SELECT * FROM intents ORDER BY created_at,intent_id")],
                "ledgers": [asdict(self._budget_snapshot(connection, row[0])) for row in
                            connection.execute("SELECT ledger_id FROM ledgers ORDER BY ledger_id")],
                "reservations": [dict(row) for row in connection.execute(
                    "SELECT * FROM reservations ORDER BY created_at,reservation_id")],
                "events": [],
            }
            for row in connection.execute("SELECT * FROM events ORDER BY sequence"):
                event = dict(row)
                event["payload"] = _json_load(event.pop("payload_json"))
                payload["events"].append(event)
        # Convert Decimal fields from BudgetSnapshot before canonical encoding.
        payload["ledgers"] = [
            {key: (format(value, "f") if isinstance(value, Decimal) else value)
             for key, value in ledger.items()} for ledger in payload["ledgers"]
        ]
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
            if os.name != "nt":
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination


# Public descriptive alias; construction normally goes through store.review_lock().
ReviewLock = ReviewLease

__all__ = [
    "ApprovalError",
    "CampaignStateError",
    "CampaignStore",
    "DispatchError",
    "DuplicateIntentError",
    "ReviewLease",
    "ReviewLock",
    "ReviewLockBusy",
]
