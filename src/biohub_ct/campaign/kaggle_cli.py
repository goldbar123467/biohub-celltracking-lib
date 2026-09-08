"""Strict adapter for the installed Kaggle CLI, using literal argument arrays.

The current CLI has no singular ``competitions submission`` command. Accepted
submissions are reconciled from account history by numeric ID or unique intent
tag. A successful submit command is not proof of a processed score.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime


class KaggleError(RuntimeError):
    pass


class SubmissionUnknown(KaggleError):
    """An external mutation may have occurred; reconcile instead of retrying."""


@dataclass(frozen=True)
class SubmissionReceipt:
    submission_id: int
    filename: str
    submitted_at_raw: str
    description: str
    status: str
    public_score: float | None
    private_score: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def _score(value: str | None, field: str) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise KaggleError(f"Invalid {field}") from exc
    if not math.isfinite(number):
        raise KaggleError(f"Nonfinite {field}")
    return number


def parse_submissions(raw: str) -> list[SubmissionReceipt]:
    reader = csv.DictReader(io.StringIO(raw.lstrip("\ufeff")))
    required = {"ref", "fileName", "date", "description", "status", "publicScore", "privateScore"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise KaggleError("Account history did not return the expected CSV schema")
    receipts = []
    seen = set()
    for row in reader:
        try:
            submission_id = int(row["ref"])
        except (ValueError, TypeError) as exc:
            raise KaggleError("Account history contains an invalid submission ID") from exc
        if submission_id <= 0 or submission_id in seen:
            raise KaggleError("Account history contains duplicate/nonpositive submission IDs")
        seen.add(submission_id)
        status = (row["status"] or "").removeprefix("SubmissionStatus.").upper()
        # Preserve unfamiliar statuses without declaring a run terminal.
        if not status or not re.fullmatch(r"[A-Z_]+", status):
            raise KaggleError("Invalid submission status")
        try:
            datetime.fromisoformat(row["date"])
        except (TypeError, ValueError) as exc:
            raise KaggleError("Invalid submission timestamp") from exc
        public = _score(row["publicScore"], "public score")
        private = _score(row["privateScore"], "private score")
        if status != "COMPLETE" and (public is not None or private is not None):
            raise KaggleError("A non-complete submission unexpectedly contains a score")
        receipts.append(SubmissionReceipt(
            submission_id, row["fileName"], row["date"], row["description"], status, public, private,
        ))
    return receipts


class KaggleCLI:
    def __init__(
        self,
        argv_prefix: list[str],
        *,
        runner: Callable = subprocess.run,
        read_timeout_seconds: float = 45,
    ) -> None:
        if not argv_prefix or any(not isinstance(a, str) or not a or "\x00" in a for a in argv_prefix):
            raise KaggleError("A verified literal Kaggle command prefix is required")
        if not math.isfinite(read_timeout_seconds) or read_timeout_seconds <= 0:
            raise KaggleError("Read timeout must be finite and positive")
        self.prefix = list(argv_prefix)
        self.runner = runner
        self.read_timeout_seconds = read_timeout_seconds

    def _read(self, args: list[str]) -> str:
        try:
            result = self.runner(
                [*self.prefix, *args], capture_output=True, text=True,
                timeout=self.read_timeout_seconds, check=False, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise KaggleError(f"Kaggle read failed: {type(exc).__name__}") from exc
        if result.returncode != 0:
            raise KaggleError(f"Kaggle read returned exit {result.returncode}")
        # Some CLI/parser paths have returned zero despite argparse errors.
        if re.search(r"(^usage:|error: (?:invalid choice|unrecognized arguments))", result.stdout + result.stderr, re.MULTILINE):
            raise KaggleError("Kaggle command syntax was rejected despite its exit status")
        return result.stdout

    def submissions(self, competition: str) -> list[SubmissionReceipt]:
        return parse_submissions(self._read(["competitions", "submissions", competition, "--csv"]))

    def submission(self, competition: str, submission_id: int) -> SubmissionReceipt | None:
        matches = [r for r in self.submissions(competition) if r.submission_id == submission_id]
        return matches[0] if matches else None

    def submission_limits(self, competition: str) -> dict:
        try:
            values = json.loads(self._read(["competitions", "submission-limits", competition, "--json"]))
        except ValueError as exc:
            raise KaggleError("Submission limits are not JSON") from exc
        if not isinstance(values, dict):
            raise KaggleError("Submission limits must be an object")
        allowed = values.get("numAllowedNow")
        if isinstance(allowed, bool) or not isinstance(allowed, int) or allowed < 0:
            raise KaggleError("Submission allowance is missing/invalid")
        # numTotal is an account/API field, not assumed to mean submissions today.
        return {"observed_at": datetime.now(UTC).isoformat(), "num_allowed_now": allowed, "raw": values}

    def gpu_quota(self) -> dict:
        try:
            rows = json.loads(self._read(["quota", "--format", "json"]))
        except ValueError as exc:
            raise KaggleError("Quota output is not JSON") from exc
        if not isinstance(rows, list):
            raise KaggleError("Quota response must be a list")
        found = [r for r in rows if isinstance(r, dict) and r.get("resource") == "GPU"]
        if len(found) != 1:
            raise KaggleError("GPU quota unavailable or ambiguous")
        row = found[0]
        parsed = {}
        for field in ("used", "remaining", "total"):
            value = row.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"\d+(?:\.\d+)?h", value):
                raise KaggleError(f"Unknown quota unit for {field}")
            parsed[field + "_hours"] = float(value[:-1])
        if parsed["remaining_hours"] > parsed["total_hours"]:
            raise KaggleError("Remaining quota exceeds total quota")
        parsed["reset_at_raw"] = row.get("refreshAt")
        parsed["observed_at"] = datetime.now(UTC).isoformat()
        parsed["raw"] = row
        return parsed

    def notebook_status(self, slug: str) -> str:
        raw = self._read(["kernels", "status", slug])
        matches = re.findall(r"KernelWorkerStatus\.([A-Z_]+)", raw)
        if len(matches) != 1:
            raise KaggleError("Notebook status was not recognized")
        return matches[0]

    def reconcile_intent(self, competition: str, unique_tag: str, filename: str) -> SubmissionReceipt | None:
        if not re.fullmatch(r"biohub-[a-f0-9]{16,64}", unique_tag):
            raise KaggleError("Intent tag must be a campaign digest tag")
        rows = [r for r in self.submissions(competition) if r.description == unique_tag]
        if len(rows) > 1:
            raise SubmissionUnknown("Multiple remote receipts share one intent tag")
        if rows and rows[0].filename != filename:
            raise SubmissionUnknown("Intent tag matched a different output filename")
        # No match is not proof of absence: account history may be delayed or
        # paginated. The controller keeps UNKNOWN/reservations until adjudicated.
        return rows[0] if rows else None

    def submit_exact(
        self,
        *,
        competition: str,
        slug: str,
        version: int,
        filename: str,
        unique_tag: str,
        verify_persisted_intent: Callable[[], None],
    ) -> dict:
        """One API attempt. The controller must persist intent BEFORE calling.

        This CLI returns remaining allowance rather than a numeric receipt.
        Always reconcile the unique tag; never automatically retry this method.
        """
        if not re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+", slug):
            raise KaggleError("Invalid exact notebook slug")
        if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
            raise KaggleError("A positive exact notebook version is required")
        if filename != "submission.csv" or not re.fullmatch(r"biohub-[a-f0-9]{16,64}", unique_tag):
            raise KaggleError("Invalid reviewed output or intent tag")
        verify_persisted_intent()
        try:
            result = self.runner(
                [*self.prefix, "competitions", "submit", competition, "-k", slug, "-v", str(version), "-f", filename, "-m", unique_tag],
                capture_output=True, text=True, timeout=60, check=False, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SubmissionUnknown(f"Submit attempt has no receipt: {type(exc).__name__}") from exc
        # Even an error exit can follow a successfully accepted server mutation.
        return {
            "attempt_finished_at": datetime.now(UTC).isoformat(),
            "returncode": result.returncode,
            "status": "RECONCILIATION_REQUIRED",
            "unique_tag": unique_tag,
            # Raw error output can include credential-bearing URLs. The command
            # result is not an authoritative receipt; retain only its hashes.
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        }
