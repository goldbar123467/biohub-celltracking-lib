from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "review_campaign_cli_test", ROOT / "scripts/review_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


def run(
    working_directory: str,
    *,
    host: str = REVIEW.VAST_WORKER_HOST,
    provider_instance_id: object = REVIEW.VAST_PROVIDER_INSTANCE_ID,
) -> dict:
    return {
        "spec": {
            "execution": {
                "host": host,
                "provider_instance_id": provider_instance_id,
                "working_directory": working_directory,
            }
        }
    }


def test_inventory_keeps_only_configured_vast_worker_roots():
    vast_r9 = "/workspace/biohub-cell-tracking/work/campaign-code-review-20260908-r9"
    current = "/workspace/biohub-cell-tracking"
    runs = [
        run(vast_r9),
        run(vast_r9),
        run(
            "/mnt/c/Users/thecl/Documents/Biohub-Cell-Tracking/work/e0-reference/package-r3",
            host="kaggle",
            provider_instance_id=None,
        ),
        run(
            "/workspace/foreign-instance",
            provider_instance_id=123,
        ),
    ]

    roots = REVIEW._configured_vast_worker_roots(
        runs,
        run(current)["spec"],
    )

    assert roots == {vast_r9, current}


@pytest.mark.parametrize(
    ("host", "provider_instance_id"),
    [
        ("kaggle", None),
        (REVIEW.VAST_WORKER_HOST, 123),
        ("another-vast-worker", REVIEW.VAST_PROVIDER_INSTANCE_ID),
    ],
)
def test_proposed_launch_rejects_unsupported_host_or_instance(
    host: str, provider_instance_id: object
):
    proposed = run(
        "/workspace/biohub-cell-tracking",
        host=host,
        provider_instance_id=provider_instance_id,
    )["spec"]

    with pytest.raises(ValueError, match="configured existing Vast instance"):
        REVIEW._configured_vast_worker_roots([], proposed)


def test_configured_vast_root_remains_subject_to_remote_containment_check():
    outside = "/workspace/another-project"
    roots = REVIEW._configured_vast_worker_roots([run(outside)])

    assert roots == {outside}
