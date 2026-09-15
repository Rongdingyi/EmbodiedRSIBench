"""Gate status vocabulary (P0-5).

BLOCKED is a first-class outcome and must never be coerced into PASS.
"""
from __future__ import annotations

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"

# release-level aggregate states (guide amendment: two-level status)
PIPELINE_PASS_WITH_BLOCKER = "PIPELINE_PASS_WITH_BLOCKER"
FULL_PILOT_PASS = "FULL_PILOT_PASS"


def combine(statuses: dict[str, str]) -> str:
    """Aggregate per-item statuses: FAIL dominates, then BLOCKED, then PASS.

    A BLOCKED item is never converted to PASS; the caller may still report a
    pipeline-level success (PIPELINE_PASS_WITH_BLOCKER) while naming the block.
    """
    values = set(statuses.values())
    if FAIL in values:
        return FAIL
    if BLOCKED in values:
        return BLOCKED
    return PASS


def release_status(primary_statuses: dict[str, str]) -> str:
    """Two-level final status used by 10_audit_release.

    FULL_PILOT_PASS  : every primary condition completed and passed its checks
    PIPELINE_PASS_WITH_BLOCKER : all completed conditions pass, some method is a
                       protocol-legal BLOCKED (recorded, never masked as PASS)
    FAIL             : any executed condition failed, or an invariant broke
    """
    if FAIL in primary_statuses.values():
        return FAIL
    if BLOCKED in primary_statuses.values():
        return PIPELINE_PASS_WITH_BLOCKER
    return FULL_PILOT_PASS


def is_pass(status: str) -> bool:
    return status == PASS


def is_blocked(status: str) -> bool:
    return status == BLOCKED
