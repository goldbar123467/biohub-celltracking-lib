"""Finite resource-ledger value objects and a small store-bound facade."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from biohub_ct.campaign.state import CampaignStore


class BudgetError(RuntimeError):
    """Base class for invalid or unsafe budget operations."""


class UnknownBudgetError(BudgetError):
    """The numerical authorization needed for admission is unresolved."""


class BudgetExceeded(BudgetError):
    """A reservation would exceed the finite authorization."""


class ReservationError(BudgetError):
    """A reservation does not exist or cannot make the requested transition."""


def decimal_amount(value: Any, field: str, *, allow_zero: bool = True) -> Decimal:
    """Parse a finite, non-negative amount without binary float arithmetic."""

    if isinstance(value, bool) or value is None:
        raise BudgetError(f"{field} must be a finite non-negative number")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise BudgetError(f"{field} must be a finite non-negative number") from exc
    if not amount.is_finite() or amount < 0 or (not allow_zero and amount == 0):
        qualifier = "positive" if not allow_zero else "non-negative"
        raise BudgetError(f"{field} must be a finite {qualifier} number")
    return amount


def decimal_text(value: Any, field: str, *, allow_zero: bool = True) -> str:
    amount = decimal_amount(value, field, allow_zero=allow_zero)
    if amount == 0:
        return "0"
    return format(amount.normalize(), "f")


@dataclass(frozen=True)
class BudgetSnapshot:
    ledger_id: str
    resource_scope: str
    unit: str
    currency: str | None
    authorized_total: Decimal | None
    confirmed_spend: Decimal
    unreconciled_spend: Decimal
    protected_reserve: Decimal
    outstanding_reservations: Decimal
    available: Decimal | None
    observation_time: str | None

    def as_json(self) -> dict[str, str | None]:
        return {
            key: (format(value, "f") if isinstance(value, Decimal) else value)
            for key, value in asdict(self).items()
        }


class BudgetLedger:
    """Convenience facade over one ledger persisted by :class:`CampaignStore`."""

    def __init__(self, store: CampaignStore, ledger_id: str):
        self.store = store
        self.ledger_id = ledger_id

    @classmethod
    def create(
        cls,
        store: CampaignStore,
        ledger_id: str,
        *,
        resource_scope: str,
        unit: str,
        authorized_total: Any | None,
        currency: str | None = None,
        confirmed_spend: Any = 0,
        unreconciled_spend: Any = 0,
        protected_reserve: Any = 0,
        observation_time: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BudgetLedger:
        store.create_ledger(
            ledger_id,
            resource_scope=resource_scope,
            unit=unit,
            authorized_total=authorized_total,
            currency=currency,
            confirmed_spend=confirmed_spend,
            unreconciled_spend=unreconciled_spend,
            protected_reserve=protected_reserve,
            observation_time=observation_time,
            metadata=metadata,
        )
        return cls(store, ledger_id)

    def snapshot(self) -> BudgetSnapshot:
        return self.store.budget_snapshot(self.ledger_id)

    def reserve(
        self,
        reservation_id: str,
        *,
        subject_kind: str,
        subject_id: str,
        amount: Any,
        allow_protected: bool = False,
        note: str | None = None,
    ) -> dict[str, Any]:
        return self.store.reserve_budget(
            self.ledger_id,
            reservation_id,
            subject_kind=subject_kind,
            subject_id=subject_id,
            amount=amount,
            allow_protected=allow_protected,
            note=note,
        )

    def settle(self, reservation_id: str, actual_amount: Any) -> dict[str, Any]:
        return self.store.settle_reservation(reservation_id, actual_amount)

    def release(self, reservation_id: str, *, reason: str) -> dict[str, Any]:
        return self.store.release_reservation(reservation_id, reason=reason)


__all__ = [
    "BudgetError",
    "BudgetExceeded",
    "BudgetLedger",
    "BudgetSnapshot",
    "ReservationError",
    "UnknownBudgetError",
    "decimal_amount",
    "decimal_text",
]
