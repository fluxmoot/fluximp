"""Month grouping of workspace conversations for the Workspaces tree.

Pure helpers with no wx dependency, so they can be unit tested without a
GUI (see ``test/tool_tests/conversation_grouping_tests.py``).

Every month decision is made in UTC, matching the timezone of
``ConversationStub.start_time``.  That value is UTC in every producer:
directory mtime converted with ``datetime.fromtimestamp(..., tz=timezone.utc)``
when a workspace is scanned, and ``datetime.now(timezone.utc)`` when a
conversation is created.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, List, Sequence, Tuple

# Day-of-month threshold that controls when the previous UTC month stops
# counting as "recent".  While the current day-of-month is below this the
# previous month is still shown un-grouped at the top of the workspace;
# from this day on it is collected into its own YYYY-MM month node.
PREVIOUS_MONTH_RECENT_DAY = 10


def _as_utc(dt: datetime) -> datetime:
    """Return *dt* as a timezone-aware UTC datetime (naive is taken as UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def month_key(dt: datetime) -> str:
    """Return the ``YYYY-MM`` month key for *dt* (interpreted as UTC)."""
    return _as_utc(dt).strftime("%Y-%m")


def month_label(key: str, count: int) -> str:
    """Return the tree label for a month node, e.g. ``2025-04 (12)``."""
    return f"{key} ({count})"


def is_recent(dt: datetime, now: datetime) -> bool:
    """Return True when *dt* belongs un-grouped at the top of the tree.

    A conversation is "recent" when it falls in the current UTC month, or
    in the previous UTC month while the current day-of-month is still
    below ``PREVIOUS_MONTH_RECENT_DAY``.
    """
    dt = _as_utc(dt)
    now = _as_utc(now)
    if (dt.year, dt.month) == (now.year, now.month):
        return True
    if now.day < PREVIOUS_MONTH_RECENT_DAY:
        previous = now.replace(day=1) - timedelta(days=1)
        if (dt.year, dt.month) == (previous.year, previous.month):
            return True
    return False


def group_conversations_by_month(
        stubs: Sequence[Any],
        now: datetime) -> Tuple[List[Any], List[Tuple[str, List[Any]]]]:
    """Split *stubs* into recent entries and descending month groups.

    Returns ``(recent, groups)`` where *recent* is the list of stubs shown
    directly under the workspace node and *groups* is a list of
    ``(month_key, [stubs])`` pairs ordered newest month first.  Every list
    is ordered newest-first, so the whole tree reads newest at the top.
    """
    now = _as_utc(now)
    ordered = sorted(stubs, key=lambda s: _as_utc(s.start_time), reverse=True)

    recent: List[Any] = []
    months: dict = {}
    order: List[str] = []
    for stub in ordered:
        dt = _as_utc(stub.start_time)
        if is_recent(dt, now):
            recent.append(stub)
        else:
            key = month_key(dt)
            if key not in months:
                months[key] = []
                order.append(key)
            months[key].append(stub)

    return recent, [(key, months[key]) for key in order]
