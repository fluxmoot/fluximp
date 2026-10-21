"""Immutable fork/turn state for the FluxImp user agent.

This module owns all fork/turn calculations and derived state. It is
dependency-free: no wx, no LangGraph, no LangChain, and no GUI view-model
imports (``ConversationMessage``/``ForkTurnInfo``).
"""

from __future__ import annotations

from enum import Enum, auto
from types import MappingProxyType
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
)

__all__ = [
    "IOFactKind",
    "IOTurnStatus",
    "ForkTurn",
    "IOTurnFact",
    "RecoveryStart",
    "ForkTurnState",
    "ForkTurnStateError",
]


class IOFactKind(Enum):
    """The kind of an I/O fact."""

    IN = "in"
    OUT = "out"
    END = "end"


class IOTurnStatus(Enum):
    """The transition status of an I/O fact."""

    NOT_APPLICABLE = auto()  # IN/END placeholder
    UNPARSED = auto()  # transient OUT: filename detected, contents not yet parsed
    INCOMPLETE = auto()  # parsed OUT, no terminal marker
    COMPLETE_NORMAL = auto()  # terminal model/report output
    COMPLETE_ERROR = auto()  # terminal max-request/infrastructure output
    CANCELLED = auto()  # terminal CANCEL_TURN_ACK output
    ABANDONED = auto()  # terminal, assigned only by whole-snapshot reconciliation


# Statuses that make an OUT terminal for active/current/recovery derivations.
_TERMINAL_STATUSES = frozenset(
    {
        IOTurnStatus.COMPLETE_NORMAL,
        IOTurnStatus.COMPLETE_ERROR,
        IOTurnStatus.CANCELLED,
    }
)

# Allowed statuses by fact kind.
_IN_ALLOWED_STATUSES = frozenset(
    {IOTurnStatus.NOT_APPLICABLE, IOTurnStatus.ABANDONED}
)
_OUT_ALLOWED_STATUSES = frozenset(
    {
        IOTurnStatus.UNPARSED,
        IOTurnStatus.INCOMPLETE,
        IOTurnStatus.COMPLETE_NORMAL,
        IOTurnStatus.COMPLETE_ERROR,
        IOTurnStatus.CANCELLED,
        IOTurnStatus.ABANDONED,
    }
)
_END_ALLOWED_STATUSES = frozenset({IOTurnStatus.NOT_APPLICABLE})


class ForkTurn(NamedTuple):
    """A pure ``(fork, turn)`` identity used across fork/turn state."""

    fork: int
    turn: int


class IOTurnFact(NamedTuple):
    """A single observed I/O fact.

    ``kind`` is always an :class:`IOFactKind` member and ``status`` is always an
    :class:`IOTurnStatus` member; neither is ever a bare string or boolean.
    Fact identity is ``(fork, turn, kind)``; ``status`` is a transition field.
    """

    fork: int
    turn: int
    kind: IOFactKind
    status: IOTurnStatus

    @property
    def identity(self) -> Tuple[int, int, IOFactKind]:
        return (self.fork, self.turn, self.kind)


class RecoveryStart(NamedTuple):
    """The startup cursor produced by :meth:`ForkTurnState.compute_start_position`."""

    fork: int
    turn: int
    is_mid_turn_recovery: bool


class ForkTurnStateError(Exception):
    """Raised when fork/turn facts violate the tabular invariants."""


class ForkTurnState:
    """An immutable, validated view over a complete set of I/O facts.

    All positions and segments are derived. Construction validates the tabular
    invariants and fails fast with :class:`ForkTurnStateError`. Functional
    updates return new states; this instance is never mutated.
    """

    def __init__(self, facts: FrozenSet[IOTurnFact] = frozenset()):
        self._facts = frozenset(facts)
        self._validate()

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        # Reject conflicting identities (same identity, different status).
        seen: Dict[Tuple[int, int, IOFactKind], IOTurnStatus] = {}
        for fact in self._facts:
            if not isinstance(fact.kind, IOFactKind):
                raise ForkTurnStateError(
                    f"fact kind must be an IOFactKind member, got {fact.kind!r}"
                )
            if not isinstance(fact.status, IOTurnStatus):
                raise ForkTurnStateError(
                    f"fact status must be an IOTurnStatus member, got {fact.status!r}"
                )
            prev = seen.get(fact.identity)
            if prev is not None and prev is not fact.status:
                raise ForkTurnStateError(
                    f"conflicting statuses for identity {fact.identity}: "
                    f"{prev} vs {fact.status}"
                )
            seen[fact.identity] = fact.status

            # Allowed status by fact kind.
            if fact.kind is IOFactKind.OUT:
                if fact.status not in _OUT_ALLOWED_STATUSES:
                    raise ForkTurnStateError(
                        f"OUT fact {fact.identity} has invalid status {fact.status!r}"
                    )
            elif fact.kind is IOFactKind.END:
                if fact.status not in _END_ALLOWED_STATUSES:
                    raise ForkTurnStateError(
                        f"END fact {fact.identity} has invalid status {fact.status!r}"
                    )
            elif fact.kind is IOFactKind.IN:
                if fact.status not in _IN_ALLOWED_STATUSES:
                    raise ForkTurnStateError(
                        f"IN fact {fact.identity} has invalid status {fact.status!r}"
                    )

        in_facts = [f for f in self._facts if f.kind is IOFactKind.IN]
        out_facts = [f for f in self._facts if f.kind is IOFactKind.OUT]
        end_facts = [f for f in self._facts if f.kind is IOFactKind.END]

        # Every OUT fact must have a matching IN fact.
        in_keys = {(f.fork, f.turn) for f in in_facts}
        for out in out_facts:
            if (out.fork, out.turn) not in in_keys:
                raise ForkTurnStateError(
                    f"OUT fact {(out.fork, out.turn)} has no matching IN fact"
                )

        # Build per-fork turn sets from IN and OUT facts only (END excluded).
        fork_turns: Dict[int, set] = {}
        for fact in list(in_facts) + list(out_facts):
            fork_turns.setdefault(fact.fork, set()).add(fact.turn)

        forks_present = set(fork_turns.keys())
        if not forks_present:
            # Empty (or END-only) state: fork 0 always exists with point 0.
            self._fork_points: Dict[int, int] = {0: 0}
            self._turns_by_fork: Dict[int, FrozenSet[int]] = {0: frozenset()}
            self._latest_fork = 0
            self._end_positions = frozenset(
                ForkTurn(f.fork, f.turn) for f in end_facts
            )
            self._outs_by_fork: Dict[int, Tuple[Tuple[int, IOTurnStatus], ...]] = {}
            return

        latest_fork = max(forks_present)
        if forks_present != set(range(0, latest_fork + 1)):
            raise ForkTurnStateError(
                "fork numbering must be gap-free from 0; missing forks "
                f"{sorted(set(range(0, latest_fork + 1)) - forks_present)}"
            )

        fork_points: Dict[int, int] = {}
        turns_by_fork: Dict[int, FrozenSet[int]] = {}
        for fork in range(0, latest_fork + 1):
            turns = fork_turns[fork]
            fp = min(turns)
            if turns != set(range(fp, max(turns) + 1)):
                raise ForkTurnStateError(
                    f"fork {fork} has missing turns: {sorted(turns)}"
                )
            fork_points[fork] = fp
            turns_by_fork[fork] = frozenset(turns)

        self._fork_points = fork_points
        self._turns_by_fork = turns_by_fork
        self._latest_fork = latest_fork
        self._end_positions = frozenset(
            ForkTurn(f.fork, f.turn) for f in end_facts
        )

        outs_by_fork: Dict[int, list] = {}
        for out in out_facts:
            outs_by_fork.setdefault(out.fork, []).append((out.turn, out.status))
        self._outs_by_fork = {
            fork: tuple(entries) for fork, entries in outs_by_fork.items()
        }

    # ------------------------------------------------------------------
    # Raw immutable facts and derived state
    # ------------------------------------------------------------------
    @property
    def facts(self) -> FrozenSet[IOTurnFact]:
        return self._facts

    @property
    def fork_points(self) -> Mapping[int, int]:
        """Fork -> lowest turn with an IN fact in that fork.

        Returns an immutable internal view; callers must never mutate it.
        """
        return MappingProxyType(self._fork_points)

    @property
    def turns_by_fork(self) -> Mapping[int, FrozenSet[int]]:
        """Fork -> set of turns from IN and OUT facts only (END excluded).

        Returns an immutable internal view; callers must never mutate it.
        """
        return MappingProxyType(self._turns_by_fork)

    @property
    def latest_fork(self) -> int:
        return self._latest_fork

    @property
    def end_positions(self) -> FrozenSet[ForkTurn]:
        return self._end_positions

    @property
    def current_position(self) -> ForkTurn:
        """Highest terminal OUT, else highest non-abandoned IN, else (0, 0)."""
        terminals = [
            ForkTurn(f, t)
            for f, entries in self._outs_by_fork.items()
            for t, status in entries
            if status in _TERMINAL_STATUSES
        ]
        if terminals:
            return max(terminals)
        non_abandoned_ins = [
            ForkTurn(f.fork, f.turn)
            for f in self._facts
            if f.kind is IOFactKind.IN and f.status is not IOTurnStatus.ABANDONED
        ]
        if non_abandoned_ins:
            return max(non_abandoned_ins)
        return ForkTurn(0, 0)

    @property
    def current_fork(self) -> int:
        return self.current_position.fork

    @property
    def current_turn(self) -> int:
        return self.current_position.turn

    @property
    def active_position(self) -> ForkTurn:
        """Highest unfinished, non-superseded, non-abandoned IN, else current."""
        candidates = self._unfinished_non_superseded_ins()
        if candidates:
            return max(candidates)
        return self.current_position

    @property
    def active_fork(self) -> int:
        return self.active_position.fork

    @property
    def active_turn(self) -> int:
        return self.active_position.turn

    @property
    def earliest_unfinished_input(self) -> ForkTurn:
        tier = self._unfinished_input_tiers()
        if tier[0]:
            return min(tier[0])
        if tier[1]:
            return min(tier[1])
        return ForkTurn(self.current_fork, self.next_turn(self.current_fork))

    @property
    def latest_unfinished_input(self) -> ForkTurn:
        tier = self._unfinished_input_tiers()
        if tier[0]:
            return max(tier[0])
        if tier[1]:
            return max(tier[1])
        return ForkTurn(self.current_fork, self.next_turn(self.current_fork))

    # ------------------------------------------------------------------
    # Calculations
    # ------------------------------------------------------------------
    def next_turn(self, fork: int) -> int:
        """Next turn in fork ``fork``: empty fork 0 -> 0; empty fork >0 raises."""
        turns = self._turns_by_fork.get(fork)
        if turns is None or not turns:
            if fork == 0:
                return 0
            raise ForkTurnStateError(
                f"cannot compute next_turn for empty fork {fork} (no IN fact)"
            )
        return max(turns) + 1

    def owning_fork(self, turn: int, view_fork: int) -> Optional[int]:
        """Rightmost fork ``f <= view_fork`` with ``turn >= fork_point(f)``."""
        result: Optional[int] = None
        for f in range(0, view_fork + 1):
            fp = self._fork_points.get(f)
            if fp is not None and turn >= fp:
                result = f
        return result

    def is_visible(self, msg_fork: int, msg_turn: int, view_fork: int) -> bool:
        if msg_fork > view_fork:
            raise ForkTurnStateError(
                f"msg_fork {msg_fork} > view_fork {view_fork}"
            )
        owner = self.owning_fork(msg_turn, view_fork)
        if owner is None:
            return True
        return owner == msg_fork

    def is_visible_with_extra_fork(
        self,
        msg_fork: int,
        msg_turn: int,
        new_fork: int,
        new_fork_point: int,
    ) -> bool:
        """Fork-mode preview with a hypothetical fork point (no mutation)."""
        if msg_fork > new_fork:
            raise ForkTurnStateError(
                f"msg_fork {msg_fork} > new_fork {new_fork}"
            )
        owner: Optional[int] = None
        for f in range(0, new_fork + 1):
            if f == new_fork:
                fp = new_fork_point
            else:
                fp = self._fork_points.get(f)
            if fp is not None and msg_turn >= fp:
                owner = f
        if owner is None:
            return True
        return owner == msg_fork

    def visible_segments(self, view_fork: int) -> frozenset:
        """"Go up, go left" tracing; returns ``(fork, start_turn, end_turn)``."""
        turns = self._turns_by_fork.get(view_fork)
        if turns is None or not turns:
            return frozenset()
        segments = []
        fork = view_fork
        turn = max(turns)
        while True:
            t = turn
            fork_turns = self._turns_by_fork.get(fork, frozenset())
            while t >= 0 and t in fork_turns:
                t -= 1
            segments.append((fork, t + 1, turn))
            next_fork: Optional[int] = None
            for f in range(fork - 1, -1, -1):
                if t in self._turns_by_fork.get(f, frozenset()):
                    next_fork = f
                    break
            if next_fork is None:
                break
            fork = next_fork
            turn = t
        return frozenset(segments)

    # ------------------------------------------------------------------
    # Linear projections (chronological views over the visible history)
    # ------------------------------------------------------------------
    def linear_turns(self, view_fork: int) -> Tuple[ForkTurn, ...]:
        """Visible ``(fork, turn)`` positions in chronological order.

        Chronological order is oldest turn first: :meth:`visible_segments`
        segments are sorted by start turn and expanded into individual turns.
        Because the "go up, go left" trace always steps left into strictly
        older (lower) turns, segment start turns increase oldest-first, so
        this equals ascending turn order for coherent layouts and stays a
        well-defined total order even for the permissive cross-fork layouts
        the constructor accepts.
        """
        segments = sorted(
            self.visible_segments(view_fork), key=lambda seg: seg[1]
        )
        return tuple(
            ForkTurn(fork, turn)
            for fork, start, end in segments
            for turn in range(start, end + 1)
        )

    def linear_facts(self, view_fork: int) -> Tuple[IOTurnFact, ...]:
        """Visible IN/OUT/END facts in chronological order.

        Ascending turn; within one turn coordinate an ``END`` fact (a gate
        before the turn begins) precedes its ``IN`` fact, which precedes its
        ``OUT`` fact(s). Exactly one ``OUT`` per position is enforced by
        construction, so the observation-order clause for multiple ``OUT`` s
        is currently vacuous; the sort is stable regardless.
        """
        return tuple(self._linear_visible_facts(view_fork, include_end=True))

    def api_request_facts(
        self, fork: Optional[int] = None
    ) -> Tuple[IOTurnFact, ...]:
        """Facts for the linear model-request message path.

        Defaults to ``linear_facts(active_fork)`` (fallback current_fork) and excludes
        ``END`` facts (the end signal must never enter a model request).
        """
        if fork is None:
            fork = (
                self.active_fork
                if self.active_fork is not None
                else self.current_fork
            )
        return tuple(self._linear_visible_facts(fork, include_end=False))

    def _linear_visible_facts(
        self, view_fork: int, include_end: bool
    ) -> Tuple[IOTurnFact, ...]:
        """IN/OUT/END facts for visible positions, chronological and sorted."""
        positions = set(self.linear_turns(view_fork))
        selected = []
        for fact in self._facts:
            if fact.kind is IOFactKind.END:
                # END facts follow the same visibility rule as every other
                # position: an end signal on a superseded branch (its
                # (fork, turn) not visible in this view) must not leak into
                # the view's linear history. is_visible is used instead of
                # the IN/OUT position set because an END fact is a gate
                # before a turn begins and normally has no IN/OUT facts at
                # its own position.
                if (include_end
                        and fact.fork <= view_fork
                        and self.is_visible(fact.fork, fact.turn, view_fork)):
                    selected.append(fact)
            elif (fact.fork, fact.turn) in positions:
                selected.append(fact)
        kind_rank = {
            IOFactKind.END: 0,
            IOFactKind.IN: 1,
            IOFactKind.OUT: 2,
        }
        selected.sort(key=lambda f: (f.turn, f.fork, kind_rank[f.kind]))
        return tuple(selected)

    def end_position(self) -> ForkTurn:
        """The next turn after the active position (the ``.end`` signal)."""
        active = self.active_position
        return ForkTurn(active.fork, active.turn + 1)

    def recovery_position(self) -> Optional[ForkTurn]:
        """Highest terminal OUT; sentinel ``(0, -1)`` if OUTs exist but none
        terminal; ``None`` if there are no OUT facts."""
        terminals = [
            ForkTurn(f, t)
            for f, entries in self._outs_by_fork.items()
            for t, status in entries
            if status in _TERMINAL_STATUSES
        ]
        if terminals:
            return max(terminals)
        if self._outs_by_fork:
            return ForkTurn(0, -1)
        return None

    def compute_start_position(
        self, checkpoint_fork: int, checkpoint_turn: int
    ) -> RecoveryStart:
        """Classify startup against :meth:`recovery_position`, preserving the
        raise/check order of today's ``main.py`` startup."""
        recovered = self.recovery_position()
        if recovered is None:
            raise ForkTurnStateError(
                "cannot compute start position without a recovery position"
            )
        file_fork, file_turn = recovered.fork, recovered.turn

        # (1) Fork-ahead error: output ahead of checkpoint.
        if file_fork > checkpoint_fork:
            raise ForkTurnStateError(
                f"recovery position {recovered} is ahead of checkpoint "
                f"({checkpoint_fork}, {checkpoint_turn})"
            )
        # (2) Mid-turn recovery.
        if checkpoint_fork > file_fork or checkpoint_turn > file_turn:
            return RecoveryStart(checkpoint_fork, checkpoint_turn, True)
        # (3) Normal recovery.
        if checkpoint_fork == file_fork and checkpoint_turn == file_turn:
            return RecoveryStart(file_fork, file_turn + 1, False)
        # (4) Residual same-fork behind error.
        raise ForkTurnStateError(
            f"checkpoint ({checkpoint_fork}, {checkpoint_turn}) is strictly "
            f"behind recovery position {recovered} in the same fork"
        )

    # ------------------------------------------------------------------
    # Functional status updates (return NEW states; never mutate self)
    # ------------------------------------------------------------------
    def with_fact(self, fact: IOTurnFact) -> "ForkTurnState":
        """Add a new IN/END fact. Raises for OUT facts or existing identities."""
        if fact.kind is IOFactKind.OUT:
            raise ForkTurnStateError(
                "with_fact() does not accept OUT facts; use with_out_upsert()"
            )
        if any(f.identity == fact.identity for f in self._facts):
            raise ForkTurnStateError(
                f"identity {fact.identity} already exists in the snapshot"
            )
        return ForkTurnState(self._facts | {fact})

    def with_facts(self, facts: Iterable[IOTurnFact]) -> "ForkTurnState":
        """Add several IN/END facts, raising on the first invalid one."""
        state = self
        for fact in facts:
            state = state.with_fact(fact)
        return state

    def with_out_upsert(
        self, fork: int, turn: int, status: IOTurnStatus
    ) -> "ForkTurnState":
        """Maintain exactly one OUT per ``(fork, turn)`` with a live status
        transition machine. Requires a matching IN; rejects ABANDONED."""
        if status is IOTurnStatus.ABANDONED:
            raise ForkTurnStateError(
                "ABANDONED is not a direct file-observation transition"
            )
        if status is IOTurnStatus.NOT_APPLICABLE:
            raise ForkTurnStateError(
                "NOT_APPLICABLE is not a valid OUT status"
            )
        if not any(
            f.kind is IOFactKind.IN and f.fork == fork and f.turn == turn
            for f in self._facts
        ):
            raise ForkTurnStateError(
                f"OUT upsert ({fork}, {turn}) requires a matching IN fact"
            )

        existing_status: Optional[IOTurnStatus] = None
        for f in self._facts:
            if (
                f.kind is IOFactKind.OUT
                and f.fork == fork
                and f.turn == turn
            ):
                existing_status = f.status
                break

        allowed = _OUT_TRANSITIONS[existing_status]
        if status not in allowed:
            raise ForkTurnStateError(
                f"invalid OUT status transition for ({fork}, {turn}): "
                f"{existing_status} -> {status}"
            )

        if existing_status is None:
            new_facts = self._facts | {
                IOTurnFact(fork, turn, IOFactKind.OUT, status)
            }
        else:
            new_facts = frozenset(
                IOTurnFact(fork, turn, IOFactKind.OUT, status)
                if f.kind is IOFactKind.OUT and f.fork == fork and f.turn == turn
                else f
                for f in self._facts
            )
        return ForkTurnState(new_facts)

    def reconcile_snapshot(self) -> "ForkTurnState":
        """Whole-snapshot reconciliation: assign ABANDONED to superseded
        unfinished turns. Never changes terminal statuses."""
        if self._latest_fork == 0 and not self._turns_by_fork.get(0):
            return self
        changed = False
        new_facts = set(self._facts)
        for fact in list(self._facts):
            if fact.kind is not IOFactKind.IN:
                continue
            if fact.status is IOTurnStatus.ABANDONED:
                continue
            if self._is_output_terminal(fact.fork, fact.turn):
                continue  # never abandon a turn with a terminal OUT
            if not self._is_superseded_for_abandonment(fact.fork, fact.turn):
                continue
            changed = True
            new_facts.discard(fact)
            new_facts.add(
                IOTurnFact(
                    fact.fork, fact.turn, IOFactKind.IN, IOTurnStatus.ABANDONED
                )
            )
            out = self._find_out(fact.fork, fact.turn)
            if out is not None and out.status is not IOTurnStatus.ABANDONED:
                changed = True
                new_facts.discard(out)
                new_facts.add(
                    IOTurnFact(
                        out.fork,
                        out.turn,
                        IOFactKind.OUT,
                        IOTurnStatus.ABANDONED,
                    )
                )
        if not changed:
            return self
        return ForkTurnState(frozenset(new_facts))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _is_output_terminal(self, fork: int, turn: int) -> bool:
        for t, status in self._outs_by_fork.get(fork, ()):
            if t == turn and status in _TERMINAL_STATUSES:
                return True
        return False

    def _find_out(self, fork: int, turn: int) -> Optional[IOTurnFact]:
        for f in self._facts:
            if (
                f.kind is IOFactKind.OUT
                and f.fork == fork
                and f.turn == turn
            ):
                return f
        return None

    def _is_superseded_for_properties(self, fork: int, turn: int) -> bool:
        """Property supersession: terminal OUT later in same fork, or a higher
        fork whose fork point owns the turn."""
        for t, status in self._outs_by_fork.get(fork, ()):
            if t > turn and status in _TERMINAL_STATUSES:
                return True
        for f, fp in self._fork_points.items():
            if f > fork and fp <= turn:
                return True
        return False

    def _is_superseded_for_abandonment(self, fork: int, turn: int) -> bool:
        """Reconciliation supersession: a later IN in the same fork, or a
        higher fork whose fork point owns the turn."""
        for f in self._facts:
            if f.kind is not IOFactKind.IN:
                continue
            if f.fork == fork and f.turn > turn:
                return True
        for f, fp in self._fork_points.items():
            if f > fork and fp <= turn:
                return True
        return False

    def _unfinished_non_superseded_ins(self):
        """Unfinished, non-superseded, non-abandoned IN positions (property
        supersession rule)."""
        result = []
        for f in self._facts:
            if f.kind is not IOFactKind.IN:
                continue
            if f.status is IOTurnStatus.ABANDONED:
                continue
            if self._is_output_terminal(f.fork, f.turn):
                continue
            if self._is_superseded_for_properties(f.fork, f.turn):
                continue
            result.append(ForkTurn(f.fork, f.turn))
        return result

    def _unfinished_input_tiers(self):
        """(tier1, tier2) unfinished non-superseded non-abandoned IN positions.

        Tier 1 requires an ``OUT(..., INCOMPLETE)`` at the own position.
        """
        tier1 = []
        tier2 = []
        for f in self._facts:
            if f.kind is not IOFactKind.IN:
                continue
            if f.status is IOTurnStatus.ABANDONED:
                continue
            if self._is_output_terminal(f.fork, f.turn):
                continue
            if self._is_superseded_for_properties(f.fork, f.turn):
                continue
            pos = ForkTurn(f.fork, f.turn)
            out_status = None
            for t, status in self._outs_by_fork.get(f.fork, ()):
                if t == f.turn:
                    out_status = status
                    break
            if out_status is IOTurnStatus.INCOMPLETE:
                tier1.append(pos)
            else:
                tier2.append(pos)
        return tier1, tier2


# Live file-status transition machine keyed by the existing OUT status
# (None = no existing OUT fact).
_OUT_TRANSITIONS: Dict[Optional[IOTurnStatus], frozenset] = {
    None: frozenset(
        {
            IOTurnStatus.UNPARSED,
            IOTurnStatus.INCOMPLETE,
            IOTurnStatus.COMPLETE_NORMAL,
            IOTurnStatus.COMPLETE_ERROR,
            IOTurnStatus.CANCELLED,
        }
    ),
    IOTurnStatus.UNPARSED: frozenset(
        {
            IOTurnStatus.INCOMPLETE,
            IOTurnStatus.COMPLETE_NORMAL,
            IOTurnStatus.COMPLETE_ERROR,
            IOTurnStatus.CANCELLED,
        }
    ),
    IOTurnStatus.INCOMPLETE: frozenset(
        {
            IOTurnStatus.COMPLETE_NORMAL,
            IOTurnStatus.COMPLETE_ERROR,
            IOTurnStatus.CANCELLED,
        }
    ),
    IOTurnStatus.COMPLETE_NORMAL: frozenset(),
    IOTurnStatus.COMPLETE_ERROR: frozenset(),
    IOTurnStatus.CANCELLED: frozenset(),
    IOTurnStatus.ABANDONED: frozenset(),
}
