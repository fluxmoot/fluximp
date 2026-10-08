"""WorkflowHistoryController - item model, routing, batching, and policies.

Phase 1 controller core: routes conversation events into an ordered item
model, schedules batched layouts on the WorkflowHistoryView, and applies
the scroll/width/expansion policies.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, List, Tuple, Any

from fluximp_protocol.user_agent.fork_turn_state import ForkTurn, IOFactKind
from fluximp_wxwidgets.conversation_manager import (
    Conversation, ConversationMessage, ConversationStatus,
)
from fluximp_wxwidgets.conditional_sequence import load_all_states


try:
    import wx
    _CALL_AFTER = wx.CallAfter
except Exception:
    _CALL_AFTER = None


# --- Sizer-crash guard layer -------------------------------------------------
# wx assertion failures cross into Python as ``wx.wxAssertionError`` (Phoenix
# builds; ``PyAssertionError`` in classic) — an *AssertionError* subclass, NOT
# RuntimeError. The original reload-desync crash escaped every ``except
# RuntimeError`` guard in the flush/width chain, so each guard that must
# survive a native layout/mixin assertion (or a mid-flight swallowed error
# turning into one) catches WX_FAILURES instead. Guards keep logging loudly:
# nothing may fail silently again.
# ``wx`` may be absent entirely (a headless import leaves the name unbound
# through the try block above); any lookup failure falls back to plain
# AssertionError, so WX_FAILURES always at least catches RuntimeError and
# AssertionError.
try:
    _WX_ASSERTION_ERROR = (getattr(wx, 'wxAssertionError', None)
                           or getattr(wx, 'PyAssertionError', None)
                           or AssertionError)
except Exception:
    _WX_ASSERTION_ERROR = AssertionError
WX_FAILURES = (RuntimeError, _WX_ASSERTION_ERROR)


@dataclass(frozen=True)
class HistoryItemId:
    """Identity of a history item (kind discriminated)."""
    kind: str  # 'card', 'panel', 'error_panel'
    fork: int
    turn: int
    agent_name: str = ""
    run_id: str = ""
    seq: int = 0


@dataclass
class HistoryItem:
    """A rendered history item: a RequestCard, MessagePanel, or error panel."""
    item_id: HistoryItemId
    messages: List[ConversationMessage] = field(default_factory=list)
    # §8.3 per-instance expansion overrides, keyed (role, ordinal) —
    # the ordinal keeps same-role sections on one card independent
    # (defect B1: the role-only key cascaded a click to every sibling).
    user_overrides: Dict[Tuple[str, int], bool] = field(default_factory=dict)
    permission_pending: bool = False
    pending_muted: bool = False
    pending: bool = False
    last_y: int = 0
    last_height: int = 0
    measured_at_width: int = 0
    exact: bool = False
    force_collapsed: bool = False  # collapse_content() freed HWNDs on this card

    # Display/derived fields
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_conditional: bool = False
    conditional_agent_name: str = ""
    is_system_agent: bool = False
    system_agent_type: str = ""
    fork: int = 0
    turn: int = 0
    error_text: str = ""
    message: Optional[ConversationMessage] = None


@dataclass
class ScrollState:
    """Captured scroll geometry used by the scroll policy (§8.1).

    One px contract (F-B2): ``view_start`` is the scroll origin in
    DEVICE PIXELS — the same space scroll_to() consumes; Scroll()
    itself takes scroll units, so a unit-space capture restored as
    though it were px landed at a fraction of the position.
    ``pixels_below`` and ``anchor_offset`` are px as well.
    """
    mode: str = "bottom"  # capture label; consumers key on geometry
    view_start: Tuple[int, int] = (0, 0)   # device px (F-B2)
    client_px: Tuple[int, int] = (0, 0)
    virtual_px: Tuple[int, int] = (0, 0)
    scroll_unit: Tuple[int, int] = (0, 0)
    pixels_below: int = 0
    anchor_item_id: Optional[HistoryItemId] = None
    anchor_offset: int = 0


class WorkflowHistoryController:
    """Routes conversation events into the history item model and view."""

    MIN_CARD_WIDTH = 520
    # §8.2: 25px left indent for conditional-agent / system-agent cards.
    # The view applies the policy (indented widgets are sized
    # card_w - INDENT_OFFSET, the indent is the sizer-row border); the
    # controller keeps the constant as the §8.2 policy document.
    INDENT_OFFSET = 25
    BOTTOM_TOLERANCE_PX = 24
    PADDING = 3
    MEASURE_CHUNK = 50
    DEFAULT_ESTIMATE_HEIGHT = 120

    # F-A7 (mid-drag fling): a drag-settle turn builds live bands only while
    # the number of brand-new widgets it would build stays inside
    # DRAG_BUILD_BUDGET — building whole visible sets per turn is what
    # starves WM_PAINT (it is dispatched only when the event queue
    # empties) and turns a mid-list drag grey with shells flashing at
    # their native-creation position. Over-budget turns virtualize
    # through the swept area and converge at the first park / the release
    # settle. DRAG_TURN_BUDGET_MS latches the same gate for an
    # in-budget-but-slow turn (tall cards cost more than they count).
    DRAG_BUILD_BUDGET = 4
    DRAG_TURN_BUDGET_MS = 30.0

    def __init__(self, conversation: Conversation, view=None,
                 file_watcher=None,
                 permission_response_handler: Optional[Callable] = None,
                 logger: Optional[logging.Logger] = None):
        self.conversation = conversation
        self._logger = logger or logging.getLogger(__name__)
        self._view = view
        self._tab = None  # Bound by set_tab() from the conversation tab
        self._file_watcher = file_watcher
        self._permission_response_handler = permission_response_handler

        self._items: List[HistoryItem] = []
        self._items_by_id: Dict[HistoryItemId, HistoryItem] = {}
        self._pending_permissions: Dict[Tuple[int, int], set] = {}
        self._batch_depth = 0
        self._flush_pending = False
        self.load_in_progress = False
        self.non_resumable = False
        # §7.1 step-0 sequence-state glob results, held for §7.4 step 8 keyed
        # by each state's own state.event_key (§7.7).
        self._loaded_sequence_states: Dict[str, object] = {}
        self._user_scrolled = False
        self._fork_preview = False
        self._fork_preview_fork = None
        self._fork_preview_point = 0

        self._current_cards: Dict[Tuple[int, int, int], HistoryItemId] = {}
        self._conditional_cards: Dict[Tuple[str, int], HistoryItemId] = {}
        self._system_agent_cards: Dict[Tuple[int, int, str, int], HistoryItemId] = {}
        self._next_seq = 0

        self._call_after = _CALL_AFTER or (lambda fn: fn())

        # §8.5 virtualization / geometry state
        self._height_stats: Dict[int, Dict[str, int]] = {}
        self._effective_card_width = 0
        self._force_scroll_bottom = False
        # One-shot reload head paint (§5.10.1): the first load flush renders
        # the first item once; later streaming flushes skip view work.
        self._load_first_paint_done = False
        self._last_scroll_pos = None
        self._use_line_count_heuristic = False
        self._measure_all_index = 0
        self._measure_all_on_complete: Optional[Callable] = None
        self._measure_all_pre_state: Optional[ScrollState] = None
        self._remeasure_index = 0
        self._remeasure_width = 0
        # §8.1 width-pass fix: the resize keep-bottom pin (armed by a
        # live width pass, consumed by the convergence finale, cleared
        # by any user scroll, a drag, or a reload bracket) and the entry
        # capture consumed inside on_width_changed.
        self._remeasure_pin = False
        self._resize_pre_state = None
        # F-A7 (mid-drag fling gate): latched when a drag-settle turn goes
        # over the build budget (_thumb_drag_gated) or blows the per-turn
        # time budget while the thumb is held; cleared at gesture start
        # (start_thumb_drag), at the parked-thumb settle (park_thumb_drag),
        # and at the release settle — the three points every gesture
        # converges at.
        self._drag_fling_gated = False
        # Windowed reload geometry cache: compute_visible_range() records the
        # total estimated height so load-mode callers skip a second O(n) pass.
        self._last_range_total = 0
        self._last_range_width = 0

        if self._view is not None:
            self._view.set_permission_response_handler(permission_response_handler)

    # ------------------------------------------------------------------
    # Item-model helpers
    # ------------------------------------------------------------------

    def _add_item(self, item: HistoryItem):
        if item.item_id in self._items_by_id:
            # F-B4 (2026-10-07 11:59 desync storm): a duplicate
            # HistoryItemId cannot be added — two items sharing one id
            # double-insert into every sizer bookkeeping structure and
            # re-create the order/native phantom every rebuild cycle
            # (the 32/33 -> 31/33 storm). Keep the FIRST instance (all
            # consumers already track it); the refusal is logged loudly
            # with full context so the duplicate-producing routing
            # branch gets pinned on the next repro (the correct
            # merge/update semantics is the follow-up fix once the logs
            # name the branch).
            self._logger.error(
                "F-B4: duplicate HistoryItemId routed (existing kept, "
                "add refused): %r", item.item_id)
            return
        self._items.append(item)
        self._items_by_id[item.item_id] = item

    def _remove_item(self, item: Optional[HistoryItem]):
        if item is None:
            return
        self._items_by_id.pop(item.item_id, None)
        if item in self._items:
            self._items.remove(item)
        # Drop the item's contribution from the running height mean.
        if item.measured_at_width > 0 and item.last_height > 0:
            stats = self._height_stats.get(item.measured_at_width)
            if stats and stats.get('count'):
                stats['sum'] = max(0, stats['sum'] - item.last_height)
                stats['count'] = max(0, stats['count'] - 1)
        if self._view is not None:
            try:
                self._view.remove_widget(item.item_id)
            except WX_FAILURES:
                pass

    def _widget(self, item: HistoryItem):
        if self._view is None:
            return None
        try:
            return self._view.get_widget(item.item_id)
        except RuntimeError:
            return None

    def _materialize(self, item: HistoryItem):
        if self._view is None:
            return None
        # §8.5 windowed reload: while a backlog streams, defer widget builds
        # to the flush's load-window apply (apply_load_window -> materialize
        # -> _populate_widget_content replay). Eagerly building a widget for
        # every routed backlog message would churn native widgets before the
        # first visible layout and orphan anything outside the padded window.
        # Live routing is unaffected (load_in_progress is False there).
        if self.load_in_progress:
            return None
        try:
            widget = self._view.materialize(item)
        except RuntimeError:
            return None
        if widget is None:
            return None
        if item.item_id.kind == 'card' and hasattr(widget, 'iter_sections'):
            try:
                self.apply_expansion_policy(item, widget)
            except Exception:
                pass
        return widget

    def _mark_mutated_member(self, item_id: HistoryItemId):
        """F-A10 follow-up: flag an item id after an out-of-band content
        mutation landed on its widget — routing responses / tool uses /
        permissions, conditional and system-agent cards, report
        sections, pending mutes, and the request-start HWND-freeing
        collapse pass. Every site that reads a card through _widget()
        and mutates it outside the view's own materialize path routes
        through here.

        A HIDDEN detached-cache member is skipped by the per-flush
        measure/width passes (the cache's idle contract), so its
        recorded height — and with it the item's offscreen spacer —
        would hold the stale pre-mutation value until some later
        revisit re-Added the card. The flag converges it exactly once
        on the next flush's refresh pass (one hidden measure, then it
        self-clears). No-op for anything but a member: live rows are
        re-measured every flush anyway. Stub views without the API are
        tolerated — the test fakes route through these same handlers.
        """
        if self._view is None:
            return
        try:
            self._view.mark_out_of_band_mutation(item_id)
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    # Routing handlers (§5)
    # ------------------------------------------------------------------

    def on_request_started(self, fork: int, turn: int, metadata: dict):
        """Create a pending card item for a newly started API request."""
        is_cond = metadata.get('is_conditional', False)
        request_seq = metadata.get('request_seq', 0)

        if is_cond:
            run_id = metadata.get('run_id', '')
            agent = metadata.get('conditional_agent_name', '')
            a_fork = metadata.get('anchor_fork', fork)
            a_turn = metadata.get('anchor_turn', turn)
            key = (run_id, request_seq)
            old = self._conditional_cards.pop(key, None)
            if old:
                self._remove_item(self._items_by_id.get(old))
            item = HistoryItem(
                item_id=HistoryItemId('card', a_fork, a_turn,
                                      agent_name=agent, run_id=run_id, seq=request_seq),
                fork=a_fork, turn=a_turn, metadata=dict(metadata),
                is_conditional=True, conditional_agent_name=agent, pending=True)
            self._add_item(item)
            self._conditional_cards[key] = item.item_id
            self._materialize(item)
        else:
            key = (fork, turn, request_seq)
            old = self._current_cards.pop(key, None)
            if old:
                self._remove_item(self._items_by_id.get(old))
            # Retry supersede (§5.2): a new REQUEST at the same (fork, turn)
            # with an advanced request_seq drops any stale still-pending card
            # at those coordinates (agent retry), matching the old
            # _create_request_card cleanup of obsolete pending cards.
            stale = []
            for (k_fork, k_turn, k_seq), iid in list(self._current_cards.items()):
                if (k_fork, k_turn) == (fork, turn) and k_seq != request_seq:
                    stale_item = self._items_by_id.get(iid)
                    if stale_item is not None and stale_item.pending:
                        stale.append((k_fork, k_turn, k_seq))
                        self._remove_item(stale_item)
            for sk in stale:
                self._current_cards.pop(sk, None)
            item = HistoryItem(
                item_id=HistoryItemId('card', fork, turn, seq=request_seq),
                fork=fork, turn=turn, metadata=dict(metadata), pending=True)
            self._add_item(item)
            self._current_cards[key] = item.item_id
            self._materialize(item)
        card = self._widget(item)
        if card is not None:
            try:
                card.set_pending()
            except RuntimeError:
                pass

        # Free HWNDs by collapsing previous cards' content sections. During
        # a reload no cards are materialized (frozen head display), so the
        # O(n) flag pass and the collapse pass are pure churn — skip them;
        # the expansion-policy defaults settle each card at finalization.
        # Live routing keeps the full pass.
        if not self.load_in_progress:
            for other in self._items:
                other.force_collapsed = True
            item.force_collapsed = False
            if self._view is not None:
                for iid in self._view.materialized_item_ids():
                    other = self._items_by_id.get(iid)
                    if other is None or other.item_id == item.item_id:
                        continue
                    w = self._widget(other)
                    if w is not None and hasattr(w, 'collapse_content'):
                        try:
                            # F-A10 follow-up: materialized_item_ids() now
                            # reaches HIDDEN detached-cache members too —
                            # reaching past the visible band is this
                            # pass's HWND-freeing POINT, but a collapsed
                            # hidden member's recorded height (and its
                            # offscreen spacer) must converge rather than
                            # drift until some later revisit. Flag ONLY
                            # when a section actually moved — the
                            # changed-signal from collapse_content() — so
                            # the steady-state repeat pass (a later
                            # request starting over already-collapsed
                            # cards) never turns the idle cache into
                            # per-start hidden re-measure churn. The next
                            # flush's refresh pass converges a flagged
                            # card once, hidden, and the flag self-clears.
                            if w.collapse_content():
                                self._mark_mutated_member(iid)
                        except RuntimeError:
                            pass
        self._schedule_flush()

    def on_message(self, message: ConversationMessage):
        """Route a single conversation message (§5.3)."""
        md = message.metadata
        role = message.role

        if md.get('is_conditional_report'):
            item = self._find_conditional_card(md)
            if item is None:
                item = self._create_conditional_card(md, message)
            item.messages.append(message)
            item.pending = False
            card = self._widget(item)
            if card is not None:
                try:
                    if getattr(card, 'is_pending', False):
                        card.set_response("")
                    card.add_report_section(message.content)
                except RuntimeError:
                    pass
                else:
                    # F-A10 follow-up: the conditional card can be a
                    # hidden detached-cache member — its report section
                    # just resized it, so flag the id for the next
                    # flush's one converging (hidden) measure.
                    self._mark_mutated_member(item.item_id)
            self._schedule_flush()
            return

        if role == 'error' and md.get('run_terminal'):
            # Run-terminal marker (MAX_REQUESTS/CANCEL for a conditional run).
            item = self._find_conditional_card(md)
            if item is None:
                item = self._create_conditional_card(md, message)
            item.messages.append(message)
            item.pending = False
            card = self._widget(item)
            if card is not None:
                try:
                    if getattr(card, 'is_pending', False):
                        card.set_response("")
                        # F-A10 follow-up: the terminal conditional card
                        # can be a hidden detached-cache member — the
                        # pending state cleared underneath the spacer's
                        # recorded height, so flag the id for the next
                        # flush's one converging (hidden) measure.
                        self._mark_mutated_member(item.item_id)
                except RuntimeError:
                    pass
            self._schedule_flush()
            return

        if md.get('is_system_agent'):
            self._route_system_agent(message)
            return

        if md.get('is_conditional'):
            self._route_conditional(message)
            return

        if role == 'assistant' and md.get('is_api_response'):
            self._route_primary_api_response(message)
            return

        if role == 'tool':
            self._route_primary_tool(message)
            return

        if role == 'permission_request':
            self._route_primary_permission(message)
            return

        # User messages and MODEL_MESSAGE_TO_USER -> top-level panel.
        self._add_panel(message)
        self._schedule_flush()

    def on_message_updated(self, message: ConversationMessage):
        """Route an in-place message update (§5.4)."""
        if message.metadata.get('is_system_agent'):
            self._route_system_agent(message)
            return

        request_seq = message.metadata.get('request_seq', 0)
        item = self._items_by_id.get(self._current_cards.get(
            (message.fork, message.turn, request_seq)))
        if item is None and request_seq > 0:
            item = self._items_by_id.get(self._current_cards.get(
                (message.fork, message.turn, 0)))
        if item is None:
            return
        card = self._widget(item)
        if card is not None:
            try:
                card.update_tool_result(message)
            except RuntimeError:
                pass
            else:
                # F-A10 follow-up: the in-place update can land on a
                # hidden detached-cache member (a tool result resolved
                # while the user reads elsewhere) — the result body
                # resized it, so flag the id for the next flush's one
                # converging (hidden) measure.
                self._mark_mutated_member(item.item_id)
        for m in item.messages:
            if m is message:
                break
        else:
            item.messages.append(message)
        self._schedule_flush()

    # ------------------------------------------------------------------
    # Routing internals
    # ------------------------------------------------------------------

    def _last_primary_position(self) -> Tuple[int, int]:
        pos = (0, -1)
        for m in self.conversation.messages:
            if not m.metadata.get('is_conditional', False) and \
               not m.metadata.get('is_system_agent', False):
                pos = (m.fork, m.turn)
        return pos

    def _find_conditional_card(self, md: dict) -> Optional[HistoryItem]:
        agent = md.get('conditional_agent_name', '')
        run_id = md.get('run_id', '')
        for item in reversed(self._items):
            if item.is_conditional and item.conditional_agent_name == agent:
                if not run_id or item.item_id.run_id == run_id or not item.item_id.run_id:
                    return item
        return None

    def _create_conditional_card(self, md: dict, message: ConversationMessage) -> HistoryItem:
        agent = md.get('conditional_agent_name', '')
        run_id = md.get('run_id', '')
        a_fork = md.get('anchor_fork', 0)
        a_turn = md.get('anchor_turn', -1)
        if a_turn == -1:
            a_fork, a_turn = self._last_primary_position()
        item = HistoryItem(
            item_id=HistoryItemId('card', a_fork, a_turn,
                                  agent_name=agent, run_id=run_id, seq=self._next_seq),
            fork=a_fork, turn=a_turn, metadata=dict(md),
            is_conditional=True, conditional_agent_name=agent, pending=False)
        self._next_seq += 1
        self._add_item(item)
        self._materialize(item)
        return item

    def _route_system_agent(self, message: ConversationMessage):
        agent_type = message.metadata.get('system_agent_type', '')
        status = message.metadata.get('status', 'running')

        # Scrutinator runs inside a tool execution; render as a subcard of the
        # last primary card rather than a standalone card.
        if agent_type == 'scrutinator':
            host = None
            for item in reversed(self._items):
                if item.item_id.kind == 'card' and not item.is_system_agent \
                        and not item.is_conditional:
                    host = item
                    break
            if host is not None:
                # Identity dedupe: both route passes re-enter here with
                # the SAME object (STARTED via on_message, the STOPPED
                # in-place mutation via on_message_updated); append once so
                # the presence-based replay builds exactly one subcard.
                if not any(m is message for m in host.messages):
                    host.messages.append(message)
                card = self._widget(host)
                if card is not None:
                    try:
                        if status == 'running':
                            card.add_system_agent_subcard(agent_type, message.metadata)
                        else:
                            card.update_system_agent_subcard(agent_type, message.metadata)
                    except RuntimeError:
                        pass
                    else:
                        # F-A10 follow-up: the HOST card can be a hidden
                        # detached-cache member — the scrutinator subcard
                        # it just gained resized it, so flag the id for
                        # the next flush's one converging (hidden)
                        # measure.
                        self._mark_mutated_member(host.item_id)
                self._schedule_flush()
                return

        pos = self._last_primary_position()
        ord_ = message.metadata.get('system_agent_ord', 0)
        key = (message.fork, message.turn, agent_type, ord_)
        item = self._items_by_id.get(self._system_agent_cards.get(key))
        if item is None:
            item = HistoryItem(
                item_id=HistoryItemId('card', pos[0], pos[1],
                                      agent_name=agent_type, seq=ord_),
                fork=pos[0], turn=pos[1], metadata=dict(message.metadata),
                is_system_agent=True, system_agent_type=agent_type,
                pending=(status == 'running'))
            self._add_item(item)
            self._system_agent_cards[key] = item.item_id
            self._materialize(item)
            card = self._widget(item)
            if card is not None:
                try:
                    if status == 'running':
                        card.set_pending()
                    else:
                        card.set_response('', metadata=dict(message.metadata), collapsed=True)
                except RuntimeError:
                    pass
        else:
            card = self._widget(item)
            if card is not None and status == 'completed' and item.pending:
                try:
                    card.set_response('', metadata=dict(message.metadata), collapsed=True)
                except RuntimeError:
                    pass
            item.pending = (status == 'running')
        # Identity dedupe: both route passes re-enter here with the SAME
        # object (STARTED via on_message, the in-place STOPPED mutation via
        # on_message_updated); append once so _replay_card renders this
        # record exactly once.
        if not any(m is message for m in item.messages):
            item.messages.append(message)
        self._schedule_flush()

    def _route_conditional(self, message: ConversationMessage):
        md = message.metadata
        item = self._find_conditional_card(md)
        if item is None:
            item = self._create_conditional_card(md, message)
        card = self._widget(item)
        if message.role == 'assistant' and md.get('is_api_response'):
            response_metadata = {k: v for k, v in md.items()
                                 if k not in ('reasoning', 'is_api_response')}
            item.messages.append(message)
            item.pending = False
            if card is not None:
                try:
                    card.set_response(message.content, md.get('reasoning', ''),
                                      metadata=response_metadata)
                except RuntimeError:
                    pass
                else:
                    # F-A10 follow-up: the conditional card can be a
                    # hidden detached-cache member — the response body
                    # resized it, so flag the id for the next flush's
                    # one converging (hidden) measure.
                    self._mark_mutated_member(item.item_id)
        elif message.role == 'tool':
            item.messages.append(message)
            if card is not None:
                try:
                    card.add_tool_use(message)
                    if md.get('status') == 'completed':
                        card.update_tool_result(message)
                except RuntimeError:
                    pass
                else:
                    # F-A10 follow-up: the conditional card can be a
                    # hidden detached-cache member — the tool section
                    # resized it, so flag the id for the next flush's
                    # one converging (hidden) measure.
                    self._mark_mutated_member(item.item_id)
        elif message.role == 'permission_request':
            item.messages.append(message)
            if card is not None:
                try:
                    card.add_permission_request(message, resolved=False)
                except RuntimeError:
                    pass
                else:
                    # F-A10 follow-up: the conditional card can be a
                    # hidden detached-cache member — a permission
                    # section resized it, so flag the id for the next
                    # flush's one converging (hidden) measure.
                    self._mark_mutated_member(item.item_id)
        self._schedule_flush()

    def _find_primary_card(self, fork: int, turn: int, request_seq: int) -> Optional[HistoryItem]:
        item = self._items_by_id.get(self._current_cards.get((fork, turn, request_seq)))
        if item is None and request_seq > 0:
            item = self._items_by_id.get(self._current_cards.get((fork, turn, 0)))
        return item

    def _route_primary_api_response(self, message: ConversationMessage):
        md = message.metadata
        request_seq = md.get('request_seq', 0)
        item = self._find_primary_card(message.fork, message.turn, request_seq)
        if item is None:
            item = HistoryItem(
                item_id=HistoryItemId('card', message.fork, message.turn, seq=request_seq),
                fork=message.fork, turn=message.turn, metadata=dict(md), pending=False)
            self._add_item(item)
            self._current_cards[(message.fork, message.turn, request_seq)] = item.item_id
            self._materialize(item)
        item.messages.append(message)
        item.pending = False
        card = self._widget(item)
        if card is not None:
            response_metadata = {k: v for k, v in md.items()
                                 if k not in ('reasoning', 'is_api_response')}
            try:
                card.set_response(message.content, md.get('reasoning', ''),
                                  metadata=response_metadata)
            except RuntimeError:
                pass
            else:
                # F-A10 follow-up: the card can be a hidden
                # detached-cache member — the response body resized it,
                # so flag the id for the next flush's one converging
                # (hidden) measure.
                self._mark_mutated_member(item.item_id)
        self._schedule_flush()

    def _route_primary_tool(self, message: ConversationMessage):
        md = message.metadata
        request_seq = md.get('request_seq', 0)
        item = self._find_primary_card(message.fork, message.turn, request_seq)
        if item is None:
            item = HistoryItem(
                item_id=HistoryItemId('card', message.fork, message.turn, seq=request_seq),
                fork=message.fork, turn=message.turn, metadata=dict(md), pending=True)
            self._add_item(item)
            self._current_cards[(message.fork, message.turn, request_seq)] = item.item_id
            self._materialize(item)
        item.messages.append(message)
        card = self._widget(item)
        if card is not None:
            try:
                card.add_tool_use(message)
                if md.get('status') == 'completed':
                    card.update_tool_result(message)
            except RuntimeError:
                pass
            else:
                # F-A10 follow-up: the card can be a hidden
                # detached-cache member (tool output streams while the
                # user reads elsewhere) — the tool section resized it,
                # so flag the id for the next flush's one converging
                # (hidden) measure.
                self._mark_mutated_member(item.item_id)
        self._schedule_flush()

    def _route_primary_permission(self, message: ConversationMessage):
        md = message.metadata
        request_seq = md.get('request_seq', 0)
        item = self._find_primary_card(message.fork, message.turn, request_seq)
        if item is None:
            item = HistoryItem(
                item_id=HistoryItemId('card', message.fork, message.turn, seq=request_seq),
                fork=message.fork, turn=message.turn, metadata=dict(md), pending=True)
            self._add_item(item)
            self._current_cards[(message.fork, message.turn, request_seq)] = item.item_id
            self._materialize(item)
        item.messages.append(message)
        # §5.5: during a reload the backlog's permission requests render
        # resolved (no live approve/deny surface). Finalization re-arms the
        # genuinely live buttons via the ordinary pending flush.
        turn_terminal = self.is_turn_terminal(message.fork, message.turn)
        turn_terminal = turn_terminal or self.load_in_progress
        item.permission_pending = not turn_terminal
        self._pending_permissions.setdefault((message.fork, message.turn), set()).add(item.item_id)
        if turn_terminal:
            item.permission_pending = False
        card = self._widget(item)
        if card is not None:
            try:
                card.add_permission_request(message, resolved=turn_terminal)
            except RuntimeError:
                pass
            else:
                # F-A10 follow-up: the card can be a hidden
                # detached-cache member — a permission section resized
                # it, so flag the id for the next flush's one
                # converging (hidden) measure.
                self._mark_mutated_member(item.item_id)
        self._schedule_flush()

    def _add_panel(self, message: ConversationMessage):
        item = HistoryItem(
            item_id=HistoryItemId('panel', message.fork, message.turn, seq=self._next_seq),
            fork=message.fork, turn=message.turn, message=message, pending=False)
        self._next_seq += 1
        self._add_item(item)
        self._materialize(item)

    # ------------------------------------------------------------------
    # Terminal events (§5.5-5.7)
    # ------------------------------------------------------------------

    def is_turn_terminal(self, fork: int, turn: int) -> bool:
        info = self.conversation.fork_turns.get(ForkTurn(fork, turn))
        if info is None:
            return False
        return info.status in (ConversationStatus.COMPLETED,
                               ConversationStatus.ERROR,
                               ConversationStatus.CANCELLED)

    def _deactivate_permissions_for_turn(self, fork: int, turn: int):
        item_ids = self._pending_permissions.pop((fork, turn), set())
        for iid in item_ids:
            item = self._items_by_id.get(iid)
            if item is None:
                continue
            item.permission_pending = False
            card = self._widget(item)
            if card is not None and hasattr(card, 'deactivate_permission'):
                try:
                    card.deactivate_permission()
                except RuntimeError:
                    pass

    def restore_live_permissions(self):
        """§5.5 restore: re-arm genuinely live approve/deny buttons after reload.

        Permissions streamed during a reload were rendered resolved because
        ``load_in_progress`` forced ``turn_terminal`` True (see
        ``_route_primary_permission``). At finalization, cards whose turn is
        genuinely non-terminal get ``permission_pending`` re-set and their
        button panels re-shown; genuinely terminal turns stay deactivated and
        are dropped from ``_pending_permissions``. A non-resumable workflow
        (§7.8) never restores the affordance.
        """
        if self.non_resumable:
            # §7.8: no event reaches permission responses on a non-resumable
            # workflow; never restore the affordance.
            return
        for key, item_ids in list(self._pending_permissions.items()):
            fork, turn = key
            if self.is_turn_terminal(fork, turn):
                # Genuinely terminal: stay deactivated, drop the pending marker.
                del self._pending_permissions[key]
                continue
            for iid in item_ids:
                item = self._items_by_id.get(iid)
                if item is None:
                    continue
                item.permission_pending = True
                card = self._widget(item)
                if card is not None and hasattr(card, 'activate_permission'):
                    try:
                        card.activate_permission()
                    except RuntimeError:
                        pass
        if self._view is not None:
            try:
                self._view.layout_history()
            except RuntimeError:
                pass


    def _clear_pending_for_turn(self, fork: int, turn: int):
        """Clear any still-pending cards for a terminal turn.

        Matches the old tab behavior (on_turn_complete / _show_error /
        on_cancel_ack called clear_pending on the current request card).
        """
        for item in self._items:
            if item.item_id.kind != 'card':
                continue
            if item.fork != fork or item.turn != turn:
                continue
            if not item.pending and not item.pending_muted:
                continue
            item.pending = False
            card = self._widget(item)
            if card is not None and hasattr(card, 'clear_pending'):
                try:
                    card.clear_pending()
                except RuntimeError:
                    pass

    def on_turn_complete(self, fork: int, turn: int):
        self._deactivate_permissions_for_turn(fork, turn)
        self._clear_pending_for_turn(fork, turn)
        self._flush()

    def on_cancel_ack(self, fork: int, turn: int):
        self._deactivate_permissions_for_turn(fork, turn)
        self._clear_pending_for_turn(fork, turn)
        self._flush()

    def on_error(self, fork: int, turn: int, error_message: str):
        self._deactivate_permissions_for_turn(fork, turn)
        self._clear_pending_for_turn(fork, turn)
        # Terminal-turns deactivate the live send/approve surface in the tab;
        # the error panel is rendered as a top-level assistant panel.
        msg = ConversationMessage(
            fork=fork, turn=turn, role='assistant',
            content=f"Warning: Error: {error_message}",
            timestamp=datetime.now(timezone.utc),
            metadata={'is_error': True})
        item = HistoryItem(
            item_id=HistoryItemId('error_panel', fork, turn, seq=self._next_seq),
            fork=fork, turn=turn, error_text=error_message, message=msg, pending=False)
        self._next_seq += 1
        self._add_item(item)
        self._materialize(item)
        self._flush()

    def on_fork_turn_update(self):
        """§5.8: visibility may have changed; re-check each item and flush.

        The single mechanism every visibility change flows through. A reload
        streams records from EVERY fork (the two-phase scan), so the item
        model holds the fork-shadowed items the live Send-Fork destroy
        removed; this re-check re-derives exactly that set from the pure
        positional predicate ``fork_turn_state.is_visible`` and destroys it
        from both structures, so a reopen rebuilds the identical visible
        set (§12). Offscreen fork-mode preview items are hidden, not
        destroyed; they are destroyed only when the forking message is
        sent (§5.8) - this sweep is the ordinary re-check that fire drives.
        ``conversation.messages`` is untouched (§7.6 append-only: the lost
        branch's history stays in the data).

        Items with fork > latest_fork cannot legitimately appear (the seed
        builds every fork's IN facts before streaming, §7.2; the live send
        posts the eager IN fact before firing, §5.10) and ``is_visible``
        raises for them - the fork guard skips them defensively.
        """
        if self.conversation is None:
            return
        state = self.conversation.fork_turn_state
        view_fork = state.latest_fork
        shadowed = [
            item for item in self._items
            if (item.fork <= view_fork
                and not state.is_visible(item.fork, item.turn, view_fork))
        ]
        if not shadowed:
            return
        for item in shadowed:
            self._remove_item(item)
            # A shadowed position contributes no restorable set (§5.5):
            # pop the registry so the §7.4 step-9 settle names only the
            # genuinely-visible pending turns.
            self._pending_permissions.pop((item.fork, item.turn), None)
        self._schedule_flush()

    # ------------------------------------------------------------------
    # Pending-card lifecycle (§5.1, §9)
    # ------------------------------------------------------------------

    def mute_pending(self, item_id: HistoryItemId):
        item = self._items_by_id.get(item_id)
        if item is None:
            return
        item.pending_muted = True
        item.pending = False
        card = self._widget(item)
        if card is not None:
            try:
                if getattr(card, 'is_pending', False):
                    card.set_response("")
                    # F-A10 follow-up: a pending card being muted can be
                    # a hidden detached-cache member (the user scrolled
                    # away mid-run) — the mute cleared its pending
                    # state, so flag it inside the mutation for the
                    # next flush's one converging (hidden) measure.
                    self._mark_mutated_member(item_id)
            except RuntimeError:
                pass

    def mute_pending_for_conditional(self, agent_name: str) -> Optional[HistoryItemId]:
        for item in reversed(self._items):
            if item.is_conditional and item.conditional_agent_name == agent_name and item.pending:
                self.mute_pending(item.item_id)
                return item.item_id
        return None

    def add_report_to_conditional_card(self, agent_name: str, content: str) -> Optional[HistoryItemId]:
        for item in reversed(self._items):
            if item.is_conditional and item.conditional_agent_name == agent_name:
                if item.pending or item.pending_muted:
                    self.mute_pending(item.item_id)
                card = self._widget(item)
                if card is not None:
                    try:
                        card.add_report_section(content)
                    except RuntimeError:
                        pass
                    else:
                        # F-A10 follow-up: the target card can be a
                        # hidden detached-cache member (the agent
                        # reports while the user reads elsewhere) — a
                        # report section resized it, so flag the id for
                        # the next flush's one converging (hidden)
                        # measure.
                        self._mark_mutated_member(item.item_id)
                return item.item_id
        return None

    # ------------------------------------------------------------------
    # Batching and flush (§6)
    # ------------------------------------------------------------------

    def begin_batch(self):
        self._batch_depth += 1

    def end_batch(self):
        self._batch_depth = max(0, self._batch_depth - 1)
        if self._batch_depth == 0 and self._flush_pending:
            self._flush_pending = False
            self._flush()

    def _schedule_flush(self):
        if self._batch_depth > 0:
            self._flush_pending = True
            return
        self._flush()

    def _flush(self):
        if self._view is None:
            return
        try:
            if not self._view.is_valid():
                return
            if self._fork_preview:
                return
            # Capture pre-change scroll geometry for the §8.1 policy.
            state = self._view.capture_scroll_state()
            # §8.5: reconcile the padded visible range, re-measure the
            # materialized widgets, refresh spacers, and lay out once.
            if self.load_in_progress:
                # One-shot head paint (§5.10.1): the sole pre-finalization
                # display requirement is the near-instant first user message.
                # The first load flush renders ONLY the first item and forces
                # a synchronous paint; every later streaming flush does NO
                # view work so the GUI thread stays free to paint and the
                # frozen first card is never invalidated or flickered. The
                # full backlog (exact heights, chronological order, end-of-load
                # scroll) converges at finalization (§7.4).
                if not self._load_first_paint_done and self._items:
                    self.compute_visible_range()
                    total_h = self._load_window_total_height()
                    self._view.apply_load_window(
                        self._items, 0, 0, total_h, measure=True)
                    self._view.layout_history()
                    self._view.set_virtual_height(total_h)
                    # Force a synchronous paint: the replay/finalize CallAfter
                    # chain keeps the GUI event queue non-empty, and on Windows
                    # WM_PAINT is dispatched only when the queue empties — so
                    # without this the first card stays invisible until
                    # finalization (§5.10.1).
                    self._view.force_paint()
                    self._load_first_paint_done = True
            else:
                first, last = self.compute_visible_range()
                # Apply the expansion policy now that routing has populated
                # each card's sections (a card is empty at _materialize time,
                # so the policy must run against the fully-routed sections
                # here, §8.3).
                for item in self._items:
                    if item.item_id.kind != 'card':
                        continue
                    card = self._widget(item)
                    if card is not None and hasattr(card, 'iter_sections'):
                        try:
                            # F-A10: with the revisit cache live this
                            # registry read reaches HIDDEN detached-cache
                            # cards too — flag one only when a section
                            # actually moved (the changed-signal), so the
                            # steady-state policy re-affirmation never
                            # converts the idle-cache skip into per-flush
                            # hidden re-measure churn; a moved cached
                            # card then converges on the refresh pass
                            # below (one hidden measure, and the flag
                            # self-clears).
                            if self.apply_expansion_policy(item, card):
                                self._mark_mutated_member(item.item_id)
                        except Exception:
                            pass
                self._view.apply_item_list(self._items, first, last)
                self._view.refresh_materialized_heights()
                self._view.refresh_spacer_heights()
                # F-B4: refill last_y from the POST-change heights — the
                # rule-4 anchor lookup must land where the anchor item
                # sits NOW (a user toggle resized a card between capture
                # and policy), never on stale pre-change estimates. The
                # near-bottom / pixels_below decision still uses the
                # pre-change state (§8.1); only the anchor geometry
                # moves forward.
                self.compute_visible_range()
                self._view.layout_history()
            self._apply_scroll_policy(state)
        except WX_FAILURES:
            self._logger.exception("history flush failed")

    def _apply_scroll_policy(self, state: ScrollState):
        """§8.1 scroll policy over pre-change ScrollState geometry."""
        try:
            if (self._view is not None
                    and hasattr(self._view, 'is_thumb_drag_active')
                    and self._view.is_thumb_drag_active()):
                # F-A2: never script a scroll beneath a held thumb drag —
                # a scripted jump fights the OS drag mapping and pops the
                # thumb. The release settle (settle_thumb_drag) re-runs
                # the policy over fresh geometry and consumes any pending
                # force-bottom then.
                return
            if self._force_scroll_bottom:
                self._force_scroll_bottom = False
                self._view.scroll_to_bottom()
                # The programmatic scroll echo is suppressed (§8.6), so the
                # single-path recompute never runs for this jump: reconcile
                # the visible range at the new geometry now, otherwise the
                # list bottom stays spacers after the end-of-load scroll.
                try:
                    first, last = self.compute_visible_range()
                    if self.load_in_progress:
                        total_h = self._load_window_total_height()
                        self._view.apply_load_window(
                            self._items, first, last, total_h)
                        self._view.layout_history()
                        self._view.set_virtual_height(total_h)
                    else:
                        self._view.apply_item_list(self._items, first, last)
                        self._view.layout_history()
                except WX_FAILURES:
                    self._logger.exception(
                        "post-force-bottom range reconcile failed")
                return
            if self.load_in_progress:
                # Viewport stays at the list head while chunks stream.
                return
            if state.pixels_below <= self.BOTTOM_TOLERANCE_PX:
                self._user_scrolled = False
                self._view.scroll_to_bottom()
                return
            # The user is reading: restore the logical top anchor.
            if self._user_scrolled:
                pos = None
                if state.anchor_item_id is not None:
                    try:
                        pos = self._view.position_of_anchor(
                            state.anchor_item_id, state.anchor_offset)
                    except RuntimeError:
                        pos = None
                if pos is None and state.anchor_item_id is not None:
                    # §8.1: the anchor item was removed (fork truncation,
                    # replay) — restore the next visible item instead;
                    # the last resort stays the captured absolute
                    # position, now in device px (F-B2).
                    for j, it in enumerate(self._items):
                        if it.item_id == state.anchor_item_id:
                            if j + 1 < len(self._items):
                                try:
                                    pos = self._view.position_of_anchor(
                                        self._items[j + 1].item_id, 0)
                                except RuntimeError:
                                    pos = None
                            break
                if pos is None:
                    pos = state.view_start
                self._view.scroll_to(pos)
                # F-B3: the scroll_to echo is suppressed (§8.6), so the
                # single-path recompute never runs for this scripted jump:
                # reconcile the visible band at the restored position, or
                # the viewport sits on stale spacers (grey) until the
                # next user scroll — the force-bottom precedent above.
                try:
                    first, last = self.compute_visible_range()
                    self._view.apply_item_list(self._items, first, last)
                    self._view.layout_history()
                except WX_FAILURES:
                    self._logger.exception(
                        "post-restore range reconcile failed")
        except WX_FAILURES:
            self._logger.exception("scroll policy apply failed")

    # ------------------------------------------------------------------
    # Policies (§8)
    # ------------------------------------------------------------------

    def reposition(self, pos=None, force=False):
        """§8.6 single-path scroll recompute entry point.

        ``pos`` is the view-normalized device-px position; ``force``
        bypasses the echo-dedup so a gesture settle (F-A1) is never
        swallowed by an identical last position.
        """
        if pos is not None:
            if not force and pos == self._last_scroll_pos:
                return  # scroll_to() echo: empty delta, no recompute.
            self._last_scroll_pos = pos
        if self._batch_depth > 0:
            return
        try:
            near_bottom = (self._view is not None and
                           self._view.is_near_bottom(self.BOTTOM_TOLERANCE_PX))
        except WX_FAILURES:
            near_bottom = False
        # Scrolling within the bottom tolerance does not suspend auto-follow.
        self._user_scrolled = not near_bottom
        # §8.1 width-pass fix: a genuine user scroll retracts the
        # resize follow-through — the convergence finale must never
        # re-pin a reader who moved away mid-convergence.
        self._remeasure_pin = False
        if self._view is None:
            return
        try:
            if self.load_in_progress:
                # Frozen head display (§5.10.1): the window never changes
                # during a reload, so a user scroll only moves the scrollbar
                # — no view work (the backlog stays virtual until
                # finalization).
                return
            first, last = self.compute_visible_range()
            if not force and self._thumb_drag_gated(first, last):
                # F-A7 fling gate: virtualize through the swept area —
                # spacer heights stay current (F-A6) and the staged
                # layout keeps the frozen range in place (F-A2); the
                # band materializes at the first park / the release
                # settle, never mid-flight.
                self._view.refresh_spacer_heights()
                self._view.layout_history()
                return
            # F-A7: an in-drag brand-new band is a full visible-set build
            # per turn; time it so a slow-but-in-budget turn still latches
            # the gate for the rest of the gesture (tall cards cost more
            # than they count).
            timed = (not force and self._drag_active())
            t0 = time.perf_counter() if timed else 0.0
            self._view.apply_item_list(self._items, first, last)
            # F-A6: spacer rows must carry current estimates on this hot
            # path too — untouched offscreen spacers keep stale heights
            # after any measurement otherwise (range / band drift).
            self._view.refresh_spacer_heights()
            self._view.layout_history()
            if timed and (time.perf_counter() - t0) * 1000.0 >= \
                    self.DRAG_TURN_BUDGET_MS:
                self._drag_fling_gated = True
        except WX_FAILURES:
            self._logger.exception("reposition view update failed")

    def record_scroll_position(self, pos):
        """Track a live user/scripted scroll position for the echo-dedup.

        F-A1/A3: THUMBTRACK coalescing and scroll_to()'s own echo feed
        this so an identical position arriving in reposition() stays a
        no-op; no recompute runs here.
        """
        if pos is not None:
            self._last_scroll_pos = pos

    def _drag_active(self) -> bool:
        """F-A7 helper: True while a genuine thumbnail-drag gesture is held."""
        if (self._view is None
                or not hasattr(self._view, 'is_thumb_drag_active')):
            return False
        try:
            return bool(self._view.is_thumb_drag_active())
        except WX_FAILURES:
            return False

    def _thumb_drag_gated(self, first: int, last: int) -> bool:
        """F-A7: True when this drag-settle turn must virtualize through the
        swept area instead of building the band live.

        Mid-list, a moving viewport crosses spacer territory the visible
        band never materialized: building that brand-new band means
        shell + populate + expansion + HtmlWindow reflow for every
        visible card, per settle turn, far faster than the queue can
        drain — WM_PAINT starves (it is dispatched only when the queue
        empties) and the viewport goes grey with fresh shells flashing at
        their native-creation position (the mid-drag grey / thin flashing
        band symptom). Slow drags bring a card or two per turn, inside
        the DRAG_BUILD_BUDGET, and stay live. An over-budget turn
        LATCHES the gate for the rest of the gesture (as does an
        in-budget turn that blows DRAG_TURN_BUDGET_MS — tall cards cost
        more than they count): a fling that starts slow then accelerates
        converges only at park/release, never mid-flight. Gated turns
        still refresh spacer heights (F-A6) and stage the layout (F-A2,
        frozen range); the gesture start (start_thumb_drag), the parked
        settle (park_thumb_drag) and the release settle all clear the
        latch and run one ungated full recompute.
        """
        try:
            if not self._drag_active():
                return False
            if self._drag_fling_gated:
                return True
            misses = 0
            n = len(self._items)
            lo = max(0, int(first))
            hi = min(int(last), n - 1)
            for i in range(lo, hi + 1):
                iid = self._items[i].item_id
                if self._view.get_widget(iid) is None:
                    misses += 1
            if misses <= self.DRAG_BUILD_BUDGET:
                return False
            self._drag_fling_gated = True
            return True
        except WX_FAILURES:
            # Never let the gate itself become the failure mode: fall
            # through to the ungated (pre-F-A7) behavior.
            return False

    def start_thumb_drag(self):
        """F-A7: gesture-start reset — a new gesture never inherits the
        previous one's fling latch (a release settle swallowed by the
        echo-suppression latch is deferred by the view, not run, so it
        cannot be relied on to have cleared before the next touchdown)."""
        self._drag_fling_gated = False

    def park_thumb_drag(self):
        """F-A9/F-A7: the thumb parked mid-gesture — the view's park POLL
        converged (posted-message CallAfter chain; the first cut's
        one-shot wx.Timer WM_TIMER never fired under the native MSW drag
        loop) while the drag latch still holds. Clear the fling latch and
        run ONE ungated full recompute at the parked position, ending the
        grey sweep while the thumb is still held: the parked band
        materializes (flash-free — view F-A7 birth positions), and the
        view force-paints on return. The reading-pace unlatch (view
        _on_scroll, F-A9 fourth cut) clears the same latch on sparse
        tracks; this method is the stillness convergence. The release
        settle (settle_thumb_drag) remains the authoritative convergence —
        frozen-range reapply via _flush, height re-measure, §8.1 policy
        over fresh geometry."""
        self._drag_fling_gated = False
        self.reposition(force=True)

    def settle_thumb_drag(self, pos=None):
        """F-A1/A2/A5: the thumb-drag gesture-end settle.

        One full recompute with the frozen scrollbar range re-applied
        (the view stages virtual-size changes while a drag is active),
        then a fresh flush applies the §8.1 policy over current geometry
        — a pending end-of-load force-bottom is consumed here, after the
        gesture, instead of fighting it. The view force-paints after
        this returns so starved paints cannot survive the gesture
        (F-A5).
        """
        try:
            # F-A7: the release settle converges the whole gesture — the
            # fling gate must never suppress it, and the latch clears
            # here for the next gesture.
            self._drag_fling_gated = False
            self.reposition(pos, force=True)
            self._flush()
        except WX_FAILURES:
            self._logger.exception("thumb-drag settle failed")

    def on_width_changed(self, width: int):
        """§8.2/§8.5 full width pass."""
        if width <= 0:
            return
        new_width = max(self.MIN_CARD_WIDTH, width)
        old_width = self._effective_card_width
        self._effective_card_width = new_width
        self._resize_pre_state = None
        if self._view is not None and not self.load_in_progress:
            try:
                self._resize_pre_state = self._view.capture_scroll_state()
            except WX_FAILURES:
                self._resize_pre_state = None
        if old_width == new_width:
            # Floor guard: effective width pinned (or unchanged) -> relayout only.
            try:
                if self._view is not None:
                    self._view.layout_history()
            except WX_FAILURES:
                self._logger.exception("width pass layout failed")
            # Resize keep-bottom (§8.1 width-pass fix): a vertical-only
            # resize lands exactly here, so the keep-bottom decision
            # above still applies over the new client geometry; no pin
            # — this branch re-measures nothing at a new width.
            self._apply_resize_scroll_policy(self._resize_pre_state, hold_pin=False)
            return
        self.invalidate_cache_for_width(new_width)
        if self._view is None:
            return
        if self.load_in_progress:
            # Windowed reload (§8.5): re-measure the one displayed widget at
            # the new width and re-apply the frozen head window.
            # apply_item_list would force a full-sizer rebuild out of window
            # mode, and the chunked background pass would churn offscreen
            # widgets on every GUI idle slice — the §7.4 step-5 measure-all
            # at finalization owns the full pass instead.
            try:
                self._view.remeasure_materialized(new_width)
                self.compute_visible_range()
                total_h = self._load_window_total_height()
                self._view.apply_load_window(
                    self._items, 0, 0 if self._items else -1, total_h)
                self._view.layout_history()
                self._view.set_virtual_height(total_h)
            except WX_FAILURES:
                self._logger.exception(
                    "width pass failed during windowed reload")
            return
        try:
            self._view.refresh_spacer_heights()
            first, last = self.compute_visible_range()
            self._view.apply_item_list(self._items, first, last)
            self._view.remeasure_materialized(new_width)
            self._view.layout_history()
        except WX_FAILURES:
            # The original crash chain: wx's sizer GetItem index assert
            # (an AssertionError subclass, not RuntimeError) escaped this
            # guard and died in the size timer. Any residual desync is now
            # repaired by apply_item_list's integrity gate; log what hit.
            self._logger.exception("width pass failed")
        # Resize keep-bottom (§8.1 width-pass fix): the relayout moved
        # the bottom, and the chunked pass below moves it again as the
        # stale rows converge. Apply the scroll decision over the
        # pre-resize capture — exactly like a flush — and hold the
        # keep-bottom pin for the convergence finale.
        self._apply_resize_scroll_policy(self._resize_pre_state,
                                        hold_pin=True)
        self._resize_pre_state = None
        self._start_chunked_remeasure(new_width)

    def _apply_resize_scroll_policy(self, pre, hold_pin: bool):
        """Resize keep-bottom policy (§8.1 width-pass fix).

        The width pass was the one geometry-changing path with no scroll
        decision (§8.2/§8.5 specced widths/measurement only), so a live
        view pinned at the bottom lost its pin the moment a resize moved
        the bottom — and the flush policy cannot recover it: the next
        flush's capture reads below-tolerance, no rule matches, and
        auto-follow stays dead until the user physically scrolls back
        within 24px of the bottom.

        ``pre`` is the capture taken at width-pass entry (the native
        WM_SIZE adjustment has already run; our relayout has not), so
        its pixels_below is a mixed-geometry read that overstates the
        distance after a client-height shrink. The authentic pre-resize
        auto-follow signal is therefore the ``_user_scrolled`` flag —
        rule 3 kept clearing it while the stream followed — with the
        capture's own near-bottom read as the co-signal (the native
        clamp variant: a resize-generated scroll event can flip the
        flag, and a still-at-bottom captured position still re-engages).

        ``hold_pin`` keeps a keep-bottom decision alive for the chunked
        re-measure convergence, whose finale re-pins once more after the
        last rows land (the third bottom shift of a resize). Any
        genuine user scroll clears the pin (reposition), a drag latch
        cancels it (drag guard), and a reload bracket clears it in the
        chunked-step reset.
        """
        if pre is None or self._view is None:
            return
        try:
            drag = (hasattr(self._view, 'is_thumb_drag_active')
                    and self._view.is_thumb_drag_active())
            keep_bottom = ((not self._user_scrolled)
                           or pre.pixels_below <= self.BOTTOM_TOLERANCE_PX)
            if hold_pin:
                self._remeasure_pin = keep_bottom and not drag
            if drag:
                # F-A2: never script beneath a held thumb drag; the
                # release settle re-runs the policy over fresh geometry.
                return
            if keep_bottom:
                # Rule-3 re-engage precedent: explicit flag clear,
                # scripted bottom jump, and the force-bottom band
                # reconcile at the new bottom (the scripted echo is
                # suppressed, F-A3, so nothing else reconciles it).
                self._user_scrolled = False
                self._view.scroll_to_bottom()
                first, last = self.compute_visible_range()
                self._view.apply_item_list(self._items, first, last)
                self._view.layout_history()
                return
            if self._user_scrolled:
                # §8.1 rule 4 over the pre-resize capture: the reading
                # position follows the anchor across the relayout (the
                # policy's own lane owns the drag guard and reconcile).
                self._apply_scroll_policy(pre)
        except WX_FAILURES:
            self._logger.exception("resize scroll policy failed")

    def _auto_open_default(self, role: str) -> bool:
        return {'header': True, 'permission': True, 'report': True}.get(role, False)

    def apply_expansion_policy(self, item: HistoryItem, card) -> bool:
        """Apply user overrides / defaults to a materialized card (§8.3).

        Returns True when any section's expansion state actually CHANGED
        (a native expand/collapse ran on the card), False otherwise.

        F-A10 (dead-cache defect fix follow-up): the flush's policy loop
        walks every card, and with the revisit cache live a HIDDEN
        detached-cache card is reachable there through the registry —
        but cached widgets are skipped by the per-flush measure/width
        passes, so a policy mutation landing on one must be flagged for
        exactly one converging (hidden) measure, or the item's offscreen
        spacer holds the stale pre-mutation height until some later
        revisit re-Adds the card (one-flush height drift). The
        changed-signal is what keeps that flag honest: a merely
        re-affirmed policy — every section already in its target state,
        the steady-state case on every flush — must NOT flag the card,
        or the idle-cache skip would silently turn into a re-measure of
        every hidden cached card on every flush. The idempotent callers
        elsewhere ignore the return.
        """
        changed = False
        try:
            sections = list(card.iter_sections())
        except RuntimeError:
            return False
        for role, ordinal, section in sections:
            if role == 'header':
                continue
            if (role, ordinal) in item.user_overrides:
                # §8.3: per-instance override — (role, ordinal), never
                # the role alone (defect B1's sibling cascade).
                target = item.user_overrides[(role, ordinal)]
            elif role == 'permission' and not item.permission_pending:
                # A resolved permission (turn terminal or already answered)
                # stays collapsed; only pending permissions default expanded.
                target = False
            elif item.force_collapsed:
                # collapse_content() freed this card's HWNDs when a newer
                # request started; keep those programmatic collapses.
                target = section.is_expanded
            else:
                target = self._auto_open_default(role)
            try:
                if target and not section.is_expanded:
                    section.expand(notify=False)
                    changed = True
                elif not target and section.is_expanded:
                    section.collapse(notify=False)
                    changed = True
            except RuntimeError:
                pass
        return changed

    def on_user_toggle(self, item_id: HistoryItemId, role: str,
                       ordinal: int, expanded: bool):
        item = self._items_by_id.get(item_id)
        if item is None:
            return
        # §8.3: records are per section instance — (role, ordinal). The
        # role-only key cascaded one click to every same-role sibling on
        # the card (defect B1); the ordinal keeps siblings independent.
        item.user_overrides[(role, int(ordinal))] = expanded
        self._schedule_flush()

    # ------------------------------------------------------------------
    # Geometry model and virtualization (§8.5)
    # ------------------------------------------------------------------

    def effective_card_width(self) -> int:
        """Effective card width at the current client width (§8.2 floor)."""
        if self._view is None:
            return self.MIN_CARD_WIDTH
        try:
            client_w = self._view.get_client_width()
        except (RuntimeError, AttributeError):
            client_w = 0
        return max(self.MIN_CARD_WIDTH, client_w)

    def record_height(self, item: HistoryItem, height: int):
        """Write an exact measured height into the width-keyed cache row.

        F-A8 (estimate-stat idempotency): a re-measure of an already
        recorded item REPLACES its sample, it never adds a second one.
        Every flush's refresh_materialized_heights re-measures the whole
        materialized band, and measure_slice / remeasure_materialized /
        re-materialize all overlap previously measured items, so the
        naive accumulate double-counted whichever rows happened to be on
        screen: the running mean _estimate_for_item feeds to every
        not-yet-measured item (and that _remove_item subtracts a single
        contribution from) drifted toward the on-screen sample, and the
        pool count grew without bound over a live session. Subtract the
        item's previous contribution — the same arithmetic _remove_item
        uses — before recording the new one.
        """
        width = self.effective_card_width()
        # F-A8: drop the item's previous sample first, so a re-measure at
        # the same width — or a move to a new width — is net zero for the
        # running mean instead of a duplicate sample.
        if item.measured_at_width > 0 and item.last_height > 0:
            old_stats = self._height_stats.get(item.measured_at_width)
            if old_stats and old_stats.get('count'):
                old_stats['sum'] = max(0, old_stats['sum'] - item.last_height)
                old_stats['count'] = max(0, old_stats['count'] - 1)
        item.last_height = max(0, int(height))
        item.measured_at_width = width
        item.exact = True
        stats = self._height_stats.setdefault(width, {'sum': 0, 'count': 0})
        stats['sum'] += item.last_height
        stats['count'] += 1

    def _row_border_v(self, item: HistoryItem) -> int:
        """F-A4 (§8.2): one history row's vertical sizer-border
        contribution — primary cards carry 2px TOP + 2px BOTTOM, panels
        and error panels 2px wx.ALL (4px vertical either way), and
        indented cards are LEFT-only (0px vertical). The estimate model
        counts ROW totals (widget height + this border) so the sizer min
        height — and with it the scrollbar range — is identical whether
        a row is a widget or a spacer: the range can no longer churn
        beneath a held thumb drag as the materialized band swaps
        (defect A2's structural driver).
        """
        if item is not None and item.item_id.kind == 'card':
            return 0 if (item.is_conditional or item.is_system_agent) else 4
        return 4

    def _estimate_for_item(self, item: HistoryItem, width: int) -> int:
        """Row-total height estimate for spacer / range math (§8.5).

        Returns the widget-height estimate PLUS the row's vertical border
        (F-A4) — exactly what the sizer counts for that row, so totals,
        item tops (``last_y``) and anchor positions all live in the same
        border-inclusive space the real layout uses.
        """
        border = self._row_border_v(item)
        if item.measured_at_width == width and item.exact:
            return item.last_height + border
        if item.last_height > 0:
            # Stale-width reuse: the old-width height is the new-width
            # non-exact estimate; frozen until a synchronous measure.
            return item.last_height + border
        if self._use_line_count_heuristic:
            return self._line_count_estimate(item, width) + border
        stats = self._height_stats.get(width, {})
        if stats.get('count'):
            return stats['sum'] // stats['count'] + border
        return self.DEFAULT_ESTIMATE_HEIGHT + border

    def _line_count_estimate(self, item: HistoryItem, width: int) -> int:
        """Cheap line-count fallback (escalation clause, inactive by default)."""
        lines = 0
        for m in item.messages:
            lines += max(1, m.content.count('\n') + 1)
        return max(self.DEFAULT_ESTIMATE_HEIGHT, int(lines * 18) + 40)

    def height_for_item(self, item: HistoryItem) -> int:
        """Public ROW-TOTAL estimate used by the view for spacers — the
        widget height plus the row's vertical border (F-A4)."""
        return self._estimate_for_item(item, self.effective_card_width())

    def invalidate_cache_for_width(self, new_width: int):
        """Mark every row measured at a different width as non-exact (§8.5)."""
        for item in self._items:
            if item.measured_at_width != new_width:
                item.exact = False
                # last_height is retained as the stale-width estimate.

    def compute_visible_range(self, items=None) -> Tuple[int, int]:
        """Padded visible range over an item list from the scroll geometry.

        Also records the list's total estimated height (the windowed
        reload's virtual size) so load-mode callers do not run a second
        O(n) estimate pass over the whole backlog on every flush.
        """
        width = self.effective_card_width()
        self._last_range_width = width
        if items is None:
            items = self._items
        n = len(items)
        if n == 0:
            self._last_range_total = 0
            return (0, -1)
        if self._view is None:
            self._last_range_total = 0
            return (0, n - 1)
        try:
            client_h = self._view.get_client_height()
            scroll_y = self._view.current_scroll_y_px()
        except (RuntimeError, AttributeError):
            self._last_range_total = 0
            return (0, n - 1)
        # Fill last_y for every item (used by anchor restore) while scanning.
        tops = []
        y = 0
        for item in items:
            item.last_y = y
            tops.append(y)
            y += self._estimate_for_item(item, width)
        self._last_range_total = y
        # F-A7 defect fix: clamp the stale-tail degenerate range. This
        # scan defaulted first=0, so a scroll position at/beyond the
        # model's total height (the frozen drag range outliving a shrunk
        # estimate model — force_collapsed churn, width-pass convergence
        # — or spacer/estimate drift) matched no item and fell through to
        # (0, n-1): an apply on that range materialized the WHOLE list
        # mid-gesture — the multi-second queue-clogging storm behind the
        # "drag greys out and sticks" symptom. The trailing screenful
        # clamps instead; the padding below still applies.
        first = n - 1
        for i in range(n):
            if tops[i] + self._estimate_for_item(items[i], width) > scroll_y:
                first = i
                break
        last = first
        for i in range(first, n):
            if tops[i] > scroll_y + client_h:
                break
            last = i
        first = max(0, first - self.PADDING)
        last = min(n - 1, last + self.PADDING)
        return (first, last)

    def _load_window_total_height(self) -> int:
        """Total height recorded by the latest compute_visible_range() pass.

        Windowed-reload callers use this instead of a second O(n) estimate
        pass; the width guard covers a stale cache after a width change.
        """
        if self._last_range_width == self.effective_card_width():
            return self._last_range_total
        width = self.effective_card_width()
        y = 0
        for item in self._items:
            item.last_y = y
            y += self._estimate_for_item(item, width)
        return y

    def _sync_item_list(self, items):
        """Apply a padded visible range over an arbitrary list (fork preview)."""
        if self._view is None:
            return
        first, last = (0, len(items) - 1)
        try:
            first, last = self.compute_visible_range(items)
        except (RuntimeError, AttributeError):
            pass
        try:
            self._view.apply_item_list(items, first, last)
            self._view.layout_history()
        except WX_FAILURES:
            self._logger.exception("fork preview view sync failed")

    def _start_chunked_remeasure(self, new_width: int):
        self._remeasure_index = 0
        self._remeasure_width = new_width
        self._chunked_remeasure_step()

    def _chunked_remeasure_step(self):
        if self._view is None:
            return
        try:
            if not self._view.is_valid():
                return
        except WX_FAILURES:
            return
        if self.load_in_progress:
            # A reload streams while the pass is pending: stop churning
            # offscreen widgets; finalization's measure-all owns the pass.
            # A pending resize keep-bottom pin cannot outlive its pass
            # (§8.1 width-pass fix): the reload bracket owns the scroll
            # from here, and no later finale may consume a stale pin.
            self._remeasure_pin = False
            self._remeasure_index = 0
            return
        items = self._items
        start = self._remeasure_index
        end = min(len(items), start + self.MEASURE_CHUNK)
        if start < end:
            try:
                self._view.measure_slice(items[start:end])
            except WX_FAILURES:
                self._logger.exception(
                    "chunked remeasure slice failed (index %d)", start)
            self._remeasure_index = end
        if end < len(items):
            self._call_after(self._chunked_remeasure_step)
            return
    
        try:
            self._view.refresh_spacer_heights()
            self._view.layout_history()
            if self._remeasure_pin:
                # Resize keep-bottom pin (§8.1 width-pass fix): the
                # pre-resize capture was at/near the bottom (auto-follow
                # pinned), and this convergence finale is the THIRD
                # bottom shift of a resize — the last rows just became
                # exact at the new width. Re-pin once and clear, so the
                # pin can never survive into a different state. The
                # scripted-jump echo is suppressed (F-A3), and the
                # post-force-bottom precedent reconciles the band at
                # the new bottom. A user scroll (reposition clears the
                # pin) or a live drag (drag guard) already cancelled
                # it; here only the drag guard can still apply.
                drag = (hasattr(self._view, 'is_thumb_drag_active')
                        and self._view.is_thumb_drag_active())
                self._remeasure_pin = False
                if not drag:
                    self._user_scrolled = False
                    self._view.scroll_to_bottom()
                    first, last = self.compute_visible_range()
                    self._view.apply_item_list(self._items, first, last)
                    self._view.layout_history()
        except WX_FAILURES:
            self._logger.exception(
                "chunked remeasure final layout failed")

    def measure_all(self, on_complete: Optional[Callable] = None):
        """§7.4 step-5 measure-all: chunked materialize/measure/dematerialize."""
        self._measure_all_on_complete = on_complete
        self._measure_all_index = 0
        self._measure_all_pre_state = None
        if self._view is not None:
            try:
                self._measure_all_pre_state = self._view.capture_scroll_state()
            except WX_FAILURES:
                pass
        self._measure_all_step()

    def _measure_all_step(self):
        if self._view is None:
            self._finish_measure_all()
            return
        try:
            if not self._view.is_valid():
                self._finish_measure_all()
                return
        except WX_FAILURES:
            self._finish_measure_all()
            return
        items = self._items
        start = self._measure_all_index
        end = min(len(items), start + self.MEASURE_CHUNK)
        if start < end:
            try:
                self._view.measure_slice(items[start:end])
            except WX_FAILURES:
                self._logger.exception(
                    "measure-all slice failed (index %d)", start)
            self._measure_all_index = end
        if end < len(items):
            self._call_after(self._measure_all_step)
            return
        self._finish_measure_all()

    def _finish_measure_all(self):
        pre = self._measure_all_pre_state
        self._measure_all_pre_state = None
        # F-A7 third cut (crash fix, 2026-10-06 19:42 log): never put
        # the pre-pass scroll position back beneath a HELD thumb drag.
        # In that repro the user dragged slowly through the whole load
        # window; the measure-all finished mid-gesture, Scroll() fought
        # the OS drag capture, and the corrupted native scroll state
        # tripped the end-of-load windowed->full whole-band rebuild into
        # a hard native death — silent (no Python exception, no log
        # line), the log tail dying mid card-construction burst at the
        # SAME site as the first F-A7 crash. scroll_to() itself now
        # guard-checks the drag state (single choke point), so this is
        # the documented intent and the belt. §7.4 step-5's put-back
        # exists so the pass cannot trip step 6's user-scrolled check —
        # under a held drag the user IS scrolled (the flag is already
        # truthy), so the end-of-load decision is identical without the
        # put-back, and the release settle converges the position when
        # the gesture ends.
        # F-A9 fourth-cut correction (2026-10-06 20:54 log): the same
        # fatal burst died a THIRD time with every scripted scroll now
        # drag-guarded (this guard + scroll_to's choke-point) — so the
        # third-cut attribution was incomplete: the one constant across
        # all three deaths (19:08, 19:42, 20:54) is the finalization
        # rebuild's own mass destroy+rebuild churn — every registered
        # widget (both mega-cards) destroyed with deferred deletions
        # pending while the whole visible band was constructed live —
        # removed by the view's F-A9 preserve-and-reuse _rebuild_sizer.
        # This guard stays: a scripted scroll beneath a held native
        # drag is a real hazard class (F-A2) regardless of the crash
        # mechanism.
        if (pre is not None and self._view is not None
                and not self._drag_active()):
            try:
                self._view.scroll_to(pre.view_start)
            except WX_FAILURES:
                pass
        on_complete = self._measure_all_on_complete
        self._measure_all_on_complete = None
        if on_complete is not None:
            on_complete()

    # ------------------------------------------------------------------
    # Fork-mode preview on the item model (§5.8)
    # ------------------------------------------------------------------

    def enter_fork_preview(self, fork_point: int, on_cancel: Optional[Callable] = None):
        self._fork_preview = True
        self._fork_preview_fork = self.conversation.fork_turn_state.latest_fork + 1
        self._fork_preview_point = fork_point
        visible = []
        for item in self._items:
            f, t = item.fork, item.turn
            if self.conversation.fork_turn_state.is_visible_with_extra_fork(
                    f, t, self._fork_preview_fork, fork_point):
                visible.append(item)
        if self._view is not None:
            try:
                self._sync_item_list(visible)
            except WX_FAILURES:
                self._logger.exception("fork preview entry sync failed")

    def exit_fork_preview(self, cancelled: bool = False):
        self._fork_preview = False
        if cancelled:
            # Cancel: restore every item in the model (§5.8).
            if self._view is not None:
                try:
                    self._sync_item_list(self._items)
                except WX_FAILURES:
                    self._logger.exception(
                        "fork preview cancel sync failed")
            return
        # Fork send: mutate the item model to the new fork's visible set so
        # old-fork continuation items past the fork point are destroyed and
        # do not reappear alongside the new fork's messages (§5.8).
        new_fork = getattr(self, '_fork_preview_fork', None)
        fork_point = getattr(self, '_fork_preview_point', 0)
        if new_fork is None:
            new_fork = self.conversation.fork_turn_state.latest_fork
        keep = []
        for item in self._items:
            f, t = item.fork, item.turn
            if self.conversation.fork_turn_state.is_visible_with_extra_fork(
                    f, t, new_fork, fork_point):
                keep.append(item)
        removed = [item for item in self._items if item not in keep]
        for item in removed:
            self._remove_item(item)
        if self._view is not None:
            try:
                self._sync_item_list(self._items)
            except WX_FAILURES:
                self._logger.exception("fork preview exit sync failed")
        self._fork_preview_fork = None
        self._fork_preview_point = 0

    # ------------------------------------------------------------------
    # Reload streaming driver (§7.1)
    # ------------------------------------------------------------------

    def set_tab(self, tab):
        """Bind the owning conversation tab (for footer suppression)."""
        self._tab = tab

    def has_running_loaded_state(self) -> bool:
        """True when any §7.1 step-0 loaded sequence state is still running.

        The §7.7 restart rule extends the input/Send lock to untracked orphan
        runs on reopen (assume alive until stream/tombstone evidence resolves
        them). A park resolution that settles on disk is mirrored back into
        these entries via ``resolve_loaded_sequence_state``, so a resolved
        chain stops reading as running here and the lock releases.
        """
        return any(
            getattr(state, 'status', None) == "running"
            for state in self._loaded_sequence_states.values()
        )

    def resolve_loaded_sequence_state(self, event_key: str, resolved_state) -> None:
        """Replace a §7.1 step-0 loaded state with its resolved snapshot (§7.7).

        Branches 1/2 of the tab's restart rule resolve a parked ``running``
        state on disk only (fresh ``load_state`` → ``save_state``) — the
        in-memory entry read by ``has_running_loaded_state()`` would
        otherwise keep the stale ``running`` status and hold the
        input/Send lock for the rest of the session. The tab calls this
        with the post-resolution disk state so the lock releases exactly
        when the disk says so.
        """
        self._loaded_sequence_states[event_key] = resolved_state

    def reorder_items_chronologically(self):
        """Sort the render order to match the watcher's timestamp sort (§7.4).

        The watcher's finalize-once predicate copy-assigns the timestamp sort
        on ``conversation.messages``, but the item model was built in the
        watcher's two-phase scan order (primary inputs -> primary outputs ->
        conditional), which clusters roles on reopen. Re-sort ``_items`` by
        each item's earliest message timestamp so the final flush renders in
        chronological order. The sort is stable: items with no timestamp keep
        their relative order at the tail.
        """
        def _earliest(item):
            ts = None
            if item.messages:
                try:
                    ts = min(m.timestamp for m in item.messages)
                except (TypeError, ValueError):
                    ts = None
            if ts is None and item.message is not None:
                ts = item.message.timestamp
            return (ts is None, ts)
        self._items.sort(key=_earliest)

    def load_existing(self):
        """Stream an existing conversation's backlog through the watcher (§7.1).

        Runs on the GUI thread (deferred by the main frame until after the
        page is added). Steps 0.5-4: snapshot stream sizes, begin load (footer
        suppression + scroll capture), empty short-circuit, seed IN/END facts,
        and start the watcher reload. Finalization is driven by the watcher's
        on_reload_complete (→ finalize_reload in the tab).
        """
        if self.conversation is None:
            return
        conversation = self.conversation

        # Step 0: non-resumable probe — sequence-state glob (§7.1 step 0,
        # §7.7, §7.8). Load every conditional_agents/*.pkl through the §7.7
        # load dispatch; any failure (unpickle error or an unrecognized
        # value) flags the workflow non-resumable before any short-circuit
        # or seed, and installs the permanent label immediately. Each
        # successfully loaded state is held for §7.4 step 8 keyed by its own
        # state.event_key.
        try:
            self._loaded_sequence_states = load_all_states(
                conversation.conversation_dir)
            self.non_resumable = False
        except Exception:
            self.non_resumable = True
            self._loaded_sequence_states = {}
            if self._tab is not None:
                try:
                    self._tab._install_non_resumable_label()
                except WX_FAILURES:
                    pass
        # Step 0.5: snapshot stream sizes before start_reload so the
        # finalize-once predicate captures the pre-reload backlog.
        conversation._refresh_stream_sizes()

        # Step 1: begin load (set the flag, suppress the footer, capture
        # scroll state).
        self._begin_load()

        # Step 2: empty short-circuit - no io streams at all.
        io_dir = conversation.io_dir
        has_streams = (
            any(io_dir.glob("*.in.txt"))
            or any(io_dir.glob("*.out.txt"))
            or any(io_dir.glob("*.end"))
        )
        has_conditional = any(
            p.is_dir() and p.name.startswith("conditional_")
            for p in io_dir.iterdir()
        ) if io_dir.exists() else False
        if not has_streams and not has_conditional:
            self._end_load()
            return

        # Step 3: seed the IN/END facts from filenames only (the watcher
        # upserts against this snapshot, §7.2).
        conversation.fork_turn_state = conversation.seed_in_end_facts(io_dir)

        # Step 4: start the watcher reload.
        if self._file_watcher is not None:
            try:
                self._file_watcher.start_reload(conversation.conversation_id)
            except ValueError:
                # Conversation not registered (closed before the deferred
                # kickoff) - settle back to the resting state.
                self._end_load()

    def _begin_load(self):
        """Begin a reload: set the flag, suppress the footer, capture scroll."""
        self.load_in_progress = True
        if self.conversation is not None:
            self.conversation.reload_in_progress = True
        self._user_scrolled = False
        self._load_first_paint_done = False
        # F-A7: no fling-gate state crosses a reload bracket (the mid-gesture
        # release settle is what clears it, and the bracket owns the viewport
        # from here).
        self._drag_fling_gated = False
        # Guard layer (reload crash class, §8.1): a stale force-bottom flag
        # from a previous load's aborted finalization (a fork-preview
        # early-return consumed neither the flag nor the final flush) must
        # not jump the frozen head window to a bottom range mid-load — that
        # mid-load bottom window rebuild is the head-spacer birth path.
        self._force_scroll_bottom = False
        if self._tab is not None and not self.non_resumable:
            try:
                self._tab.enter_reload_status_mode()
            except RuntimeError:
                pass

    def _end_load(self):
        """End a reload: clear the flag, apply the scroll decision, restore footer.

        Sets the §8.1 force-bottom flag when the user has not scrolled since the
        load began; ``_flush()`` consumes it through the scroll policy. A
        non-resumable conversation (step 0, §7.8) never restores the footer —
        the permanent label stays in place.
        """
        self.load_in_progress = False
        if self.conversation is not None:
            self.conversation.reload_in_progress = False
        self._force_scroll_bottom = not self._user_scrolled
        if self._view is not None:
            try:
                self._flush()
            except RuntimeError:
                pass
        if self._tab is not None:
            try:
                if self.non_resumable:
                    self._tab._install_non_resumable_label()
                else:
                    self._tab.exit_reload_status_mode()
            except RuntimeError:
                pass
