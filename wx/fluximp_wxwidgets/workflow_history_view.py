"""WorkflowHistoryView - ScrolledWindow owning history widgets and layout.

Phase 1: a wx.ScrolledWindow subclass that materializes history items
(RequestCards / MessagePanels) into a history sizer, lays out once per
controller flush, and forwards scroll/size events back to the controller.
"""

import wx
import wx.html
import html as html_lib
import logging
import time
from typing import Optional, Callable, Dict, List, Tuple, Any

from fluximp_wxwidgets.request_card import RequestCard
from fluximp_wxwidgets.conversation_manager import ConversationMessage, PermissionRequest
from fluximp_protocol.user_agent.file_io.defs import parse_utc_timestamp
from fluximp_wxwidgets.markdown_render import markdown_to_html
from fluximp_wxwidgets.workflow_history_controller import (
    HistoryItemId, ScrollState, WX_FAILURES,
)


from fluximp_wxwidgets import themes

logger = logging.getLogger(__name__)


class WorkflowHistoryView(wx.ScrolledWindow):
    """Owns the history sizer, materializes items, and handles layout/scroll."""

    MIN_CARD_WIDTH = 520
    # §8.2 card indents: every card row carries a left indent, and that
    # indent IS the row's sizer border (see _sizer_flags/_sizer_border).
    # Primary API Request cards take the shallower 20px indent; the
    # conditional-agent / system-agent cards (the amber ``-> name`` cards
    # and the scrutinator/summarizer-type ⚙ cards) take the deeper 40px
    # one.  Every row stays card_w wide: the widget is sized
    # card_w - indent and the indent carries the rest of the row.
    PRIMARY_INDENT_OFFSET = 20
    AGENT_INDENT_OFFSET = 40
    # F-A7/F-A9: the thumb is "parked" when no genuine THUMBTRACK arrives
    # for this many milliseconds while the drag stays latched — a
    # reading-paced drag re-arms the window continuously (its sparse gaps
    # also unlatch the fling gate, see _on_scroll), and a fling's parked
    # stop converges here (the posted-message poll, _drag_park_poll).
    DRAG_PARK_MS = 120
    # F-A7 (live-build flash kill, second cut): live shells birth
    # directly into this view at this far off-client position. The
    # native creation-time erase (inside CreateWindowEx the shell
    # constructors' colored children paint at the window's birth
    # position) is position-clipped out of the visible area, so the
    # flash kill needs neither the hidden-host birth NOR a Reparent()
    # into the view — the host-birth + Reparent variant exercised its
    # hierarchy change for the first time on the first post-load band
    # rebuild and hard-crashed natively (no Python exception, nothing
    # logged, the log tail dies mid card-construction burst; crash only
    # on that first post-load scroll per the user repro). Live widgets
    # are view children from native birth again, exactly the pre-F-A7
    # topology that never crashed.
    # Third-cut correction (2026-10-06 19:42 log): the SAME fatal burst
    # later died with this birth-position cut itself — no Reparent, same
    # storm site — so the Reparent was one hazard at a storm-vulnerable
    # site, not the mechanism. The surviving unique factor in that repro
    # was _finish_measure_all's scripted scroll firing beneath the
    # still-held native thumb drag (see scroll_to's F-A7 guard); the
    # birth-position topology is exonerated by both crash logs.
    # Fourth-cut correction (2026-10-06 20:54 log): the crash followed
    # the finalization burst a THIRD time with every scripted scroll
    # drag-guarded too — so neither of those was the whole mechanism:
    # the one constant across all three deaths is the rebuild's own
    # mass destroy+rebuild churn, now removed by F-A9's preserve-and-
    # reuse _rebuild_sizer (materialize's crash-fix note tracks this).
    SHELL_BIRTH_POS = (-10000, -10000)
    # F-A10 (revisit cache): bound on detached widgets kept alive for
    # scroll revisits — hidden native trees cost HWNDs and RAM
    # (rendered HTML internal representations), so the bound is a
    # genuine memory leash. The OLDEST cached entry evicts past it;
    # a far revisit then rebuilds once (the pre-cache cost).
    # User tuning (2026-10-06 test round): real conversation histories
    # run into the hundreds (a cursory test: 50 cards; a quick one:
    # 15), and a 10-widget horizon re-paid full shell+populate+
    # measure rebuilds on ordinary back-and-forth reading — raised to
    # 100, accepting the wider memory leash as an explicit trade for
    # revisit smoothness. FIFO eviction still applies past it, so
    # histories beyond the bound recycle their oldest widgets.
    DEMAT_CACHE_MAX = 100
    # Approach 1 (sizer bijection — the reload crash/squash fix): the two
    # aggregate load-window spacers are ORDINARY bookkeeping entries —
    # sentinel ids in _sizer_order at their real sizer positions (the
    # head first when present, the tail always last) — so every sizer
    # entry is order-tracked, index math can never be shifted by an
    # untracked spacer, and removal goes through _remove_sizer_entry like
    # any row. fork/turn -1 cannot collide with any real item id, and
    # these kinds never match a real item's kind.
    LOAD_HEAD_SPACER_ID = HistoryItemId('__load_head__', -1, -1)
    LOAD_TAIL_SPACER_ID = HistoryItemId('__load_tail__', -1, -1)

    # F-B4 (sizer-state instrumentation cap): a storm must not flood the
    # logs with dumps either; class defaults, instance counters.
    _SIZER_DUMP_CAP = 20
    _sizer_dump_count = 0
    _sizer_dump_suppressed = False

    def __init__(self, parent, logger: Optional[logging.Logger] = None,
                 app_settings=None):
        super().__init__(parent, style=wx.VSCROLL)
        self._logger = logger or logging.getLogger(__name__)
        # App-wide settings, forwarded to every RequestCard so the
        # View > API Request Card Metadata visibility toggles are honoured.
        self._app_settings = app_settings
        self.SetScrollRate(5, 5)

        self._controller = None
        self._tab = None
        self._history_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self._history_sizer)
        self._widgets_by_id: Dict[Any, wx.Window] = {}
        self._spacer_items: Dict[Any, wx.SizerItem] = {}
        self._sizer_order: List[Any] = []
        # F-A10 (revisit cache): dematerialized widgets cached here,
        # keyed by item id; a cached widget stays alive + registered
        # (hidden, out of the sizer), so a scroll revisit re-Adds it
        # instead of reconstructing the card shell+populate+measure
        # (the native crash medium and the chunky-rerender churn).
        # Dict insertion order doubles as the FIFO eviction order
        # (see _cache_dematerialized).
        # INVARIANT (the contract every cache site shares): a registry
        # entry in _widgets_by_id is EITHER a live sizer row OR a
        # _demat_cache member — never both, never neither. The revisit
        # cache is reachable ONLY through the registry (no reuse path
        # consults _demat_cache), so membership and registry entry must
        # always move together; _demat_cache_drop is the single
        # membership-pop helper every drop site goes through.
        self._demat_cache: Dict[Any, bool] = {}
        # F-A10 (dead-cache defect fix): ids whose cached widget was
        # mutated out of band while cached — a routing eager-build
        # through materialize's registry early return, or the flush's
        # expansion policy landing on a hidden card. Cached widgets are
        # skipped by the per-flush measure/width passes (idle by
        # design), so a mutation converges exactly once: a flagged id
        # re-measures hidden on the next refresh_materialized_heights,
        # records its height, and clears the flag; unflagged members
        # stay idle (the skip is the point of the cache).
        self._demat_dirty: set = set()
        self._items_by_id: Dict[Any, Any] = {}
        self._permission_response_handler: Optional[Callable] = None
        self._edit_handler: Optional[Callable] = None
        self._size_timer: Optional[wx.Timer] = None
        self._width = 0

        # Programmatic-scroll echo suppression (§8.6): SetVirtualSize / Scroll
        # driven by this view must not reach the controller as user scrolls.
        # Counting latch (F-A3), not a boolean: each programmatic site takes
        # one reference and releases it one event-loop turn later, so
        # overlapping scripted scrolls never open a window in which a
        # delayed clamp echo masquerades as a user scroll.
        self._scroll_echo_depth = 0
        # F-A1/A2: thumb-drag gesture state. THUMBTRACK events never
        # recompute inline — one coalesced settle runs per event-loop
        # turn, layout_history() freezes the scrollbar range for the
        # whole gesture, and the release settle re-applies it
        # (controller.settle_thumb_drag) with a forced paint.
        self._thumb_drag_active = False
        self._drag_settle_pending = False
        # F-A9 (fourth cut — park convergence): a gated fling sweep greys
        # through spacer territory by design — building whole visible sets
        # per settle turn is what starves WM_PAINT — so the parked position
        # converges the moment the thumb STOPS, not only when it is
        # released. The carrier is a POSTED-MESSAGE CallAfter poll, never a
        # wx.Timer: the first cut's one-shot WM_TIMER never reached the
        # handler while the native MSW scrollbar drag held the loop (the
        # 20:46/20:54 repro: park the thumb mid-drag and the whole message
        # area stays grey until a move or release — only posted events flow
        # mid-drag; the F-A1 settle steps prove CallAfter does). The poll
        # is armed per genuine track (_defer_drag_park) and lands in
        # park_thumb_drag + force_paint — a pause becomes a readable page
        # while the thumb is still held.
        self._drag_park_polling = False
        self._drag_park_seen_stamp = None
        # F-A9: the latest genuine THUMBTRACK's perf_counter stamp — feeds
        # the park window and the reading-pace fling unlatch in _on_scroll.
        self._last_drag_track_perf = None
        # Windowed reload rendering (§8.5): while a backlog streams, the sizer
        # holds only items[_load_first.._load_last] plus aggregate head/tail
        # spacers, never one entry per backlog item.
        self._load_window_active = False
        self._load_first = 0
        self._load_last = -1
        self._load_head_spacer: Optional[wx.SizerItem] = None
        self._load_tail_spacer: Optional[wx.SizerItem] = None
        # Never-shown measurement host (§7.4 step-5 transients):
        # materialize(hidden=True) shells are parented here — never into
        # this live view. wxWidgets child windows are created shown at
        # native creation, and the shell constructors (RequestCard /
        # MessagePanel) synchronously build further native children with
        # per-role colors before any Hide() could take effect, so transient
        # shells born inside the live view kept flashing a small square of
        # card colors at the first card's top-left through the whole
        # measure pass. A child of a hidden parent is invisible at native
        # creation; measurement is unaffected (the §8.2 width is assigned
        # explicitly and SetSize/Layout/GetBestSize are parent-independent).
        # Built 0x0 and hidden right here at view construction — mid tab
        # assembly, before any paint, so the host itself can never paint a
        # single pixel either — and never added to the history sizer; it is
        # a plain child of this view and is destroyed with it.
        self._measure_host: Optional[wx.Panel] = wx.Panel(self, size=(0, 0))
        self._measure_host.Hide()
        self.Bind(wx.EVT_SCROLLWIN, self._on_scroll)
        self.Bind(wx.EVT_SIZE, self._on_size)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_controller(self, controller):
        self._controller = controller

    def set_tab(self, tab):
        self._tab = tab

    def set_permission_response_handler(self, handler: Optional[Callable]):
        self._permission_response_handler = handler

    def set_edit_handler(self, handler: Optional[Callable]):
        self._edit_handler = handler

    def is_valid(self) -> bool:
        """True while the tab is open and the native window is alive."""
        try:
            if self._tab is not None and getattr(self._tab, '_closed', False):
                return False
            if self.IsBeingDeleted():
                return False
        except RuntimeError:
            return False
        return True

    # ------------------------------------------------------------------
    # Widget registry
    # ------------------------------------------------------------------

    def get_widget(self, item_id):
        return self._widgets_by_id.get(item_id)

    def refresh_api_metadata_display(self):
        """Re-render metadata on every materialized request card.

        Called when the View > API Request Card Metadata submenu changes so
        already-built cards reflect the new visibility without a rebuild.
        Detached-cache members (hidden but registered) are refreshed too
        and flagged via mark_out_of_band_mutation so their recorded heights
        reconverge on the next flush.
        """
        for item_id, widget in list(self._widgets_by_id.items()):
            if not isinstance(widget, RequestCard):
                continue
            try:
                widget.refresh_api_metadata_display()
            except (RuntimeError, WX_FAILURES):
                continue
            if item_id in self._demat_cache:
                self.mark_out_of_band_mutation(item_id)

    def materialized_item_ids(self):
        """Ids of currently materialized items (windowed flush policy loop)."""
        return list(self._widgets_by_id.keys())
    
    def ensure_widget(self, item):
        """Backward-compatible alias for materialize() (§8.6)."""
        return self.materialize(item)

    def _build_widget_shell(self, item, parent=None, pos=wx.DefaultPosition):
        """Build the widget shell WITHOUT content (§8.5).

        Cards build their RequestCard header-only shell; panels build a
        defer_content=True MessagePanel. Content (SetPage) is populated by
        ``_populate_widget_content`` after materialize() assigns the fixed
        §8.2 width, so the synchronous height read is deterministic.

        ``parent`` overrides the shell's parent window — the never-shown
        measurement host for hidden transient builds. RequestCard and
        MessagePanel take their parent as the first constructor argument
        and build every child of theirs under it (RequestCard._new_section
        parents each CollapsibleSection to the card itself, and every panel
        child is a MessagePanel child), so a host-parented shell's whole
        native subtree is born inside the hidden host and none of it can
        ever erase or paint into the live view.

        ``pos`` (F-A7) is the shell's BIRTH position, forwarded into the
        wx constructor. Live builds pass the far off-client
        SHELL_BIRTH_POS: the constructor-time native erase inside
        CreateWindowEx paints at the window's birth position, so a shell
        born outside the parent's visible client area paints nothing
        there (the flash kill WITHOUT a Reparent() hierarchy change — the
        host-birth + Reparent variant hard-crashed natively on the first
        post-load band rebuild). Hidden builds keep wx.DefaultPosition
        inside the never-shown host, where the hidden parent already
        clips everything.
        """
        kind = item.item_id.kind
        shell_parent = parent or self
        if kind == 'card':
            return self._create_card_shell(item, shell_parent, pos)
        if kind == 'error_panel':
            return self._build_error_panel(item, shell_parent, pos)
        return self._build_panel(item, shell_parent, pos)

    def _populate_widget_content(self, item, widget):
        """Populate widget content after the fixed width is assigned (§8.5)."""
        if item.item_id.kind == 'card':
            self._replay_card(item, widget)
        elif hasattr(widget, 'populate_content'):
            try:
                widget.populate_content()
            except WX_FAILURES:
                self._logger.exception(
                    "populate_content failed for %r", item.item_id)

    def _create_card_shell(self, item, parent=None, pos=wx.DefaultPosition):
        return RequestCard(
            parent or self,
            fork=item.fork,
            turn=item.turn,
            metadata=dict(item.metadata),
            on_permission_response=self._permission_response_handler,
            on_user_toggle=lambda role, ordinal, expanded: self._on_user_toggle(item, role, ordinal, expanded),
            is_conditional=item.is_conditional,
            conditional_agent_name=item.conditional_agent_name,
            is_system_agent=item.is_system_agent,
            system_agent_type=item.system_agent_type,
            logger=self._logger,
            app_settings=self._app_settings,
            pos=pos,
        )

    def _replay_card(self, item, card):
        has_response = False
        for m in item.messages:
            role = m.role
            md = m.metadata
            try:
                if role == 'assistant' and md.get('is_api_response'):
                    resp_meta = {k: v for k, v in md.items()
                                 if k not in ('reasoning', 'is_api_response')}
                    card.set_response(m.content, md.get('reasoning', ''), metadata=resp_meta)
                    has_response = True
                elif role == 'assistant' and md.get('is_conditional_report'):
                    card.add_report_section(m.content)
                    has_response = True
                elif role == 'tool':
                    card.add_tool_use(m)
                    if md.get('status') == 'completed':
                        card.update_tool_result(m)
                elif role == 'permission_request':
                    resolved = False
                    if self._controller is not None:
                        # §5.5: during a reload the backlog's permissions render
                        # resolved (deactivated); mirror the live routing's
                        # load_in_progress suppression on rebuild.
                        resolved = (self._controller.is_turn_terminal(item.fork, item.turn)
                                    or self._controller.load_in_progress)
                    card.add_permission_request(m, resolved=resolved)
                elif role == 'system_agent':
                    if item.is_system_agent:
                        # The standalone system-agent card's own record is
                        # its response, not an embedded subcard; rebuild the
                        # completed detail on every rematerialize exactly as
                        # the live routing's completion does (a reloaded card
                        # used to keep its creation-time pending look).
                        if md.get('status') != 'running' and not has_response:
                            card.set_response('', metadata=dict(md), collapsed=True)
                            has_response = True
                    else:
                        agent_type = md.get('system_agent_type', '')
                        # Presence-based (the reload fix): a system-agent
                        # record implies its subcard. The wire layer mutates
                        # the ONE record in place at STOPPED, so the old
                        # status=='running' creation gate made the subcard
                        # unreconstructable on every rebuild — reload
                        # finalization, scroll-off and back; only the widget
                        # materialized at live STARTED ever showed it. Add
                        # like the tool records above, then apply the
                        # terminal state (the routing append is
                        # identity-deduped, so each run renders once).
                        card.add_system_agent_subcard(agent_type, md)
                        if md.get('status') != 'running':
                            card.update_system_agent_subcard(agent_type, md)
                elif role == 'error' and md.get('run_terminal'):
                    card.set_response(m.content, metadata=dict(md))
                    has_response = True
            except WX_FAILURES:
                break
        if item.pending and not has_response and not item.pending_muted:
            card.set_pending()
        if item.pending_muted and not has_response:
            card.set_response('')

    def _build_panel(self, item, parent=None, pos=wx.DefaultPosition):
        message = item.message
        on_edit = None
        # §5.8 view gate: the fork/edit button is withheld at materialize time
        # while a reload streams (load_in_progress). Panels materialized during
        # a reload therefore render without the live "Fork" button; after
        # finalization the reload settles and the button is available on a
        # future re-materialize.
        if (message is not None and message.role == 'user'
                and self._edit_handler is not None
                and (self._controller is None
                     or (not self._controller.load_in_progress
                         and not self._controller.non_resumable))):
            on_edit = lambda m=message: self._edit_handler(m)
        return MessagePanel(parent or self, message, on_edit,
                            defer_content=True, pos=pos)

    def _build_error_panel(self, item, parent=None, pos=wx.DefaultPosition):
        msg = item.message
        if msg is None:
            from datetime import datetime, timezone
            msg = ConversationMessage(
                fork=item.fork, turn=item.turn, role='assistant',
                content=item.error_text or "",
                timestamp=datetime.now(timezone.utc),
                metadata={'is_error': True})
        return MessagePanel(parent or self, msg, defer_content=True, pos=pos)

    def _on_user_toggle(self, item, role: str, ordinal: int, expanded: bool):
        """§8.3: a user toggle carries the section's per-instance ordinal
        (defect B1 — the role-only key cascaded one click to every
        same-role sibling on the card)."""
        if self._controller is not None:
            self._controller.on_user_toggle(item.item_id, role, ordinal,
                                            expanded)

    # ------------------------------------------------------------------
    # Item list / layout
    # ------------------------------------------------------------------

    def apply_item_list(self, items, first=None, last=None):
        """§8.6: reconcile widgets and spacers to the ordered item list."""
        if self._controller is None:
            return
        if first is None or last is None:
            first, last = 0, max(0, len(items) - 1)
        # F-B1 (desync-storm birth gate, 2026-10-07 11:59 log): two model
        # records sharing one HistoryItemId (a killed-at-close run's
        # streamed terminal records plus the close-written tombstone)
        # inserted the id twice into every sizer bookkeeping structure —
        # the second Add fired "window already in a sizer", aborted its
        # pass, and re-created the order/native asymmetry the integrity
        # gate kept repairing (32/33 -> 31/33 every cycle). Normalize
        # BEFORE any bookkeeping derives from the list: first occurrence
        # wins, the rest are refused and logged (the log is the
        # discriminator between the duplicate-model mechanism and the
        # detach/insert mechanism). Callers derive first/last from the
        # model list, which post-F-B4 (_add_item's refusal) can never
        # hold duplicates; normalizing here is pure defense for any
        # other caller plus the belt for the model-level gap.
        seen = set()
        deduped = []
        dup_ids = []
        for item in items:
            dup_iid = item.item_id
            if dup_iid in seen:
                dup_ids.append(dup_iid)
                continue
            seen.add(dup_iid)
            deduped.append(item)
        if dup_ids:
            self._logger.error(
                "F-B1: duplicate item ids in the model list (kept first, "
                "refused rest): %r", dup_ids)
            items = deduped
        n = len(items)
        wanted_ids = [item.item_id for item in items]
        wanted_set = set(wanted_ids)

        # Track item objects for height re-reads and measurement.
        for item in items:
            self._items_by_id[item.item_id] = item

        # Leaving windowed reload mode (the final flush after _end_load):
        # the sizer holds only the load window, so force one structural
        # rebuild into the full item list.
        if self._load_window_active:
            self._load_window_active = False
            self._rebuild_sizer(items, first, last)
            return
        # Integrity gate (guard layer): every fast path below walks sizer
        # index i as item index i, valid only while the native sizer's
        # child count matches the bookkeeping. Under Approach 1 the order
        # mirrors every sizer entry (rows, offscreen spacers, and the
        # load-window sentinels), so a mismatch means desynced state
        # arrived from some swallowed failure outside the rebuild paths
        # (e.g. a window destroyed while attached and auto-removed by
        # wxWidgets): repair with one structural rebuild instead of
        # asserting inside GetItem or destroying live widgets at shifted
        # indices — the reload crash/squash cascade this view no longer
        # permits.
        sizer_count = self._sizer_count()
        if sizer_count is None or sizer_count != len(self._sizer_order):
            self._logger.error(
                "history sizer desync: %s sizer entries vs %d order "
                "entries — structural rebuild", sizer_count,
                len(self._sizer_order))
            self._dump_sizer_state(
                "integrity gate: %s native vs %d order"
                % (sizer_count, len(self._sizer_order)))
            self._rebuild_sizer(items, first, last)
            return
        # Drop sizer entries whose id is no longer wanted (rare removals).
        for iid in list(self._sizer_order):
            if iid not in wanted_set:
                self._remove_sizer_entry(iid)
                # F-A10 leak (removed ids): the entry detach only touches
                # the (spacer or row) slot, so a removed id's widget parked
                # in the detached revisit cache used to outlive this loop
                # — registered + cached for an id the model no longer
                # holds, invisible to the reconcile loop (it iterates the
                # new item list only) until some much later structural
                # rebuild's unwanted sweep happened to run. Evict the
                # member: the id and its hidden native tree die together.
                if iid in self._demat_cache:
                    self._demat_cache_evict(iid)

        # Fast path: sizer order already matches -> only state may differ.
        if self._sizer_order == wanted_ids:
            self._reconcile_states(items, first, last)
            return

        # Common path: new items appended at the end (live appends and
        # streaming batches) — append only the missing trailing entries and
        # keep every existing widget/spacer untouched (§8.5).
        if self._sizer_order == wanted_ids[:len(self._sizer_order)]:
            for i in range(len(self._sizer_order), n):
                item = items[i]
                iid = item.item_id
                if first <= i <= last and not getattr(item, 'suppressed', False):
                    widget = self.materialize(item)
                    if widget is not None and iid in self._demat_cache:
                        # Guard layer: a widget Added LIVE here can never
                        # be a detached-cache member (an appended id is
                        # new to the order, and every removal path evicts
                        # the removed id's cached member) — but if that
                        # invariant ever broke, stale membership would
                        # silently skip this now-live row in every
                        # per-flush width/measure pass. Enforce the same
                        # bijection guard (registry entry = live row OR
                        # cache member, never both) the window-mode
                        # insert loops and the rebuild add loop carry
                        # (defensive; unreachable today).
                        self._demat_cache_drop(iid)
                    if widget is None:
                        self._spacer_items[iid] = self._history_sizer.Add(
                            (0, self._height_for_item(item)), 0, 0)
                    elif self._safe_widget_add(widget, item) is None:
                        # F-B3: refused/failed attach degrades to the
                        # spacer lane (the id keeps its order slot; the
                        # append pass is never aborted).
                        self._spacer_items[iid] = self._history_sizer.Add(
                            (0, self._height_for_item(item)), 0, 0)
                else:
                    self._spacer_items[iid] = self._history_sizer.Add(
                        (0, self._height_for_item(item)), 0, 0)
                self._sizer_order.append(iid)
            self._reconcile_states(items, first, last)
            return

        # Structural change (removal / reorder): rebuild.
        self._rebuild_sizer(items, first, last)

    def _reconcile_states(self, items, first, last):
        for i, item in enumerate(items):
            iid = item.item_id
            in_range = first <= i <= last and not getattr(item, 'suppressed', False)
            widget = self._widgets_by_id.get(iid)
            if in_range:
                if widget is not None:
                    # Guard layer (the same straggler class the rebuild
                    # add loop carries): a wx-side-destroyed wrapper can
                    # sit in the registry until native idle and can
                    # never be re-Added — a Destroy()'d window is not
                    # Insert()able. Treat it as a miss and rebuild.
                    # With F-A10 members registered again this lane now
                    # serves cached widgets on every revisit, so the
                    # guard belongs here as well (defensive: no site of
                    # ours destroys a cached member out of band).
                    try:
                        if widget.IsBeingDeleted():
                            self._widgets_by_id.pop(iid, None)
                            self._demat_cache_drop(iid)
                            widget = None
                    except (RuntimeError, WX_FAILURES):
                        self._widgets_by_id.pop(iid, None)
                        self._demat_cache_drop(iid)
                        widget = None
                if widget is None:
                    widget = self.materialize(item)
                if widget is None:
                    continue
                if self._sizer_window_at(i) is not widget:
                    self._replace_sizer_at(i, item, widget)
            else:
                if widget is not None and iid not in self._demat_cache:
                    # Live row leaving the padded window: dematerialize
                    # into the bounded revisit cache (F-A10) — the one
                    # call that transitions a row into a cache member. A
                    # non-member registered widget here is either a live
                    # row leaving range (cache it) or an eagerly built
                    # routing/measure transient over a spacer slot
                    # (_replace_sizer_at's orphan cleanup, §8.5).
                    self._replace_sizer_at(i, item, None)
                elif iid in self._demat_cache:
                    # F-A10 dead-cache defect fix, second half: a
                    # detached-cache member is ALREADY a plain spacer
                    # slot. Reconcile runs over every item on every
                    # flush, and a member's slot holds no window, so
                    # re-running the dematerialize against it would
                    # make any identity guard vacuously true — the
                    # cached widget popped-and-destroyed ONE FLUSH
                    # after caching, and no member could ever survive
                    # to serve a revisit (the one-flush cache
                    # demolition; the spacer detach/re-insert was pure
                    # churn on top). A live member stays put — spacer
                    # intact, widget hidden and registered; a dead one
                    # drops its bookkeeping so the FIFO slot cannot
                    # squat.
                    self._cleanse_dead_member(iid)

    def _rebuild_sizer(self, items, first, last):
        """Structural rebuild (true reorder / removal, and the
        load-window -> full-sizer transition at finalization).

        F-A9 (fourth crash cut, 2026-10-06 20:54 log): a wanted widget
        is DETACHED AND REUSED — never mass destroyed + rebuilt in
        place. The old teardown destroyed EVERY registered widget in
        one synchronous pass (the entry loop's _hide_and_destroy with
        deferred native deletions pending, plus the blanket orphan
        sweep) and the add loop then live-built the whole visible band
        — at the head that is both mega-cards, several hundred native
        windows apiece alive twice over: replacements constructed while
        the old trees' deletions were still deferred. Three silent
        native deaths landed inside exactly that pass (19:08 with the
        Reparent cut, 19:42 with the scripted-scroll-under-drag hazard,
        20:54 with BOTH eliminated — that log: the fatal burst rebuilt
        items 0 and 1 live, dying ~650 ms in at the second card's
        sections), so the churn ITSELF is the storm, the one constant
        across all three. Preservation reduces the transition to ZERO
        destroys plus a few light builds in the typical case (the
        band's existing widgets are re-Added, not re-created), and the
        mega-cards' ~650 ms rebuild stall dies with it.
        """
        wanted_set = {item.item_id for item in items}
        # Detach every current sizer entry; wanted widgets survive the
        # detach hidden (the add loop below re-Adds them), everything
        # else is destroyed as before.
        for iid in list(self._sizer_order):
            self._remove_sizer_entry(iid, preserve=wanted_set)
        self._clear_load_window_entries()
        # Guard layer: a removal that failed keeps its native slot
        # attached (logged); normalize before the add loop so no straggler
        # shifts the rebuilt list's indices.
        self._drop_straggler_entries()
        self._destroy_unwanted_widgets(wanted_set)
        self._sizer_order = []
        for i, item in enumerate(items):
            iid = item.item_id
            in_range = first <= i <= last and not getattr(item, 'suppressed', False)
            if in_range:
                widget = self._widgets_by_id.get(iid)
                if widget is not None:
                    # Guard layer: a wx-side-destroyed straggler can never
                    # be re-Added — a Destroy()'d window still sits in the
                    # registry until idle; treat it as a miss and rebuild.
                    try:
                        if widget.IsBeingDeleted():
                            self._widgets_by_id.pop(iid, None)
                            # F-A10: a dead member's bookkeeping drops
                            # with the registry entry, or the stale
                            # membership would skip the fresh rebuild as
                            # a cached row forever.
                            self._demat_cache_drop(iid)
                            widget = None
                    except (RuntimeError, WX_FAILURES):
                        self._widgets_by_id.pop(iid, None)
                        self._demat_cache_drop(iid)
                        widget = None
                if widget is not None:
                    # F-A10: a reused widget is live in the sizer again
                    # — detached-cache membership drops with the detach
                    # state (membership means detached; a pending
                    # mutation flag drops with it too — live rows
                    # converge through the ordinary per-flush passes).
                    self._demat_cache_drop(iid)
                if widget is None:
                    widget = self.materialize(item)
                if widget is None:
                    self._spacer_items[iid] = self._history_sizer.Add(
                        (0, self._height_for_item(item)), 0, 0)
                elif self._safe_widget_add(widget, item) is None:
                    # F-B3: refused/failed attach degrades to the spacer
                    # lane — the id keeps its order slot and the rebuild
                    # loop can never abort mid-pass (the aborted mid-loop
                    # add was what re-created the storm's +1 every cycle).
                    self._spacer_items[iid] = self._history_sizer.Add(
                        (0, self._height_for_item(item)), 0, 0)
            else:
                # Wanted but off-range: dematerialize at this slot per
                # the §8.5 registry — F-A9 only preserves widgets the
                # new band shows IN range; off-range ones take the same
                # destination as the reconcile path's dematerialize
                # branch, which F-A10 routes to the bounded revisit
                # cache instead of a destroy.
                widget = self._widgets_by_id.get(iid)
                if widget is not None:
                    # F-A10: dematerialize-to-cache (bounded revisit
                    # cache), never destroy — a revisit re-Adds it
                    # instead of shell+populate+measure reconstruction.
                    self._cache_dematerialized(iid, widget)
                self._spacer_items[iid] = self._history_sizer.Add(
                    (0, self._height_for_item(item)), 0, 0)
            self._sizer_order.append(iid)
        # F-B4 tripwire: the rebuild is the designated repair — verify it
        # converged; a leftover means the add loop degraded (F-B3 refusals
        # keep counts, so this should be impossible post-F-B1/B2) and the
        # state dump names exactly which entries did not land.
        count = self._sizer_count()
        if count is None or count != len(self._sizer_order):
            self._logger.error(
                "F-B4: rebuild left %s native entries vs %d order "
                "entries — convergence failed", count,
                len(self._sizer_order))
            self._dump_sizer_state("post-rebuild divergence")

    # ------------------------------------------------------------------
    # Windowed reload rendering (§8.5)
    # ------------------------------------------------------------------

    def apply_load_window(self, items, first, last, total_h, measure=False):
        """Windowed reload mode: sizer = items[first..last] + aggregate spacers.

        While a backlog streams, the native layout must stay O(window): the
        sizer holds only the frozen head window — the first user panel alone
        (§5.10.1: the sole pre-finalization display requirement is the
        near-instant first user message; the rest of the backlog stays
        virtual until finalization) — plus one aggregate head spacer (if
        first > 0) and one aggregate tail spacer carrying the remaining
        backlog height — never one entry per backlog item. The full backlog
        height reaches the scrollbar via set_virtual_height() after
        layout_history().

        Growth fast path: newly in-range items (rare — the window is frozen
        during load; it fires on mode entry) are inserted before the tail
        spacer without touching any existing entry, so streaming flushes
        never rebuild or flicker visible widgets.
        """
        n = len(items)
        if n == 0 or first > last:
            # Nothing to show: an empty window keeps just the aggregate tail.
            if self._load_window_active:
                self._set_load_tail_spacer(max(0, int(total_h)))
            else:
                self._rebuild_load_window(items, 0, -1, 0, max(0, int(total_h)))
            self._load_first, self._load_last = 0, -1
            return
        # _items_by_id is populated incrementally (materialize + the window
        # insert loops); an O(n) fill here would run on every streaming flush.
        head_h = items[first].last_y if 0 <= first < n else 0
        tail_h = (int(total_h) - items[last + 1].last_y) if 0 <= last + 1 < n else 0
        if (self._load_window_active and first == self._load_first
                and last >= self._load_last):
            # Growth fast path: append newly in-range items before the tail
            # spacer; the head spacer and existing entries stay untouched.
            for i in range(self._load_last + 1, last + 1):
                item = items[i]
                iid = item.item_id
                self._items_by_id[iid] = item
                widget = (self._widgets_by_id.get(iid)
                          or self.materialize(item, measure=measure))
                if widget is not None and iid in self._demat_cache:
                    # Guard layer: a widget inserted LIVE here can never
                    # be a detached-cache member (mode entry flushes the
                    # whole cache; nothing dematerializes-to-cache in
                    # window mode) — but if that invariant ever broke,
                    # stale membership would silently skip this now-live
                    # row in every per-flush width/measure pass. Enforce
                    # the bijection (registry entry = live row OR cache
                    # member, never both) anyway (defensive; unreachable
                    # today).
                    self._demat_cache_drop(iid)
                idx = self._load_tail_index()
                if widget is None:
                    self._spacer_items[iid] = self._history_sizer.Insert(
                        idx, (0, self._height_for_item(item)), 0, 0)
                elif self._safe_widget_add(widget, item, i=idx) is None:
                    # F-B3: refused/failed attach degrades to the spacer
                    # lane — the id keeps its order slot, the loop is
                    # never aborted mid-pass.
                    self._spacer_items[iid] = self._history_sizer.Insert(
                        idx, (0, self._height_for_item(item)), 0, 0)
                # Approach 1: the sizer slot was inserted before the
                # aggregate tail spacer, so the order must mirror that —
                # insert the id before the tail sentinel, never past it.
                if (self._sizer_order
                        and self._sizer_order[-1] == self.LOAD_TAIL_SPACER_ID):
                    self._sizer_order.insert(len(self._sizer_order) - 1, iid)
                else:
                    # Invariant breach (sentinel missing): append, log, and
                    # let apply_item_list's integrity gate repair the
                    # bijection structurally on the next pass.
                    self._sizer_order.append(iid)
                    self._logger.error(
                        "load-window growth: tail sentinel missing "
                        "(order len %d)", len(self._sizer_order))
            self._set_load_tail_spacer(max(0, tail_h))
        else:
            # Range moved or shrank (user scrolled, removals): full rebuild
            # of the window (still only O(window) entries).
            self._rebuild_load_window(items, first, last, head_h, max(0, tail_h),
                                      measure=measure)
        self._load_first, self._load_last = first, last

    def _rebuild_load_window(self, items, first, last, head_h, tail_h,
                             measure=False):
        """Full rebuild of the load window (range change / mode entry).

        Approach 1: the aggregate head/tail spacers are registered here
        as ordinary order entries (sentinel ids at their real sizer
        positions), so the removal loop above detaches them through
        _remove_sizer_entry like any row and the bijection holds through
        the rebuild.
        """
        for iid in list(self._sizer_order):
            self._remove_sizer_entry(iid)
        self._clear_load_window_entries()
        # Guard layer: a removal that failed keeps its native slot
        # attached (logged); normalize before the add loop so no straggler
        # shifts the rebuilt window's indices.
        self._drop_straggler_entries()
        self._destroy_orphan_widgets()
        self._sizer_order = []
        if head_h > 0:
            self._load_head_spacer = self._history_sizer.Add((0, head_h), 0, 0)
            self._spacer_items[self.LOAD_HEAD_SPACER_ID] = self._load_head_spacer
            self._sizer_order.append(self.LOAD_HEAD_SPACER_ID)
        for i in range(first, last + 1):
            item = items[i]
            iid = item.item_id
            self._items_by_id[iid] = item
            widget = (self._widgets_by_id.get(iid)
                      or self.materialize(item, measure=measure))
            if widget is not None and iid in self._demat_cache:
                # Guard layer: a widget re-placed LIVE here can never be a
                # detached-cache member (window-mode entry destroys the
                # whole cache, and nothing dematerializes-to-cache in
                # window mode) — but if that invariant ever broke, the
                # stale membership would silently skip this now-live row
                # in every per-flush width/measure pass. Enforce the
                # bijection (registry entry = live row OR cache member,
                # never both) anyway (defensive; unreachable today).
                self._demat_cache_drop(iid)
            if widget is None:
                self._spacer_items[iid] = self._history_sizer.Add(
                    (0, self._height_for_item(item)), 0, 0)
            elif self._safe_widget_add(widget, item) is None:
                # F-B3: refused/failed attach degrades to the spacer lane
                # — the id keeps its order slot, the loop is never
                # aborted mid-pass.
                self._spacer_items[iid] = self._history_sizer.Add(
                    (0, self._height_for_item(item)), 0, 0)
            self._sizer_order.append(iid)
        # The tail spacer is always present in window mode so the growth
        # fast path has a stable insert position (height 0 at the tail);
        # Approach 1 tracks it as the LAST order entry, mirroring the
        # sizer.
        self._load_tail_spacer = self._history_sizer.Add((0, max(0, tail_h)), 0, 0)
        self._spacer_items[self.LOAD_TAIL_SPACER_ID] = self._load_tail_spacer
        self._sizer_order.append(self.LOAD_TAIL_SPACER_ID)
        self._load_window_active = True

    def _load_tail_index(self) -> int:
        """Sizer index of the aggregate tail spacer.

        Approach 1 keeps the order mirrored (the tail sentinel is tracked
        in _sizer_order), so while window mode is active the tail spacer
        is the last sizer entry by construction — the insert position the
        growth fast path inserts before.
        """
        try:
            return max(0, self._history_sizer.GetItemCount() - 1)
        except WX_FAILURES:
            return 0

    def _set_load_tail_spacer(self, height: int):
        """Reassign the aggregate tail spacer height.

        Guard layer: never null the handle on failure — the spacer's sizer
        entry stays attached, and a nulled handle used to strand it
        permanently untracked (a reload leak launch point). The entry
        stays reachable through its sentinel in _sizer_order; log loudly.
        """
        if self._load_tail_spacer is None:
            return
        try:
            self._load_tail_spacer.AssignSpacer((0, max(0, int(height))))
        except WX_FAILURES:
            self._logger.exception("load tail spacer height assign failed")

    def _clear_load_window_entries(self):
        """Reset the windowed-reload state (§8.5).

        Approach 1: the aggregate head/tail spacers are tracked as sentinel
        entries in _sizer_order (LOAD_HEAD_SPACER_ID / LOAD_TAIL_SPACER_ID),
        so the order-driven removal loops in _rebuild_sizer /
        _rebuild_load_window have already detached them through
        _remove_sizer_entry by the time this runs — nothing here detaches
        by SizerItem-wrapper identity. The old identity search silently
        leaked a stale spacer above the rebuilt list whenever wrapper
        identity did not hold, and that +1 index shift was the reload
        crash/squash root cause (it is gone).
        """
        self._load_head_spacer = None
        self._load_tail_spacer = None
        self._load_window_active = False
        self._load_first = 0
        self._load_last = -1

    def _drop_straggler_entries(self):
        """Guard layer: normalize the sizer to empty before an add loop.

        Entry removal is best-effort — each failure keeps its native slot
        attached and logs loudly. A leftover attached entry would shift
        every index of the rebuilt list (the same desync class the
        integrity gate in apply_item_list repairs), so drop all remaining
        items synchronously. Clear(delete_windows=False) keeps the windows
        alive for the caller's _destroy_orphan_widgets() pass right after.
        """
        leftover = self._sizer_count()
        if leftover:
            self._logger.error(
                "history sizer desync on rebuild: %s entries left after "
                "the removal loop — clearing", leftover)
            self._dump_sizer_state(
                "stragglers before Clear after removal loop (%s)"
                % (leftover,))
            try:
                self._history_sizer.Clear(False)
            except WX_FAILURES:
                self._logger.exception("history sizer Clear failed")

    def _hide_and_destroy(self, widget):
        """Hide, then Destroy — flicker hardening for every widget removal.

        wxWidgets' Destroy() defers the native window delete to idle
        processing, so a visible window keeps painting — at whatever
        position it currently holds — until then. Every removal site now
        hides the widget first, so the deferred-deletion interval is
        invisible: the §7.4 step-5 measure transients (built hidden by
        measure_slice anyway), the finalization load-window -> full-sizer
        transition (the frozen first card's widget is destroyed and rebuilt
        in the same flush — previously it overlapped its replacement on
        screen until idle), and the structural rebuild paths never flash
        a stale copy.
        """
        try:
            if hasattr(widget, 'stop_pending_timer'):
                widget.stop_pending_timer()
        except Exception:
            pass
        try:
            if not widget.IsBeingDeleted():
                widget.Hide()
        except RuntimeError:
            pass
        try:
            if not widget.IsBeingDeleted():
                widget.Destroy()
        except RuntimeError:
            pass

    def _destroy_orphan_widgets(self):
        """Destroy widgets registered but not in the sizer (§8.5).

        Routing handlers and measure passes may build widgets that never land
        in the sizer (deferred rebuilds, offscreen measurements). A
        structural rebuild must destroy them so native handles cannot leak.
        """
        for iid, widget in list(self._widgets_by_id.items()):
            self._hide_and_destroy(widget)
            self._widgets_by_id.pop(iid, None)
        # F-A10: this blanket sweep (windowed-reload mode entry) destroys
        # every registered widget — cached ones included, since they are
        # registered — so the cache bookkeeping must reset with it
        # (pending-mutation flags die with their members).
        self._demat_cache.clear()
        self._demat_dirty.clear()

    def _destroy_unwanted_widgets(self, wanted_set):
        """F-A9 (fourth crash cut): destroy widgets registered for ids the
        new list does NOT want.

        Replaces the blanket _destroy_orphan_widgets() sweep ON THE
        FULL-LIST rebuild path only. The old sweep destroyed EVERY
        registered widget on a structural rebuild — at the end-of-load
        transition that was the whole live band (both mega-cards) on
        top of the sizer-entry loop's own destroys: the mass
        deferred-deletion storm the finalization rebuild then
        constructed replacements through, and the one constant across
        all three silent native deaths (19:08 / 19:42 / 20:54 logs —
        see _rebuild_sizer's F-A9 block). Routing handlers and measure
        passes may still build widgets whose ids dropped from the list
        (failed rebuilds, removed items); those are destroyed exactly
        as before so native handles cannot leak. _rebuild_load_window
        keeps the old blanket sweep (window mode registers only the
        window's own widgets, and mode entry must drop the previous
        band).
        """
        for iid, widget in list(self._widgets_by_id.items()):
            if iid in wanted_set:
                continue
            self._hide_and_destroy(widget)
            self._widgets_by_id.pop(iid, None)
        # F-A10: ids the new list drops must not linger in the
        # detached revisit cache either — a later model change must
        # never resurrect a stale-model widget.
        for iid in list(self._demat_cache.keys()):
            if iid not in wanted_set:
                self._demat_cache_evict(iid)

    def _cache_dematerialized(self, iid, widget):
        """F-A10 (revisit cache): hide + register a dematerialized widget
        in the bounded detached cache.

        The widget stays alive and registered in ``_widgets_by_id`` —
        materialize's early return and the rebuild add loops re-Add it
        on a revisit — so sweeping back over visited territory is a
        re-Show, never a shell+populate+HtmlWindow-reflow
        reconstruction. Every silent native crash so far died inside
        exactly that reconstruction burst (the fourth, 2026-10-06
        21:36 log, on a plain mid-session revisit: the top mega-cards
        the user had ALREADY materialized that session rebuilt from
        scratch per revisit, the log tail dying mid-burst ~654 ms in;
        the clean 20:46 session log shows the same cards rebuilt ~29
        times across a scrolling session), and it is the constant
        "entire list chunky-rerender" of back-and-forth scrolling. The
        FIFO bound keeps native-window memory finite: the OLDEST
        cached entry evicts past DEMAT_CACHE_MAX, so a far revisit
        rebuilds once. Genuine reloads and width changes flush the
        cache wholesale (_flush_demat_cache call sites).
        """
        try:
            if widget.IsBeingDeleted():
                self._widgets_by_id.pop(iid, None)
                return
            widget.Hide()
        except RuntimeError:
            self._widgets_by_id.pop(iid, None)
            self._hide_and_destroy(widget)
            return
        self._demat_cache[iid] = True
        while len(self._demat_cache) > self.DEMAT_CACHE_MAX:
            old_iid = next(iter(self._demat_cache))
            self._demat_cache_drop(old_iid)
            old_widget = self._widgets_by_id.pop(old_iid, None)
            if old_widget is not None:
                self._hide_and_destroy(old_widget)

    def _demat_cache_evict(self, iid):
        """F-A10: drop one id from the detached cache and destroy its widget."""
        if iid not in self._demat_cache:
            return
        self._demat_cache_drop(iid)
        widget = self._widgets_by_id.pop(iid, None)
        if widget is not None:
            self._hide_and_destroy(widget)

    def _flush_demat_cache(self):
        """F-A10: destroy every detached-cache widget and clear the cache.

        Called on width changes (a cached widget re-Added after the
        change would carry the old width's min size and recorded
        height) and from every wholesale-destroy sweep's bookkeeping
        (windowed-reload mode entry clears the cache via Clear).
        """
        for iid in list(self._demat_cache):
            self._demat_cache_drop(iid)
            widget = self._widgets_by_id.pop(iid, None)
            if widget is not None:
                self._hide_and_destroy(widget)

    def _demat_cache_drop(self, iid):
        """F-A10 bookkeeping: drop detached-cache membership for iid.

        Called at EVERY membership drop — a widget going live again, an
        eviction/flush, model removal, or dead-bookkeeping hygiene — so
        the pending-mutation flag can never outlive its membership: a
        re-Added live widget converges through the ordinary per-flush
        passes, and a destroyed one owns nothing. The registry entry is
        NOT touched here; the caller owns it (re-registration, eviction
        destroy, or the wholesale sweeps)."""
        self._demat_cache.pop(iid, None)
        self._demat_dirty.discard(iid)

    def _cleanse_dead_member(self, iid):
        """F-A10 hygiene: drop a detached-cache member whose registered
        widget died out of band (a native destroy outside this view's
        own bookkeeping).

        Without this the reconcile skip lane can hold a dead membership
        forever — it never re-runs the dematerialize branch for members
        — and a dead entry would squat a FIFO slot (and, once a rebuild
        registered a fresh widget for the same id, silently skip that
        LIVE row in the per-flush measure/width passes). Nothing alive
        needs destroying."""
        widget = self._widgets_by_id.get(iid)
        dead = widget is None
        if not dead:
            try:
                dead = bool(widget.IsBeingDeleted())
            except (RuntimeError, WX_FAILURES):
                dead = True
        if dead:
            self._widgets_by_id.pop(iid, None)
            self._demat_cache_drop(iid)

    def mark_out_of_band_mutation(self, item_id):
        """F-A10: flag a registered id whose widget a caller outside the
        view's own materialize path just MUTATED — routing responses
        and tool / permission / report sections, pending mutes, the
        flush's expansion policy, the request-start HWND-freeing
        collapse pass, and an eager registry-return build (materialize
        itself). The controller routes all its own mutation sites
        through _mark_mutated_member, always keyed by a real change
        (a changed-signal), so a steady-state pass never flags and
        the idle-cache skip stays honest.

        A detached-cache member is a hidden widget skipped by the
        per-flush measure/width passes, so the mutation converges on
        the next refresh_materialized_heights — one hidden measure,
        then the flag self-clears — instead of drifting the recorded
        height (and with it the item's offscreen spacer) until some
        later revisit re-Add. No-op for anything not a member: live
        rows are measured every flush anyway."""
        if item_id in self._demat_cache:
            self._demat_dirty.add(item_id)

    def _safe_widget_add(self, widget, item, i=None):
        """F-B3 (the 2026-10-07 11:59 native death): attach a widget row
        without ever leaving the sizer half-mutated by a wx assertion.

        The crash class: a Detach/Insert pair aborted between its halves —
        Detach succeeded and the Insert raised "Adding a window already in
        a sizer" (a bookkeeping disagreement under churn — a duplicate-id
        re-Add, or an out-of-band attach) — so the widget stayed attached
        at its old slot and the half-mutated sizer went native on the next
        layout/paint (the fifth silent death of the F-A7/F-A9/F-A10
        family; 11:59:00,272 log). Two guards here:
        1. ``GetContainingSizer()`` pre-check — an already-attached or
           unusable (destroyed) wrapper is refused BEFORE the sizer is
           touched, so that assertion cannot fire again.
        2. Every caller degrades a refused/failed attach to its spacer
           lane (the id keeps its order slot), so an add loop can never
           abort mid-pass with a count desync.
        Returns the sizer item on success, None on refusal/failure.
        """
        iid = item.item_id
        try:
            containing = widget.GetContainingSizer()
            being_deleted = bool(widget.IsBeingDeleted())
        except (RuntimeError, WX_FAILURES):
            self._logger.error(
                "F-B3: attach refused for %r — widget wrapper unusable "
                "(destroyed out of band)", iid)
            return None
        if being_deleted:
            self._logger.error(
                "F-B3: attach refused for %r — widget pending deletion",
                iid)
            return None
        if containing is not None:
            self._logger.error(
                "F-B3: attach refused for %r — window already in %r "
                "(duplicate re-Add or out-of-band attach); spacer "
                "fallback", iid, containing)
            return None
        try:
            if i is None:
                result = self._history_sizer.Add(
                    widget, 0, self._sizer_flags(item),
                    self._sizer_border(item))
            else:
                result = self._history_sizer.Insert(
                    i, widget, 0, self._sizer_flags(item),
                    self._sizer_border(item))
        except WX_FAILURES:
            self._logger.exception(
                "F-B3: sizer attach failed for %r; spacer fallback", iid)
            return None
        try:
            widget.Show()
        except RuntimeError:
            pass
        return result

    def _replace_sizer_at(self, i, item, widget):
        iid = item.item_id
        try:
            sizer_item = self._history_sizer.GetItem(i)
        except WX_FAILURES:
            sizer_item = None
        current_window = sizer_item.GetWindow() if sizer_item is not None else None
        if not self._detach_index(i, item, cache=(widget is None)):
            # Detach failed: the slot stayed attached, and inserting here
            # would duplicate the entry (count desync). Leave the slot as
            # it is — apply_item_list's integrity gate repairs the
            # bijection with a structural rebuild on the next pass.
            return
        if widget is None:
            # F-A10 dead-cache defect fix — discriminate the registry
            # entry by detached-cache MEMBERSHIP, never by slot-window
            # identity. Reaching here, _detach_index(cache=True) has
            # already run, so for a MEMBER id the registry entry IS
            # this slot's just-cached window (the common reconcile
            # dematerialize) or, on a repeat dematerialize, the spacer
            # slot's cached widget. That registry entry cannot be
            # popped: F-A10 revisits reach a cached widget ONLY through
            # _widgets_by_id (no reuse path consults _demat_cache), so
            # the entry IS the cache's revisit membership. The old
            # unconditional pop exiled every cached widget the moment
            # it was cached — alive+hidden but unreachable, so EVERY
            # revisit took the full shell+populate+HtmlWindow-reflow
            # reconstruction (the chunky mid-list rerender; the
            # identity guard on the destroy changed none of it), the
            # FIFO evict recycled dead bookkeeping while the real
            # hidden native trees leaked, and no cache bound ever
            # mattered. An identity guard ALONE cannot fix this: a
            # repeat pass's spacer slot has no window, so "registered
            # is not current_window" holds vacuously there and the
            # guard pops-and-destroys the cached widget one flush
            # after caching (the one-flush cache demolition the
            # previous session's lost fix attempt shipped). Only
            # membership distinguishes the cases — and a member can
            # never be an orphan: member ids short-circuit this very
            # early return in materialize, so an eagerly-built
            # transient is never a member.
            registered = self._widgets_by_id.get(iid)
            if iid in self._demat_cache:
                if registered is None:
                    # Dead bookkeeping (widget destroyed out of band):
                    # drop the membership so the FIFO slot cannot squat.
                    self._demat_cache_drop(iid)
                # else: the slot's own cached window — keep BOTH the
                # registry entry (the revisit membership) and the cache
                # entry; only the spacer insert runs.
            elif registered is not None and registered is not current_window:
                # Genuine orphan (this branch's original purpose,
                # §8.5): a widget eagerly materialized by routing or a
                # measure pass over a spacer slot, registered but never
                # placed in the sizer. Pop + destroy it so the
                # replacement spacer cannot leak a native handle.
                self._widgets_by_id.pop(iid, None)
                self._hide_and_destroy(registered)
            self._spacer_items[iid] = self._history_sizer.Insert(
                i, (0, self._height_for_item(item)), 0, 0)
        else:
            # F-A10: re-placed live — detached-cache membership drops
            # with the detach state (membership means detached; a
            # pending-mutation flag drops with it too — a live row
            # converges through the ordinary per-flush passes). This
            # widget branch IS the revisit re-Add path: the cached
            # widget's registry hit in _reconcile_states routes here,
            # the spacer slot becomes a live row with NO measure — the
            # recorded exact height (F-A4) already matches the spacer
            # it replaces, and layout_history re-pins its width.
            # F-B3 (the 2026-10-07 11:59 native death): the old slot was
            # already detached above, so refuse/failure here MUST NOT
            # leave the sizer half-mutated — the safe attach's
            # GetContainingSizer pre-check stops a duplicate re-Add
            # before the sizer is touched, and on refusal/failure the
            # slot rolls back to a spacer so the bijection stays
            # consistent for the next pass (the widget stays hidden and
            # registered).
            self._demat_cache_drop(iid)
            if self._safe_widget_add(widget, item, i=i) is None:
                try:
                    widget.Hide()
                except RuntimeError:
                    pass
                self._spacer_items[iid] = self._history_sizer.Insert(
                    i, (0, self._height_for_item(item)), 0, 0)
                return
            self._widgets_by_id[iid] = widget

    def _detach_index(self, i, item, cache=False) -> bool:
        """Detach sizer slot ``i`` and clean up its registries.

        ``cache=True`` (F-A10) dematerializes this slot's item widget
        into the bounded revisit cache instead of destroying it (see
        _cache_dematerialized); every other caller keeps the
        remove/destroy behavior.

        Returns False when the slot could not be detached, in which case
        every registry is left untouched (guard layer: the old swallow
        here popped bookkeeping while the native entry stayed attached —
        a desync by construction).
        """
        iid = item.item_id
        try:
            sizer_item = self._history_sizer.GetItem(i)
        except WX_FAILURES:
            sizer_item = None
        window = sizer_item.GetWindow() if sizer_item is not None else None
        detached = False
        try:
            detached = bool(self._history_sizer.Detach(i))
        except WX_FAILURES:
            detached = False
        if not detached:
            self._logger.error(
                "Detach(%d) failed for %r; sizer slot kept", i, iid)
            self._dump_sizer_state(
                "slot kept after Detach(%d) failed for %r" % (i, iid))
            return False
        self._spacer_items.pop(iid, None)
        if window is not None:
            if cache and self._widgets_by_id.get(iid) is window:
                # F-A10 (revisit cache): dematerialize-to-CACHE, not
                # destroy. The window is hidden and stays registered —
                # a revisit re-Adds it (materialize's early return /
                # the rebuild add loops) instead of shell+populate+
                # measure reconstruction. That reconstruction burst is
                # the native crash medium (four silent deaths, the
                # fourth mid-session on a plain scroll REVISIT of the
                # top mega-cards the user had already materialized
                # several times that session — 2026-10-06 21:36 log)
                # and the constant chunky rerender of back-and-forth
                # middle scrolling; it now runs once per item per
                # session, not once per revisit. The identity guard
                # keeps this slot's window on the destroy path for
                # every foreign-window case; the spacer->widget
                # replace direction passes cache=False.
                self._cache_dematerialized(iid, window)
            else:
                for wid, w in list(self._widgets_by_id.items()):
                    if w is window:
                        self._widgets_by_id.pop(wid, None)
                        break
                self._hide_and_destroy(window)
        return True

    def _remove_sizer_entry(self, iid, preserve=None):
        """Detach one sizer entry by its order id (Approach 1: _sizer_order
        mirrors EVERY sizer entry — widget rows, offscreen spacers, and the
        aggregate load-window sentinels — so the index below is a true
        sizer index, and aggregate entries are removed through this same
        path; nothing is detached by SizerItem-wrapper identity).

        Guard layer: a failed Detach keeps BOTH the sizer slot and the
        order entry (and logs) — shrinking _sizer_order while the entry
        stayed attached was the exact desync that shifted every later
        index and crashed the width pass.

        F-A9 (fourth crash cut): ``preserve`` is the set of item ids a
        structural rebuild still wants — their windows are DETACHED and
        hidden but stay alive and registered, so the rebuild's add loop
        re-Adds them instead of destroying and live-rebuilding several
        hundred native windows per mega-card in one synchronous pass
        (the end-of-load storm all three silent crashes died in; see
        _rebuild_sizer's F-A9 block). Every other caller passes None
        and keeps the destroy-on-detach behavior.
        """
        try:
            idx = self._sizer_order.index(iid)
        except ValueError:
            return
        try:
            sizer_item = self._history_sizer.GetItem(idx)
        except WX_FAILURES:
            sizer_item = None
        window = sizer_item.GetWindow() if sizer_item is not None else None
        detached = False
        try:
            detached = bool(self._history_sizer.Detach(idx))
        except WX_FAILURES:
            detached = False
        if not detached:
            count = self._sizer_count()
            if count is None or idx >= count:
                # F-B2 (desync-storm convergence, 2026-10-07 11:59 log):
                # the native slot does not exist — a pure order phantom
                # (an aborted rebuild's leave-behind or a straggler-Clear
                # orphaned bookkeeping entry; the storm's Detach(0)/
                # Detach(1) failures fire only once the sizer is already
                # empty). Keep-both here latches the rebuild do-loop — the
                # phantom re-fails identically every pass — so drop the
                # order entry and its spacer bookkeeping: nothing is
                # attached at idx >= count, so no live index can shift
                # and the bijection heals.
                self._logger.error(
                    "F-B2: Detach(%d) failed for %r; phantom order entry "
                    "dropped (sizer count %s)", idx, iid, count)
                self._spacer_items.pop(iid, None)
                self._sizer_order.remove(iid)
                return
            self._logger.error(
                "Detach(%d) failed for %r; sizer slot and order entry kept",
                idx, iid)
            self._dump_sizer_state(
                "keep-both after Detach(%d) failed for %r" % (idx, iid))
            return
        self._spacer_items.pop(iid, None)
        if window is not None:
            if preserve is not None and iid in preserve:
                # F-A9: detach-only — the window survives (hidden) and
                # stays registered for the rebuild's add loop to re-Add.
                try:
                    window.Hide()
                except RuntimeError:
                    pass
            else:
                for wid, w in list(self._widgets_by_id.items()):
                    if w is window:
                        self._widgets_by_id.pop(wid, None)
                        break
                self._hide_and_destroy(window)
        self._sizer_order.remove(iid)

    def _sizer_count(self):
        """Child count of the history sizer; None when unreadable."""
        try:
            return self._history_sizer.GetItemCount()
        except WX_FAILURES:
            return None

    def _dump_sizer_state(self, reason: str):
        """F-B4 instrumentation (2026-10-07 11:59 desync storm): one
        bounded, guarded ERROR dump of the order/native bijection — the
        state that pins a phantom's exact birth on the guaranteed repro.
        Capped per view lifetime (a storm must not flood the logs with
        dumps either); the dump itself must never throw."""
        try:
            if self._sizer_dump_count >= self._SIZER_DUMP_CAP:
                if not self._sizer_dump_suppressed:
                    self._sizer_dump_suppressed = True
                    self._logger.error(
                        "F-B4: sizer state dumps suppressed (cap %d "
                        "reached)", self._SIZER_DUMP_CAP)
                return
            self._sizer_dump_count += 1
            pieces = [reason]
            count = self._sizer_count()
            pieces.append(
                "native=%s order=%d spacers=%d demat=%d registry=%d"
                % (count, len(self._sizer_order), len(self._spacer_items),
                   len(self._demat_cache), len(self._widgets_by_id)))
            for i, iid in enumerate(self._sizer_order[:self._SIZER_DUMP_CAP]):
                if count is None or i >= count:
                    native = "beyond-native-tail"
                else:
                    native = ("window" if self._sizer_window_at(i) is not None
                              else "spacer")
                pieces.append("  [%d] %r -> %s" % (i, iid, native))
            if count is not None and count > len(self._sizer_order):
                pieces.append(
                    "  %d native entries beyond the order tail"
                    % (count - len(self._sizer_order)))
            seen_ids = set()
            dup_ids = []
            for iid in self._sizer_order:
                if iid in seen_ids:
                    dup_ids.append(iid)
                else:
                    seen_ids.add(iid)
            if dup_ids:
                pieces.append("  DUPLICATE order ids: %r" % (dup_ids,))
            unplaced = [iid for iid in self._widgets_by_id
                        if iid not in self._sizer_order][:20]
            if unplaced:
                pieces.append(
                    "  registered but unplaced: %r" % (unplaced,))
            self._logger.error(
                "F-B4 sizer state dump:\n    " + "\n    ".join(pieces))
        except Exception:
            try:
                self._logger.exception("F-B4 sizer state dump failed")
            except Exception:
                pass

    def _sizer_window_at(self, i):
        try:
            sizer_item = self._history_sizer.GetItem(i)
        except WX_FAILURES:
            return None
        if sizer_item is None:
            return None
        return sizer_item.GetWindow()

    def materialize(self, item, width=None, measure=True, hidden=False):
        """§8.5: build a widget at the fixed §8.2 width and measure once.

        ``measure=False`` skips the synchronous height measurement (2x Layout
        + HtmlWindow internal-representation sync + GetBestSize). The
        windowed reload paths use it: measuring every window widget as it
        streams blocks the GUI thread and starves paint events (the "cards
        flashing / first card never appears" symptom). Exact heights
        converge at the §7.4 step-5 measure-all and the final flush's
        refresh_materialized_heights, before the end-of-load scroll choice.

        ``hidden=True`` builds the shell inside the never-shown measurement
        host (``_measure_host``) instead of into this live view, for
        transient measurement builds (§7.4 step-5 measure slices).
        wxWidgets child windows are created shown at native creation, and
        the shell constructors (RequestCard / MessagePanel) synchronously
        build further native children (border panels, role labels,
        collapsible headers) and assign per-role background colors before
        any Hide() could take effect — so a shell built into the live view
        exposed a small default-size square of card colors at the view's
        top-left even with an immediate post-build Hide() (the residual
        "small square flashing in the first card's corner" symptom). A
        child of a hidden parent is invisible at native creation, so a
        host-parented build never paints at all. Measurement is
        unaffected: the §8.2 width is assigned explicitly here, and
        SetMinSize/SetSize/Layout/GetBestSize are all parent-independent —
        hidden widgets still reflow HtmlWindows under SetSize/Layout.
        Every sizer path Show()s the widget after Add, so a hidden build
        can never leak into the display hidden. F-A7: LIVE builds birth
        directly into this view at the far off-client SHELL_BIRTH_POS —
        the constructors' native-creation erase is position-clipped out
        of the visible area (the thin flashing band of the mid-drag grey
        defect is dead) with NO Reparent() hierarchy change: the
        host-birth + Reparent variant hard-crashed natively on the first
        post-load band rebuild (see the F-A7 crash-fix comment below).
        """
        if getattr(item, 'suppressed', False):
            # View > API Request Card toggle (off): a suppressed card is never
            # materialized — routing mutations land on the model and replay
            # when the toggle shows the card again.
            return None
        iid = item.item_id
        widget = self._widgets_by_id.get(iid)
        if widget is not None:
            # F-A10 (dead-cache defect fix): a registered widget is
            # either a live sizer row or, for a detached-cache member,
            # a hidden cached card an eager out-of-band caller (the
            # controller routing _materialize, ensure_widget) is about
            # to MUTATE. (_reconcile_states never reaches this early
            # return for a member — it reads the registry first and
            # re-Adds through _replace_sizer_at's widget branch, which
            # measures nothing: the recorded exact height (F-A4) is
            # authoritative.) Membership MEANS detached — the widget
            # stays out of the sizer here — so membership is NOT
            # dropped; the id is FLAGGED instead: cached widgets are
            # skipped by the per-flush measure/width passes, and
            # without the flag a mutated hidden card would land
            # un-measured until some later revisit re-Added it
            # (recorded-height drift under its offscreen spacer).
            # refresh_materialized_heights converges a flagged member
            # exactly once, then it idles again. For a live row the
            # drop is pure hygiene — a live registered widget is never
            # a member under the invariant, and a stale entry here, if
            # one ever appeared, would silently skip this live row in
            # every per-flush re-measure and layout width pass.
            if iid in self._demat_cache:
                self._demat_dirty.add(iid)
            else:
                self._demat_cache_drop(iid)
            return widget
        if width is None:
            width = self._item_width(item)
        # F-A7 crash fix (live-build flash kill, second cut): live shells
        # birth DIRECTLY into this view — the pre-F-A7 native topology that
        # never crashed — at the far off-client SHELL_BIRTH_POS. The first
        # cut built live shells in the host and Reparent()ed them into the
        # view, and that hierarchy change hard-crashed natively (no Python
        # exception, nothing logged) the first time the path ran for real:
        # the first post-load band rebuild, the windowed->full structural
        # rebuild that destroys the load window's widgets and live-builds a
        # whole visible set in one synchronous pass — the log tail dies
        # mid card-construction burst, and the crash lands only on that
        # FIRST scroll after a load (user repro: never crashed before
        # F-A7, later scrolls are fine). The flash kill survives the
        # revert: the constructors' native-creation erase (see the
        # hidden= paragraph) paints at the window's BIRTH position, and a
        # shell born far outside the parent's visible client area paints
        # nothing there — every constructor child is born inside that
        # off-client shell, the sizer's Layout then moves the row on
        # screen before the hidden widget is ever Shown, so the first
        # live pixel the shell paints is at its final sizer position.
        # Third-cut correction (2026-10-06 19:42 log): the same
        # end-of-load burst later crashed WITHOUT any Reparent — the
        # user dragged slowly through the whole load window, the
        # measure-all finished under the held thumb, and the burst died
        # ~115 ms in, native and silent, at the third card's sections.
        # The Reparent was therefore ONE hazard at this storm site, not
        # the mechanism; the surviving unique factor in that repro was
        # _finish_measure_all's scripted scroll firing beneath the held
        # drag (now gated in scroll_to, and at its call site). The
        # birth-position topology itself is exonerated by both logs.
        # Fourth-cut correction (2026-10-06 20:54 log): the SAME burst
        # died a third time — every scripted scroll drag-guarded, no
        # Reparent anywhere — so the surviving hazard is the rebuild's
        # own mass destroy+rebuild churn, removed by the F-A9
        # preserve-and-reuse rebuild (_rebuild_sizer).
        if hidden:
            # Transient measurement build: born inside the never-shown
            # measurement host, so the shell AND every native child its
            # constructors build are invisible at native creation (see
            # the hidden= paragraph in the docstring). Never fall back to
            # the live view when the host is unavailable — a born-visible
            # build is exactly the defect the host exists to prevent; the
            # callers of a measurement build tolerate a skipped item
            # (None).
            parent = self._measure_host
            if parent is None:
                return None
            pos = wx.DefaultPosition
        else:
            # Live sizer-row build: born straight into the view at the
            # off-client birth position — no host, no Reparent.
            parent = self
            pos = self.SHELL_BIRTH_POS
        try:
            widget = self._build_widget_shell(item, parent, pos)
        except (RuntimeError, WX_FAILURES):
            return None
        if widget is None:
            return None
        # Belt-and-suspenders — hidden= transients keep their shown state
        # pinned through the create -> _hide_and_destroy() cycle, and a
        # live shell stays hidden until its sizer path Adds and Shows it,
        # so nothing can ever paint at the pre-layout position.
        try:
            widget.Hide()
        except RuntimeError:
            pass
        # §8.2: fixed width assigned before content SetPage so the
        # synchronous height read is deterministic (§8.5).
        try:
            widget.SetMinSize((width, -1))
            widget.SetMaxSize((width, -1))
            # Give the widget its final width immediately so Layout() arranges
            # children (and HtmlWindows reflow) at the §8.2 width.
            widget.SetSize((width, -1))
        except RuntimeError:
            return None
        # Guard layer: content population is record-dependent (streamed
        # history records); a malformed record must not abort a flush
        # mid-transition with a half-applied sizer, nor leak an
        # unregistered shown native shell into the live view.
        try:
            self._populate_widget_content(item, widget)
        except Exception:
            self._logger.exception(
                "content population failed for %r; destroying shell", iid)
            self._hide_and_destroy(widget)
            return None
        # Apply the expansion policy against the fully populated sections
        # before measuring (§8.3): replay materializes cards with every
        # routed section present, so the measured height must reflect the
        # policy's expanded/collapsed state (defaults + user overrides +
        # force_collapsed). The controller's live _materialize applies the
        # same policy (harmlessly idempotent on the double application).
        if (self._controller is not None
                and item.item_id.kind == 'card'
                and hasattr(widget, 'iter_sections')):
            try:
                self._controller.apply_expansion_policy(item, widget)
            except Exception:
                pass
        self._widgets_by_id[iid] = widget
        self._items_by_id[iid] = item
        # F-A7 invariant fix (drag-end misalignment): a live build ALWAYS
        # measures synchronously — even during a held thumb drag. The
        # first cut skipped the measure mid-drag to cheapen settle turns,
        # but an unmeasured row then stood in the sizer at its NATURAL
        # height while the estimate model, the offscreen spacers, and the
        # F-A2-frozen scrollbar range all carried the CACHED exact
        # heights — every in-budget or parked band materialized during
        # the gesture drifted the content off the frozen range (user
        # repro: first card cut off at the top, viewport well past the
        # last cards with grey below at the bottom, both correct only
        # after the release flush re-measured everything and popped). The
        # fling gate (_thumb_drag_gated) already keeps build storms off
        # the hot path, so the measure returns only on in-budget turns
        # (at most DRAG_BUILD_BUDGET cards) and the one-shot parked /
        # released settles — the pre-F-A7 costs that rendered the ends
        # correctly.
        # Synchronous measure once at materialization (§8.5) — skipped by
        # the windowed reload paths (see the measure= doc above).
        if measure:
            self._measure_widget(item, widget)
        return widget

    def dematerialize(self, item):
        """Dematerialize a widget back to a spacer at its sizer slot."""
        iid = item.item_id
        widget = self._widgets_by_id.get(iid)
        if widget is None:
            return
        try:
            idx = self._sizer_order.index(iid)
        except ValueError:
            idx = -1
        if idx >= 0:
            self._replace_sizer_at(idx, item, None)
        else:
            # Orphan widget (built for measurement / routing, not in the
            # sizer). F-A10: drop any detached-cache membership too — a
            # cached id can reach this branch (dematerialize on an
            # already-detached widget), and membership must never
            # outlive the widget (a stale entry would squat a cache
            # slot until evicted).
            self._demat_cache_drop(iid)
            self._widgets_by_id.pop(iid, None)
            self._hide_and_destroy(widget)

    def remove_widget(self, item_id):
        """Remove a widget OR a spacer for the id (§5.9)."""
        self._items_by_id.pop(item_id, None)
        if item_id in self._sizer_order:
            self._remove_sizer_entry(item_id)
            # F-A10 leak (removed ids): when the dropped entry was a
            # live row, the detach's destroy handled the widget; when it
            # was a SPACER slot, the id's widget can be parked in the
            # detached cache — and the plain early return below never
            # reached the cache cleanup, so a cached widget stayed
            # registered + cached for an id the model no longer holds,
            # invisible to the reconcile loop (it iterates the new item
            # list only) until some structural rebuild's unwanted sweep
            # happened to run. The id is gone from the model: evict the
            # member (which pops the registry entry and destroys the
            # hidden widget too).
            if item_id in self._demat_cache:
                self._demat_cache_evict(item_id)
            return
        widget = self._widgets_by_id.pop(item_id, None)
        self._spacer_items.pop(item_id, None)
        # F-A10: an id leaving the model must also leave the detached
        # revisit cache (it can only be cached here while NOT in the
        # sizer, so this path is the cache case's remove path too). The
        # drop clears any pending-mutation flag as well.
        self._demat_cache_drop(item_id)
        if widget is not None:
            self._hide_and_destroy(widget)

    def _measure_widget(self, item, widget):
        """Layout, sync HtmlWindow heights, then read the widget's height."""
        try:
            widget.Layout()
            self._sync_html_heights(widget)
            widget.Layout()
            height = widget.GetBestSize().GetHeight()
        except RuntimeError:
            return
        if self._controller is not None:
            try:
                self._controller.record_height(item, height)
            except Exception:
                pass

    def _sync_html_heights(self, widget):
        """Re-measure every HtmlWindow at its laid-out width (§8.5).

        ``wx.html.HtmlWindow`` reflows its internal representation only after
        it has been assigned its final width by a sizer layout. Reading the
        internal representation at build time (when the control still has its
        default narrow width) produced an over-tall min height and the large
        vertical gaps; reading it here, after ``widget.Layout()``, gives the
        true wrapped height at the real §8.2 width.
        """
        try:
            children = widget.GetChildren()
        except RuntimeError:
            return
        for child in children:
            if isinstance(child, wx.html.HtmlWindow):
                try:
                    ir = child.GetInternalRepresentation()
                    if ir:
                        h = ir.GetHeight() + 10
                        child.SetMinSize((-1, max(h, 30)))
                except RuntimeError:
                    pass
            else:
                self._sync_html_heights(child)

    def _height_for_item(self, item) -> int:
        if item is not None and getattr(item, 'suppressed', False):
            # View > API Request Card toggle (off): a suppressed card's row
            # collapses to zero height (no controller needed for the gate).
            return 0
        if self._controller is not None:
            try:
                return self._controller.height_for_item(item)
            except Exception:
                pass
        return 120

    def _card_indent(self, item) -> Optional[int]:
        """§8.2 / §2.1: the left indent for a card row, None for a panel.

        Every API Request card is indented: the conditional-agent and
        system-agent cards (the amber ``-> name`` cards, and
        scrutinator/summarizer-type ⚙ cards) take the deeper 40px indent,
        the primary cards the shallower 20px one.  MessagePanels and error
        panels are not cards and carry no indent — their 2px ``wx.ALL``
        margin is the row's whole horizontal border.
        """
        if item is None or item.item_id.kind != 'card':
            return None
        if item.is_conditional or item.is_system_agent:
            return self.AGENT_INDENT_OFFSET
        return self.PRIMARY_INDENT_OFFSET

    def _widget_width(self, iid, item, card_w: int) -> int:
        """Widget width for one row at ``card_w`` (§8.2 width policy).

        Every row is ``card_w`` wide: a card widget is sized ``card_w -
        indent`` — its left indent (20px primary, 40px conditional/system)
        carries the rest of the row — and MessagePanels are sized
        ``card_w - 4`` (row = widget + 2px ``wx.ALL``).  ``item`` may be
        None from registry lookups; the kind then decides, and an unknown
        card takes the shallower primary indent.
        """
        indent = self._card_indent(item)
        if indent is not None:
            return card_w - indent
        if iid.kind == 'card':
            return card_w - self.PRIMARY_INDENT_OFFSET
        return card_w - 4

    def _item_width(self, item) -> int:
        try:
            client_w = self.GetClientSize().GetWidth()
        except RuntimeError:
            client_w = 0
        card_w = max(client_w, self.MIN_CARD_WIDTH)
        return self._widget_width(item.item_id, item, card_w)

    def _sizer_flags(self, item) -> int:
        # §8.2: for a card the left indent IS the sizer border (row =
        # widget + indent, so every row stays card_w wide). A sizer item
        # carries one border value for every side its flags name, so the
        # indent cannot combine with the 2px TOP/BOTTOM margins the card
        # rows used to carry; card rows take the indent alone and get
        # their vertical separation from their neighbors' borders and the
        # card's own internal padding. Panels keep the 2px all-round
        # margin.
        if item is not None and item.item_id.kind == 'card':
            return wx.EXPAND | wx.LEFT
        return wx.EXPAND | wx.ALL

    def _sizer_border(self, item) -> int:
        """Border for the item's sizer entry: the row's left indent for a
        card (§8.2 row = widget + indent), the 2px margin elsewhere."""
        indent = self._card_indent(item)
        return indent if indent is not None else 2

    def refresh_spacer_heights(self):
        """Re-apply every offscreen spacer's height (§8.5)."""
        for iid in list(self._sizer_order):
            if iid in self._widgets_by_id and iid not in self._demat_cache:
                # A live row's slot holds the widget itself — nothing to
                # refresh. A detached-cache MEMBER's slot holds the
                # spacer that must track the item's recorded height:
                # F-A10 members stay registered (the registry entry is
                # the revisit membership), so the registry-presence
                # check alone — written under the pre-cache "registered
                # => live row" invariant — would skip their spacers and
                # let a converged out-of-band mutation (the dirty
                # member's re-measure) drift the sizer off the estimate
                # model until a revisit re-Added the card (F-A6 drift).
                continue
            sizer_item = self._spacer_items.get(iid)
            if sizer_item is None:
                continue
            item = self._items_by_id.get(iid)
            if item is None:
                continue
            try:
                sizer_item.AssignSpacer((0, self._height_for_item(item)))
            except RuntimeError:
                pass

    def refresh_materialized_heights(self):
        """Synchronously re-measure every materialized widget (§8.5).

        F-A7 invariant fix (drag-end misalignment): this runs during a
        held thumb drag again. A mid-gesture flush must leave the sizer
        rows, the estimate model, and the F-A2-frozen scrollbar range in
        agreement (F-A4) — the first cut skipped it to spare per-widget
        measure churn beneath a gesture, but any change landing
        un-measured drifted the content off the frozen range until the
        release settle converged it (the mid-drag wrong-ends / pops-on-
        release repro). With every drag-time build measuring at
        materialize again, this pass is idempotent on an exact band and
        converges only what streaming actually touched.
        """
        for iid, widget in list(self._widgets_by_id.items()):
            if iid in self._demat_cache:
                # F-A10: detached-cache widgets are hidden and idle —
                # their recorded exact heights stay authoritative
                # (F-A4); re-measuring them per flush is pure churn.
                # The ONE exception is a flagged member: something
                # mutated the hidden card out of band (a routing eager
                # build through materialize's registry early return, or
                # the flush's expansion policy), and the recorded height
                # must converge NOW — the item's offscreen spacer
                # holds the stale pre-mutation height until a revisit
                # re-Adds it otherwise (one-flush geometry drift under
                # the scrollbar). The flag self-clears, so a member
                # costs at most one hidden Layout+GetBestSize per
                # mutation, never per flush.
                if iid not in self._demat_dirty:
                    continue
                self._demat_dirty.discard(iid)
            item = self._items_by_id.get(iid)
            if item is None:
                continue
            try:
                widget.SetMinSize((self._item_width(item), -1))
                widget.SetMaxSize((self._item_width(item), -1))
            except RuntimeError:
                pass
            self._measure_widget(item, widget)

    def remeasure_materialized(self, new_width: int):
        """§8.5: re-measure the materialized range at a new effective width."""
        # F-A10: cached widgets carry the OLD width's min size and
        # recorded height — one re-Added after the change would sit
        # wrong in the new geometry, so the cache is flushed wholesale
        # here and far revisits rebuild at the new width.
        self._flush_demat_cache()
        for iid, widget in list(self._widgets_by_id.items()):
            item = self._items_by_id.get(iid)
            if item is None:
                continue
            w = self._widget_width(iid, item, new_width)
            try:
                widget.SetMinSize((w, -1))
                widget.SetMaxSize((w, -1))
            except RuntimeError:
                continue
            self._measure_widget(item, widget)

    def measure_slice(self, items):
        """§7.4 step-5 / §8.5: materialize, measure, immediately dematerialize.

        Transient builds are never born into the live view:
        materialize(hidden=True) parents the shell to the never-shown
        measurement host (_measure_host). wxWidgets child windows are
        created shown at native creation, and the shell constructors
        synchronously build further visible native children with per-role
        background colors — churn that even an immediate post-build Hide()
        cannot suppress: a small default-size square of card colors kept
        flashing at the view's top-left for every measured item (the
        residual symptom; the full-width variant — whole cards flashing
        until the deferred idle-time Destroy() — was the first, fixed by
        the hidden build itself). A child of a hidden parent is invisible
        at native creation, so nothing in the create/measure/destroy cycle
        can ever paint. Measurement is unaffected: SetSize/Layout still
        drive the HtmlWindow reflow and GetBestSize is sizer arithmetic.
        """
        for item in items:
            iid = item.item_id
            if iid not in self._widgets_by_id:
                self.materialize(item, hidden=True)
                self.dematerialize(item)
            else:
                widget = self._widgets_by_id.get(iid)
                if widget is not None:
                    self._measure_widget(item, widget)
                    if iid in self._demat_cache:
                        # F-A10: this slice just measured a registered
                        # widget — a detached-cache member included — so
                        # any pending out-of-band-mutation flag on it is
                        # converged and must not re-fire on a later
                        # flush's refresh pass.
                        self._demat_dirty.discard(iid)

    def layout_history(self):
        """One layout + one virtual-size update for the whole history (§8.2).

        F-A2: while a thumb drag is active the scrollbar range is FROZEN —
        the OS maps mouse movement to the current range for the whole
        gesture, so a range change underneath is a thumb pop. Stage the
        relayout (widget sizing + sizer Layout) and defer the
        virtual-size update to the gesture-end settle
        (controller.settle_thumb_drag).
        """
        try:
            if not self.is_valid():
                return
        except RuntimeError:
            return
        client_w = self.GetClientSize().GetWidth()
        self._width = client_w
        card_w = max(client_w, self.MIN_CARD_WIDTH)
        for iid, widget in list(self._widgets_by_id.items()):
            if iid in self._demat_cache:
                # F-A10: detached-cache widgets are hidden and out of
                # the sizer; their width is re-assigned at re-Add.
                continue
            try:
                w = self._widget_width(iid, self._items_by_id.get(iid), card_w)
                widget.SetMinSize((w, -1))
                widget.SetMaxSize((w, -1))
            except RuntimeError:
                pass
        if self._thumb_drag_active:
            # Gesture staging (F-A2): lay the current band out, leave the
            # scrollbar range untouched until the release settle.
            try:
                self.Layout()
            except WX_FAILURES:
                self._logger.exception("history layout failed (drag staging)")
            return
        # Suppress the scroll echo from Layout/SetVirtualSize: the scrollbar
        # range change is programmatic, not a user scroll (§8.6). The
        # counting latch (F-A3) keeps suppression engaged across overlapping
        # programmatic drivers and outlives the synchronous clamp echoes.
        self._begin_programmatic_scroll()
        try:
            self.Layout()
            min_h = self._history_sizer.GetMinSize().GetHeight()
            new_w = max(client_w, card_w)
            # F-A7 hardening: apply the scrollbar range only when it
            # CHANGES. Steady-state turns (scroll settles, idempotent
            # flushes) re-applied an identical virtual size — one more
            # programmatic range update under the echo latch plus a
            # redundant scrollbar recalculation per turn for nothing, and
            # a release settle re-applying the same range must never
            # clamp or nudge the thumb on unchanged geometry.
            old_w, old_h = self._xy_of(self.GetVirtualSize())
            if (old_w, old_h) != (new_w, min_h):
                self.SetVirtualSize((new_w, min_h))
        except WX_FAILURES:
            self._logger.exception("history layout failed")

    def set_virtual_height(self, total_h: int):
        """Override the virtual size (windowed reload geometry, §8.5).

        layout_history() derives the virtual size from the sizer's min
        height; in windowed reload mode the sizer holds only the visible
        window plus aggregate spacers, so the controller applies the
        full estimated backlog height here after the layout.

        F-A2: during an active thumb drag the range is frozen for the
        whole gesture; the next streaming flush (or finalization)
        re-applies it after the release.
        """
        if self._thumb_drag_active:
            return
        try:
            client_w = self.GetClientSize().GetWidth()
            card_w = max(client_w, self.MIN_CARD_WIDTH)
            self._begin_programmatic_scroll()
            self.SetVirtualSize((max(client_w, card_w), max(0, int(total_h))))
        except RuntimeError:
            pass

    
    def force_paint(self):
        """Force a synchronous paint of the current window contents.

        One-shot reload head paint (§5.10.1): the first card is laid out during
        a replay chunk, but the continuous replay/finalize CallAfter chain keeps
        the GUI event queue non-empty, and on Windows WM_PAINT is only dispatched
        when the queue is empty — so without a synchronous paint the first card
        is never visible until finalization. ``Refresh`` + ``Update`` forces the
        paint immediately, independent of the event queue.
        """
        try:
            self.Refresh()
            self.Update()
        except RuntimeError:
            pass
        # The history view sits inside the tab inside the AuiNotebook; on
        # Windows a child-only Update can be clipped if the parent chain has
        # pending paints, so also force the owning tab to paint synchronously.
        if self._tab is not None:
            try:
                self._tab.Refresh()
                self._tab.Update()
            except RuntimeError:
                pass

    def get_client_width(self) -> int:
        try:
            return self.GetClientSize().GetWidth()
        except RuntimeError:
            return 0

    def get_client_height(self) -> int:
        try:
            return self.GetClientSize().GetHeight()
        except RuntimeError:
            return 0

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    def capture_scroll_state(self) -> ScrollState:
        """§8.1: capture pre-change scroll geometry.

        F-B2's one px contract: ``view_start`` is the scroll origin in
        DEVICE PIXELS (CalcUnscrolledPosition(0,0)) — never
        GetViewStart()'s scroll units; scroll_to() consumes px, and a
        units capture restored as though it were px landed at a
        fraction of the position.
        """
        # Crash fix: the two-int CalcUnscrolledPosition overload returns
        # a PLAIN (x, y) tuple on this wx build (the shipped origin.x
        # dereference raised AttributeError inside every _flush — at
        # load_existing/_end_load and every end_batch — and the
        # AttributeError escaped the controller's WX_FAILURES guard,
        # killing the flush and hanging the reload). Normalize through
        # _calc_unscrolled.
        pos = self._calc_unscrolled(0, 0)
        try:
            vs = self.GetVirtualSize()
        except RuntimeError:
            vs = (0, 0)
        try:
            cs = self.GetClientSize()
        except RuntimeError:
            cs = (0, 0)
        try:
            unit = self.GetScrollPixelsPerUnit()
        except RuntimeError:
            unit = (0, 0)
        scroll_y = self.current_scroll_y_px()
        pixels_below = max(0, vs[1] - cs[1] - scroll_y)
        anchor_id = None
        anchor_offset = 0
        if (self._controller is not None
                and not self._controller.load_in_progress):
            # The O(n) anchor scan is unused during a reload: the scroll
            # policy keeps the viewport pinned at the list head until the
            # end-of-load decision, so skip it on every streaming flush.
            try:
                items = self._controller._items
            except Exception:
                items = []
            for item in items:
                y = getattr(item, 'last_y', 0)
                if y > scroll_y:
                    break
                anchor_id = item.item_id
                anchor_offset = scroll_y - y
        return ScrollState(
            mode='restore', view_start=pos,
            client_px=(cs[0], cs[1]), virtual_px=(vs[0], vs[1]),
            scroll_unit=unit, pixels_below=pixels_below,
            anchor_item_id=anchor_id, anchor_offset=anchor_offset)

    def position_of_anchor(self, item_id, offset: int = 0):
        """Anchor -> device px (the scroll_to px contract, F-B2).

        Sign convention (probe-proven fix): capture_scroll_state()
        defines ``anchor_offset`` DOWNWARD into the anchor item —
        ``scroll_y - anchor_top``, the px of the anchor card the
        viewport top cuts through — so the restore target is
        ``anchor_top + offset``. The shipped ``- offset`` landed every
        rule-4 restore 2x offset px too high (probe: anchor top 1559,
        offset 106, captured 1665 — the old arithmetic restored 1455).
        A materialized widget's position is corrected through
        CalcUnscrolledPosition: ScrolledWindow children report
        client-space (scroll-displaced) positions, and subtracting a
        logically-captured offset from that was the −scroll-depth
        error that slammed deep reading positions to the top (defect
        B2). Offscreen anchors use the item's model geometry instead.
        """
        try:
            # F-A10 (revisit cache): a dematerialized (detached-cache)
            # widget is out of the sizer — its client position is its
            # pre-detach coordinate, not the item's live slot, so a
            # cached anchor must resolve through the model geometry
            # below instead of the live-widget position read.
            widget = (self._widgets_by_id.get(item_id)
                      if item_id not in self._demat_cache else None)
            if widget is not None:
                x_w, y_w = self._xy_of(widget.GetPosition())
                y = self._calc_unscrolled(x_w, y_w)[1]
                return (0, max(0, y + offset))
        except RuntimeError:
            pass
        # Offscreen / detached anchor: the item's last_y model geometry
        # (F-B2's restore target space). F-A10: a dematerialized
        # (detached-cache) widget also routes here — its client
        # position is its pre-detach coordinate, not the item's live
        # slot, so the anchor must resolve through the model geometry.
        if self._controller is not None:
            try:
                item = self._controller._items_by_id.get(item_id)
            except Exception:
                item = None
            if item is not None and getattr(item, 'last_y', None) is not None:
                return (0, max(0, item.last_y + offset))
        return None

    def scroll_to(self, pos):
        """Single scripted-scroll executor (§8.6).

        F-B2's one px contract: ``pos`` is the target view origin in
        DEVICE PIXELS, converted here to Scroll()'s scroll-unit
        coordinates (Scroll() takes units — GetViewStart()'s space —
        so a px value fed straight to it overshot by the scroll rate
        and clamped, landing deep targets on the bottom instead of the
        anchor). The scrolled-to px is recorded on the controller so
        reposition()'s echo-dedup matches the returning echo exactly.
        """
        if pos is None:
            return
        if self._thumb_drag_active:
            # F-A7 third cut (drag-held scripted-scroll elimination): no
            # scripted scroll may ever run beneath a held native thumb
            # drag. F-A2 guards the policy lanes (force-bottom, rule-4
            # restore, resize keep-bottom), but one scripted scroll sat
            # outside every lane: _finish_measure_all's position put-back
            # (§7.4 step 5) fires the moment the measure chunks drain,
            # and with the user mid-drag on the scrollbar (reload → slow
            # drag through the load window; user repro, 2026-10-06 19:42
            # log) Scroll()/SetScrollPos fought the OS drag capture — the
            # corrupted native scroll state then tripped the end-of-load
            # windowed→full whole-band rebuild into a hard native death
            # (no Python exception, no log line; the log tail dies ~115
            # ms into the burst, mid CollapsibleSection construction —
            # the SAME fatal site the first F-A7 crash hit with the
            # Reparent cut, now with the birth-position topology). The
            # release settle (settle_thumb_drag) re-runs the §8.1 policy
            # over fresh geometry, so skipping here loses nothing:
            # convergence is the gesture's, not the scripted scroll's.
            return
        try:
            x, y = int(pos[0]), int(pos[1])
            ux, uy = self._xy_of(self.GetScrollPixelsPerUnit())
            ux = ux or 1
            uy = uy or 1
            xu = -1 if x < 0 else int(round(x / ux))
            yu = max(0, int(round(y / uy)))
            self._begin_programmatic_scroll()
            self.Scroll(xu, yu)
            if self._controller is not None:
                self._controller.record_scroll_position(
                    (max(0, xu) * ux, yu * uy))
        except (RuntimeError, TypeError, ValueError):
            pass

    def scroll_to_bottom(self):
        """Scripted jump to the bottom — routed through scroll_to() so the
        px contract, the echo-suppression latch (F-A3) and the echo-dedup
        feed are shared by every scripted scroll (§8.6's single executor
        as the plan intended)."""
        try:
            if not self.is_valid():
                return
            self.Layout()
            vs = self.GetVirtualSize()
            cs = self.GetClientSize()
            self.scroll_to((0, max(0, vs[1] - cs[1])))
        except RuntimeError:
            pass

    def is_near_bottom(self, tolerance: int = 24) -> bool:
        """True if the current scroll position is within tolerance of the bottom."""
        try:
            vs = self.GetVirtualSize()
            cs = self.GetClientSize()
            max_y = max(0, vs[1] - cs[1])
            return (max_y - self.current_scroll_y_px()) <= tolerance
        except RuntimeError:
            return True

    def current_scroll_y_px(self) -> int:
        """Current vertical scroll offset in pixels."""
        return self._calc_unscrolled(0, 0)[1]

    @staticmethod
    def _xy_of(value):
        """Normalize a wx.Point / wx.Size / plain-tuple API result to a
        plain (x, y) int tuple.

        Crash fix: this wx build wraps some window-API results as
        wx.Point / wx.Size objects and others — notably the two-int
        CalcUnscrolledPosition overload — as PLAIN TUPLES, so
        attribute dereferences (.x/.y) on a tuple result raised
        AttributeError; AttributeError is NOT in WX_FAILURES, so it
        escaped every controller guard and killed the flush (the
        reload-open hang). Sequence access is the one spelling every
        form supports; attribute access is fallback only.
        """
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            pass
        try:
            return (int(value.x), int(value.y))
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            return (int(value.width), int(value.height))
        except (AttributeError, TypeError, ValueError):
            pass
        return (0, 0)

    def _calc_unscrolled(self, x, y):
        """CalcUnscrolledPosition(x, y) normalized to a plain px tuple.

        The two-int overload returns a plain (x, y) tuple on this wx
        build (the reload-open AttributeError came from dereferencing
        .x on it); RuntimeError from a half-destroyed window and
        unreadable results map to (0, 0).
        """
        try:
            result = self.CalcUnscrolledPosition(int(x), int(y))
        except (RuntimeError, TypeError, ValueError):
            return (0, 0)
        return self._xy_of(result)

    # ------------------------------------------------------------------
    # F-B2 event-space contract
    # ------------------------------------------------------------------

    def _on_scroll(self, event):
        # F-A1: a THUMBTRACK never recomputes inline — the per-echo full
        # view pass was the drag churn (widget churn per track event,
        # SetVirtualSize under the held thumb, WM_PAINT starvation). Latch
        # the gesture, feed the position to the echo-dedup, and coalesce
        # to one deferred settle per event-loop turn.
        pos_px = self._event_position_px(event)
        if event.GetEventType() == wx.wxEVT_SCROLLWIN_THUMBTRACK:
            if not self._scroll_echo_suppressed():
                # A scripted echo can share the event type; only a genuine
                # user track latches the gesture.
                if not self._thumb_drag_active and \
                        self._controller is not None:
                    # F-A7 gesture start: reset the previous gesture's
                    # fling latch — a release settle can still sit
                    # deferred behind the echo-suppression latch
                    # (_deferred_thumb_settle), so it cannot be relied on
                    # to have run before the next touchdown.
                    try:
                        self._controller.start_thumb_drag()
                    except WX_FAILURES:
                        pass
                # F-A9 (fourth cut, pace-aware unlatch): this genuine
                # track is reading-paced whenever the gap since the
                # previous genuine track exceeds half the park window.
                # The fling latch lasts the whole gesture once set (a
                # single over-budget turn sets it, and DRAG_TURN_BUDGET_MS
                # trips easily on the mega-cards), which sentenced a
                # reader-paced drag to grey until release for the sin of
                # one slow turn mid-gesture. Clearing it on sparse gaps
                # re-evaluates the build budget once per reading-paced
                # turn: in-budget turns build live again, an over-budget
                # turn re-latches until the next sparse gap.
                now = time.perf_counter()
                if (self._last_drag_track_perf is not None
                        and self._controller is not None
                        and (now - self._last_drag_track_perf) * 1000.0
                        >= self.DRAG_PARK_MS // 2):
                    try:
                        self._controller.start_thumb_drag()
                    except WX_FAILURES:
                        pass
                self._last_drag_track_perf = now
                self._thumb_drag_active = True
                if self._controller is not None:
                    self._controller.record_scroll_position(pos_px)
                self._defer_drag_settle()
                # F-A9: (re)arm the park poll on every genuine track —
                # the posted-message carrier (see _drag_park_poll).
                self._defer_drag_park()
            event.Skip()
            return
        if self._thumb_drag_active:
            # The gesture ends on the first non-track scroll event
            # (THUMBRELEASE, or the thumb handing straight to another
            # action): settle synchronously — full recompute with the
            # frozen range re-applied (layout_history now runs its full
            # body), the §8.1 policy over fresh geometry, then a forced
            # synchronous paint (F-A2/A5).
            self._thumb_drag_active = False
            self._drag_settle_pending = False
            self._last_drag_track_perf = None
            self._stop_drag_park_timer()
            if (self._controller is not None
                    and not self._scroll_echo_suppressed()):
                try:
                    self._controller.settle_thumb_drag(pos_px)
                except WX_FAILURES:
                    self._logger.exception("thumb-drag settle failed")
                self.force_paint()
            else:
                # An echo ended the latch: still clear the gesture (the
                # staging must never outlive it) and re-apply the staged
                # range inline so the scrollbar is never left frozen on
                # mid-gesture geometry.
                try:
                    min_h = self._history_sizer.GetMinSize().GetHeight()
                    self._begin_programmatic_scroll()
                    self.SetVirtualSize(
                        (max(self.GetClientSize().GetWidth(),
                             self.MIN_CARD_WIDTH), min_h))
                except WX_FAILURES:
                    self._logger.exception(
                        "post-drag virtual-size reapply failed")
                # F-A7 defect fix: the old code DROPPED the release
                # settle here — the queued CallAfter chain above kept the
                # echo latch non-zero at release time, so the release
                # recompute (band rebuild at the final position, frozen
                # range reapply, height re-measure, §8.1 policy) never
                # ran at all and the viewport rested on whatever band the
                # last settle left — grey and stuck until the next
                # genuine scroll. Defer the settle behind the latch
                # instead: the suppression drains one event-loop turn
                # later and the deferred settle re-checks before running.
                if self._controller is not None:
                    wx.CallAfter(self._deferred_thumb_settle, pos_px)
            event.Skip()
            return
        # Programmatic-scroll echo suppression (§8.6): only genuine user
        # scrolls reach the controller's reposition.
        if (self._controller is not None
                and not self._scroll_echo_suppressed()):
            self._controller.reposition(pos_px)
        event.Skip()

    def _defer_drag_settle(self):
        """F-A1: one coalesced settle per event-loop turn while dragging."""
        if self._drag_settle_pending:
            return
        self._drag_settle_pending = True
        wx.CallAfter(self._drag_settle_step)

    def _drag_settle_step(self):
        self._drag_settle_pending = False
        if self._thumb_drag_active and self._controller is not None:
            # One coalesced pass per turn: the visible band follows the
            # latest tracked position while the range stays frozen
            # (layout_history stages during the drag, F-A2) and the
            # queue drains between deferred passes so WM_PAINT keeps up.
            # F-A7: the controller budget-gates the brand-new-band builds
            # here — an over-budget fling turn virtualizes through the
            # swept area (grey, converging at the first park / the
            # release settle) because the live build churn of whole
            # visible sets per turn is exactly what starves paint
            # mid-list.
            try:
                self._controller.reposition()
            except WX_FAILURES:
                self._logger.exception("drag settle recompute failed")

    def _defer_drag_park(self):
        """F-A9 (fourth cut): (re)arm the park-convergence poll on a
        genuine track — a POSTED-MESSAGE carrier, never wx.Timer.

        The first cut carried the park window on a one-shot wx.Timer,
        and the 20:46/20:54 test sessions proved its WM_TIMER never
        reached the handler while the native MSW scrollbar drag holds
        the loop (user repro: stop the mouse mid-drag -> the whole
        message area stays grey until the mouse moves again or the
        thumb is released; the release settle always converges, and
        any motion does too — exactly when posted events flow again).
        wx.CallAfter posts ARE dispatched mid-drag — the F-A1 settle
        steps that build every drag band run on that very carrier —
        so the poll rides it: armed per genuine track, it dies on the
        first poll after a NEWER track arrived (that track re-arms a
        chain of its own), holds through the park window on a
        stationary thumb, and converges the parked band with one full
        recompute plus a synchronous paint while the gesture is
        still held.
        """
        if self._drag_park_polling:
            return
        self._drag_park_polling = True
        self._drag_park_seen_stamp = self._last_drag_track_perf
        wx.CallAfter(self._drag_park_poll)

    def _stop_drag_park_timer(self):
        """F-A9: the release settle owns the gesture's convergence — no
        park poll may fire after the gesture ended. (Method name kept:
        the release branch and stop_all_pending_timers call this as
        the park cancel; the wx.Timer it used to stop is gone.)"""
        self._drag_park_polling = False

    def _drag_park_poll(self):
        """F-A9: the arm/hold/converge poll behind the park window.

        Each step compares the latest genuine-track stamp against the
        stamp this chain was armed with. A CHANGED stamp means motion
        is flowing — the chain is obsolete (that track arms its own).
        A matching stamp inside the park window holds the chain via
        one CallAfter: while the thumb is stationary nothing else of
        ours is queued, so the poll's own micro-call is the only
        posted work, and the chain self-terminates the instant it
        converges (polling cleared + park_thumb_drag + force_paint) —
        it can never become a paint-starving heartbeat during motion.
        """
        if not self._drag_park_polling:
            return
        try:
            if not self.is_valid():
                self._drag_park_polling = False
                return
        except RuntimeError:
            self._drag_park_polling = False
            return
        if not self._thumb_drag_active or self._controller is None:
            self._drag_park_polling = False
            return
        stamp = self._last_drag_track_perf
        if stamp != self._drag_park_seen_stamp:
            # A newer genuine track arrived since this chain was armed:
            # obsolete — re-arm against the newer stamp and let this
            # chain end (one chain lives at a time).
            self._drag_park_polling = False
            self._defer_drag_park()
            return
        now = time.perf_counter()
        if stamp is None or (now - stamp) < self.DRAG_PARK_MS / 1000.0:
            # Still inside the park window with the thumb stationary —
            # hold the poll alive (each hold is one micro-call; the
            # window lasts DRAG_PARK_MS and ends in convergence).
            wx.CallAfter(self._drag_park_poll)
            return
        if self._scroll_echo_suppressed():
            # A scripted scroll's delayed echo is still in flight: wait
            # one more turn — the latch drains one event-loop turn
            # after its last begin.
            wx.CallAfter(self._drag_park_poll)
            return
        self._drag_park_polling = False
        try:
            self._controller.park_thumb_drag()
        except WX_FAILURES:
            self._logger.exception("thumb-drag park settle failed")
            return
        self.force_paint()

    def _deferred_thumb_settle(self, pos_px):
        """F-A7: run the release settle once the echo-suppression latch
        has drained (scheduled from the echo-suppressed release branch — a
        suppressed echo must never cancel a genuine gesture-end recompute;
        the old swallow there stranded the viewport on stale-band grey
        until the next user scroll)."""
        if self._thumb_drag_active or self._controller is None:
            # A NEW genuine gesture began inside the delay window — its
            # own settle path owns convergence from here.
            return
        if self._scroll_echo_suppressed():
            # Overlapping scripted scrolls still hold references; wait one
            # more turn (each begin releases one CallAfter later).
            wx.CallAfter(self._deferred_thumb_settle, pos_px)
            return
        try:
            self._controller.settle_thumb_drag(pos_px)
        except WX_FAILURES:
            self._logger.exception("deferred thumb-drag settle failed")
            return
        self.force_paint()

    # ------------------------------------------------------------------
    # Programmatic-scroll echo suppression (§8.6) — counting latch (F-A3)
    # ------------------------------------------------------------------

    def _begin_programmatic_scroll(self):
        """Take one suppression reference; released one event-loop turn
        later. Overlapping scripted scrolls (a flush that both scrolls
        and re-lays-out) keep suppression engaged for the whole window,
        so a delayed clamp echo can never masquerade as a user scroll
        mid-gesture — the one-CallAfter boolean was exactly that hole
        in the drag popping loop."""
        self._scroll_echo_depth += 1
        wx.CallAfter(self._clear_programmatic_scroll)

    def _clear_programmatic_scroll(self):
        """Release one suppression reference (scheduled per begin)."""
        self._scroll_echo_depth = max(0, self._scroll_echo_depth - 1)

    def _scroll_echo_suppressed(self) -> bool:
        return self._scroll_echo_depth > 0

    def _event_position_px(self, event):
        """Normalize an EVT_SCROLLWIN position to device px (F-B2).

        Real event contract (crash fix): wx.ScrollWinEvent's
        GetPosition() returns a plain INT — the thumb position in
        SCROLL UNITS, never a Point/tuple — so the tuple-unpack read
        raised TypeError and treated every real event as unreadable
        (None), silently starving the echo-dedup. Read the int, get
        the axis from GetOrientation(), and multiply by the scroll
        rate into the device-px space scroll_to() records. None when
        the position is not readable.
        """
        try:
            pos = event.GetPosition()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return None
        if pos is None:
            return None
        try:
            rate = self.GetScrollPixelsPerUnit()
        except (RuntimeError, AttributeError, TypeError, ValueError):
            rate = (1, 1)
        ux, uy = self._xy_of(rate)
        ux = ux or 1
        uy = uy or 1
        if isinstance(pos, int):
            try:
                orientation = event.GetOrientation()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                orientation = None
            if orientation is not None and orientation == getattr(
                    wx, 'HORIZONTAL', None):
                return (int(pos) * ux, 0)
            return (0, int(pos) * uy)
        # Defensive: a port handing back Point-/tuple-shaped positions.
        try:
            return (int(pos[0]) * ux, int(pos[1]) * uy)
        except (TypeError, ValueError, IndexError, KeyError):
            return None

    def is_thumb_drag_active(self) -> bool:
        """F-A2: True while the scrollbar thumb is held in a drag — the
        scroll policy and range application must stay out of the way."""
        return self._thumb_drag_active

    def _on_size(self, event):
        event.Skip()
        if self._size_timer is None:
            self._size_timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self._on_size_debounce, self._size_timer)
        if self._size_timer.IsRunning():
            self._size_timer.Stop()
        self._size_timer.Start(200, oneShot=True)

    def _on_size_debounce(self, event):
        # Guard layer: a dead native window (teardown racing the timer) must
        # not raise out of the timer handler into the event loop.
        if self._controller is None:
            return
        try:
            width = self.GetClientSize().GetWidth()
        except WX_FAILURES:
            return
        self._controller.on_width_changed(width)

    def stop_all_pending_timers(self):
        # F-A7: stop the drag-park window too — the tab is tearing down
        # and no park settle may fire into a half-dead view.
        self._stop_drag_park_timer()
        for widget in list(self._widgets_by_id.values()):
            try:
                if hasattr(widget, 'stop_pending_timer'):
                    widget.stop_pending_timer()
            except Exception:
                pass


class MessagePanel(wx.Panel):
    """Panel displaying a single message in the conversation."""

    def __init__(self, parent, message: ConversationMessage,
                 on_edit: Optional[Callable[[], None]] = None,
                 defer_content: bool = False, pos=wx.DefaultPosition):
        super().__init__(parent, pos=pos)

        self.message = message
        self.on_edit = on_edit
        self.populated = False

        # Ctrl+A on a focused card expands the selection to ALL of the
        # card's text, whether it is on screen or not. CHAR_HOOK is
        # delivered to the focused window first and propagates up the
        # parent chain, so binding on the panel catches the chord from
        # this panel, its message HtmlWindow, or the Fork button. Only
        # user/assistant panels bind: permission-request panels must
        # keep native Ctrl+A behavior for their explanation TextCtrl.
        if message.role in ('user', 'assistant'):
            self.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)

        # Determine appearance based on role
        self._setup_appearance()

        # Main horizontal sizer for border + content
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Add left border accent for User and Assistant messages
        if self.border_color:
            border_panel = wx.Panel(self, size=(4, -1))
            border_panel.SetBackgroundColour(self.border_color)
            main_sizer.Add(border_panel, 0, wx.EXPAND)

        # Content sizer (vertical)
        self._content_sizer = wx.BoxSizer(wx.VERTICAL)

        # Header with role and timestamp
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # For tool messages, add status icon to the role label
        role_text = self._get_role_label()
        if message.role == 'tool':
            status = message.metadata.get('status', 'pending')
            if status == 'pending':
                role_text = "⏳ " + role_text
            else:
                execution_successful = message.metadata.get('execution_successful')
                status_icon = "✗" if not execution_successful else "✓"
                role_text = f"{status_icon} {role_text}"

        self.role_label = wx.StaticText(self, label=role_text)
        font = self.role_label.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.role_label.SetFont(font)
        self.role_label.SetForegroundColour(self.role_color)
        header_sizer.Add(self.role_label, 0, wx.RIGHT, 5)

        fork_turn_label = wx.StaticText(self, label=f"Fork {message.fork}, Turn {message.turn}")
        fork_turn_label.SetForegroundColour(themes.to_wx(themes.active().muted_fg))
        header_sizer.Add(fork_turn_label, 0)

        header_sizer.AddStretchSpacer(1)

        # Fork button for user messages
        if message.role == 'user' and on_edit:
            fork_btn = wx.Button(self, label="Fork", style=wx.BU_EXACTFIT)
            fork_btn.Bind(wx.EVT_BUTTON, lambda e: on_edit())
            header_sizer.Add(fork_btn, 0)

        self._content_sizer.Add(header_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Content
        self.content_ctrl: Optional[wx.html.HtmlWindow] = None
        self.meta_label: Optional[wx.StaticText] = None

        # Add content sizer to main sizer
        main_sizer.Add(self._content_sizer, 1, wx.EXPAND)

        self.SetSizer(main_sizer)
        self.SetBackgroundColour(self.bg_color)

        if not defer_content:
            self.populate_content()

    def populate_content(self):
        """Build the role-specific content (§8.5).

        Runs after materialize() assigns the fixed §8.2 width, so SetPage
        happens at the final width and the synchronous height read is
        deterministic.
        """
        if self.populated:
            return
        self.populated = True
        message = self.message
        sizer = self._content_sizer
        if message.role == 'permission_request':
            self._add_permission_request_ui(sizer, message)
        elif message.role == 'tool':
            # Tool message - show content with status indicator
            content_text = self._create_html_text(self._format_tool_content(message))
            sizer.Add(content_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
            self.content_ctrl = content_text
        elif message.role == 'assistant':
            # model_message_to_user records carry their rendering format in
            # metadata: 'markdown' renders rich via mistune; anything else
            # (missing key, legacy logs without the header field, None,
            # plain_text, unknown values) keeps the plain path.
            fmt = (message.metadata or {}).get('format')
            if not isinstance(fmt, str):
                fmt = None
            content_text = self._create_html_text(message.content, fmt)
            sizer.Add(content_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
            # Keep the HtmlWindow reference so the card's CHAR_HOOK
            # select-all handler (Ctrl+A) can reach the full message
            # text, visible or not.
            self.content_ctrl = content_text
        else:
            content_text = self._create_html_text(message.content)
            sizer.Add(content_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
            # Same as above: keep the reference for the Ctrl+A
            # select-all handler.
            self.content_ctrl = content_text

        # Metadata (token counts, tool execution status, etc.)
        show_metadata = (
            'prompt_tokens' in message.metadata or
            'cached_tokens' in message.metadata or
            'reasoning_tokens' in message.metadata or
            'execution_successful' in message.metadata or
            'tags' in message.metadata
        )
        if message.metadata and show_metadata:
            meta_str = self._format_metadata(message.metadata)
            if meta_str:
                self.meta_label = wx.StaticText(self, label=meta_str)
                self.meta_label.SetForegroundColour(themes.to_wx(themes.active().muted_fg))
                sizer.Add(self.meta_label, 0, wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.Layout()

    def _setup_appearance(self):
        """Set up colors based on message role."""
        # Complementary border colors for User and Assistant
        self.border_color: Optional[wx.Colour] = None

        theme = themes.active()
        roles = {
            'user': (themes.to_wx(theme.msg_user_bg), themes.to_wx(theme.msg_user_fg)),
            'assistant': (themes.to_wx(theme.msg_assistant_bg), themes.to_wx(theme.msg_assistant_fg)),
            'tool_result': (themes.to_wx(theme.msg_tool_result_bg), themes.to_wx(theme.msg_tool_result_fg)),
            'permission_request': (themes.to_wx(theme.msg_permission_bg), themes.to_wx(theme.msg_permission_fg)),
        }
        # Fallback for an unrecognised role/status (theme's plain body).
        default_colours = (themes.to_wx(theme.msg_default_bg),
                           themes.to_wx(theme.msg_default_fg))

        # Set complementary border colors for User and Assistant
        if self.message.role == 'user':
            self.border_color = themes.to_wx(theme.msg_user_border)
        elif self.message.role == 'assistant':
            self.border_color = themes.to_wx(theme.msg_assistant_border)

        # Tool messages have dynamic colors based on status
        if self.message.role == 'tool':
            status = self.message.metadata.get('status', 'pending')
            if status == 'pending':
                self.bg_color = themes.to_wx(theme.msg_tool_pending_bg)
                self.role_color = themes.to_wx(theme.msg_tool_pending_fg)
            elif status == 'completed':
                # Check execution_successful from metadata (boolean)
                execution_successful = self.message.metadata.get('execution_successful')
                if not execution_successful:
                    self.bg_color = themes.to_wx(theme.msg_tool_fail_bg)
                    self.role_color = themes.to_wx(theme.msg_tool_fail_fg)
                else:
                    self.bg_color = themes.to_wx(theme.msg_tool_ok_bg)
                    self.role_color = themes.to_wx(theme.msg_tool_ok_fg)
            else:
                self.bg_color, self.role_color = roles.get(
                    'tool_result', default_colours)
        else:
            self.bg_color, self.role_color = roles.get(
                self.message.role, default_colours)

    def _format_tool_content(self, message: ConversationMessage) -> str:
        """Format tool message content based on status."""
        status = message.metadata.get('status', 'pending')
        content = message.content
        tool_calls = message.metadata.get('tool_calls')

        parts = []
        if tool_calls:
            for name, args in tool_calls:
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                parts.append(f"🔧 {name}({args_str})")
            parts.append("")  # separator

        if status == 'pending':
            if content:
                parts.append(content)
            return "\n".join(parts)
        else:
            # Completed - show both tool use and result
            result = message.metadata.get('result', '')
            content_rstripped = content.rstrip('\r\n ')
            if content_rstripped:
                parts.append(content_rstripped)
            if result:
                parts.append(result)
            return "\n".join(parts)

    def update_content(self):
        """Update the panel content when message changes (for tool messages)."""
        if self.message.role != 'tool' or not self.content_ctrl:
            return

        # Update header icon and color
        status = self.message.metadata.get('status', 'pending')
        if status == 'pending':
            role_text = "â³ " + self._get_role_label()
        else:
            execution_successful = self.message.metadata.get('execution_successful')
            status_icon = "â" if not execution_successful else "â"
            role_text = f"{status_icon} {self._get_role_label()}"

        self.role_label.SetLabel(role_text)

        # Update appearance based on new status
        self._setup_appearance()
        self.SetBackgroundColour(self.bg_color)
        self.role_label.SetForegroundColour(self.role_color)

        # Update the content text (after _setup_appearance so bg_color is current)
        self._set_html_content(self.content_ctrl, self._format_tool_content(self.message))
        
        # Create or update metadata label if we now have metadata
        show_metadata = (
            'prompt_tokens' in self.message.metadata or
            'cached_tokens' in self.message.metadata or
            'reasoning_tokens' in self.message.metadata or
            'execution_successful' in self.message.metadata or
            'tags' in self.message.metadata
        )
        if show_metadata:
            meta_str = self._format_metadata(self.message.metadata)
            if meta_str:
                if self.meta_label is None:
                    # Create metadata label if it doesn't exist
                    self.meta_label = wx.StaticText(self, label=meta_str)
                    self.meta_label.SetForegroundColour(themes.to_wx(themes.active().muted_fg))
                    # Need to add to sizer - find the main content sizer
                    sizer = self.GetSizer()
                    if sizer and isinstance(sizer, wx.BoxSizer):
                        # Get the content sizer (second item in horizontal main sizer)
                        if sizer.GetItemCount() > 0:
                            content_sizer = sizer.GetItem(0).GetSizer()
                            if content_sizer:
                                content_sizer.Add(self.meta_label, 0, wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
                                content_sizer.Layout()
                else:
                    # Update existing label
                    self.meta_label.SetLabel(meta_str)
        
        self.Refresh()

    def _get_role_label(self) -> str:
        """Get display label for the role."""
        labels = {
            'user': 'You',
            'assistant': 'Assistant',
            'tool': 'Tool',
            'tool_result': 'Tool Result',
            'permission_request': 'Permission Request',
        }
        return labels.get(self.message.role, self.message.role.title())

    def _set_html_content(self, ctrl: wx.html.HtmlWindow, text: str,
                          fmt: Optional[str] = None):
        """Set HTML content on an HtmlWindow (§8.5).

        Height is measured after Layout in WorkflowHistoryView._measure_widget
        (which reads the HtmlWindow's internal representation at the laid-out
        §8.2 width); measuring here at the control's default width produced
        over-tall min heights and excessive whitespace.

        ``fmt`` routes the body: ``'markdown'`` renders through
        markdown_render.markdown_to_html (the module's ``''`` contract
        covers a missing mistune or a rendering failure -- both fall
        back to this plain escape+<br> body).
        """
        body = ""
        if isinstance(fmt, str) and fmt.lower() == 'markdown':
            body = markdown_to_html(text)
        if not body:
            escaped = html_lib.escape(text)
            body = escaped.replace('\n', '<br>\n')
        bg_hex = self.bg_color.GetAsString(wx.C2S_HTML_SYNTAX)
        fg_hex = themes.to_wx(themes.active().text_fg).GetAsString(wx.C2S_HTML_SYNTAX)

        html_content = (
            f'<html><body bgcolor="{bg_hex}">'
            f'<font face="sans-serif" size="3" color="{fg_hex}">{body}</font>'
            f'</body></html>'
        )

        ctrl.SetPage(html_content)
        ctrl.SetBackgroundColour(self.bg_color)

    def _create_html_text(self, text: str,
                          fmt: Optional[str] = None) -> wx.html.HtmlWindow:
        """Create an HtmlWindow for displaying text content with auto-sizing.

        Markdown windows also bind the link-click handler so that live
        [text](url) links open in the default browser; plain windows keep
        today's inert-link behavior exactly.
        """
        ctrl = wx.html.HtmlWindow(self, style=wx.html.HW_SCROLLBAR_NEVER)
        self._set_html_content(ctrl, text, fmt)
        if isinstance(fmt, str) and fmt.lower() == 'markdown':
            ctrl.Bind(wx.html.EVT_HTML_LINK_CLICKED, self._on_html_link_clicked)
        return ctrl

    def _on_html_link_clicked(self, event: wx.html.HtmlLinkEvent):
        """Open a clicked markdown link in the default browser.

        wxHtmlWindow does nothing with link clicks on its own, so before
        this handler links rendered styled-but-inert. The href is
        scheme-checked before it reaches the browser: LLM-controlled link
        targets must never launch arbitrary protocol handlers (file://,
        javascript:, anything else -- http/https only).
        """
        try:
            info = event.GetLinkInfo()
            href = info.GetHref() if info is not None else None
        except WX_FAILURES:
            return
        if not href:
            return
        scheme = href.split(':', 1)[0].lower()
        if scheme not in ('http', 'https'):
            logger.debug("Blocked markdown link (non-http(s) scheme): %r", href)
            return
        try:
            import webbrowser
            webbrowser.open(href)
        except Exception:
            # A failed browser launch must never take the card down.
            logger.exception("Failed to open markdown link %r", href)
        # No event.Skip() on purpose: wxHTML has no default link action,
        # so the click is fully consumed here either way.

    def _on_char_hook(self, event: wx.KeyEvent):
        """Ctrl+A: select ALL of this card's text, on-screen or not.

        CHAR_HOOK is delivered to the focused window first and propagates
        up the parent chain, so this fires whenever the focus is in this
        card (the panel itself, the message HtmlWindow, or the Fork
        button). The message HtmlWindow is created without scrollbars and
        sized to its full content, so SelectAll() reaches text currently
        scrolled outside the viewport. Every other key/event path skips
        so native behavior (tab navigation, Ctrl+C copy, typing in nested
        controls) is untouched.
        """
        if (event.ControlDown() and not event.AltDown()
                and not event.ShiftDown()
                and event.GetKeyCode() in (ord('A'), ord('a'))):
            ctrl = self.content_ctrl
            if ctrl is not None and self.populated:
                try:
                    ctrl.SelectAll()
                    # Move focus into the text so the follow-up native
                    # Ctrl+C / right-click Copy targets this selection.
                    ctrl.SetFocus()
                    ctrl.Refresh()
                except WX_FAILURES:
                    pass
                return  # consumed; do NOT Skip (stops propagation)
        event.Skip()

    def _add_permission_request_ui(self, sizer: wx.Sizer, message: ConversationMessage):
        """Add UI for permission request."""
        request: PermissionRequest = message.metadata.get('request')
        if not request:
            return

        # Tool name and reason
        info_text = f"Tool: {request.tool_name}\nReason: {request.reason}"
        info_ctrl = self._create_html_text(info_text)
        sizer.Add(info_ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        # Additional details
        if request.details:
            details_str = "\n".join(f"{k}: {v}" for k, v in request.details.items())
            details_ctrl = self._create_html_text(details_str)
            sizer.Add(details_ctrl, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5)

        # Approve/Deny buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.explanation_ctrl = wx.TextCtrl(self, value="",
                                           style=wx.TE_MULTILINE,
                                           size=(-1, 40))
        self.explanation_ctrl.SetHint("Explanation (required if denying)")
        btn_sizer.Add(self.explanation_ctrl, 1, wx.RIGHT, 5)

        approve_btn = wx.Button(self, label="Approve")
        approve_btn.Bind(wx.EVT_BUTTON, self._on_approve)
        btn_sizer.Add(approve_btn, 0, wx.RIGHT, 5)

        deny_btn = wx.Button(self, label="Deny")
        deny_btn.Bind(wx.EVT_BUTTON, self._on_deny)
        btn_sizer.Add(deny_btn, 0)

        sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)

    def _on_approve(self, event: wx.Event):
        """Handle approve button."""
        if self.on_edit:  # Using on_edit callback for permission response
            self.on_edit()

    def _on_deny(self, event: wx.Event):
        """Handle deny button."""
        if self.on_edit:
            self.on_edit()

    def _format_metadata(self, metadata: dict) -> str:
        """Format metadata as a string."""
        from datetime import datetime

        # For assistant messages (MODEL_MESSAGE_TO_USER), show tags instead of
        # redundant agent/model/token info (already shown in the RequestCard)
        if self.message.role == 'assistant':
            parts = []
            tags = metadata.get('tags')
            if tags:
                parts.append(f"Tags: {', '.join(tags)}")
            return " | ".join(parts)

        # Original formatting for other roles (tool, etc.)
        parts = []
        # Universal header fields (first 6)
        if metadata.get('agent_type'):
            parts.append(f"Agent: {metadata['agent_type']}")
        if metadata.get('provider'):
            parts.append(f"Provider: {metadata['provider']}")
        model = metadata.get('model')
        requested_model = metadata.get('requested_model')
        if model and requested_model and model != requested_model:
            parts.append(f"Model: {model} (but requested {requested_model})")
        elif model:
            parts.append(f"Model: {model}")
        elif requested_model:
            parts.append(f"Model: {requested_model}")
        if metadata.get('prompt_tokens'):
            parts.append(f"Prompt: {metadata['prompt_tokens']} tokens")
        if metadata.get('completion_tokens'):
            parts.append(f"Completion: {metadata['completion_tokens']} tokens")
        if metadata.get('cached_tokens'):
            parts.append(f"Cached: {metadata['cached_tokens']} tokens")
        if metadata.get('reasoning_tokens'):
            parts.append(f"Reasoning: {metadata['reasoning_tokens']} tokens")
        # Calculate and display request duration if we have timing info
        # Use start_time from request header and end_time from response header
        if metadata.get('end_time') and metadata.get('start_time'):
            try:
                end_time = parse_utc_timestamp(metadata['end_time'])
                start_time = parse_utc_timestamp(metadata['start_time'])
                duration = (end_time - start_time).total_seconds()
                if duration >= 0:
                    parts.append(f"Time: {duration:.1f}s")
            except (ValueError, TypeError):
                pass
        # Tool execution status
        if 'execution_successful' in metadata:
            success = metadata['execution_successful']
            parts.append("Status: Success" if success else "Status: Failed")
        return " | ".join(parts)
