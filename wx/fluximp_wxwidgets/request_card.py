"""RequestCard - Groups all messages for one API request-response cycle."""

import wx
import wx.html
import html as html_lib
import logging
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, List, Any

from fluximp_wxwidgets.collapsible_section import (
    CollapsibleSection,
    _WidgetLoggerProxy,
    _WIDGET_TRACE,
)
from fluximp_protocol.user_agent.file_io.defs import parse_utc_timestamp
from fluximp_wxwidgets.conversation_manager import ConversationMessage, PermissionRequest

logger = logging.getLogger(__name__)


class RequestCard(wx.Panel):
    """Groups all messages for one API request-response cycle."""

    def __init__(self, parent, fork: int, turn: int,
                 metadata: dict = None,
                 on_permission_response: Optional[Callable] = None,
                 on_user_toggle: Optional[Callable] = None,
                 is_conditional: bool = False,
                 conditional_agent_name: str = "",
                 is_system_agent: bool = False,
                 system_agent_type: str = "",
                 logger: Optional[logging.Logger] = None,
                 pos=wx.DefaultPosition):
        # Logging plan (2026-10-10 resume-desync diagnosis): the card's ~15
        # per-step DEBUG lines are silenced by default (debug-dropping
        # proxy — INFO and above pass through) and ONE INFO identity line
        # per card replaces them. Set FLUXIMP_WIDGET_TRACE=1 to restore the
        # raw logger and the per-step trace for a specific repro.
        base_logger = logger or logging.getLogger(__name__)
        self._logger = (base_logger if _WIDGET_TRACE
                        else _WidgetLoggerProxy(base_logger))
        self._logger.debug('RequestCard.__init__: START fork=%s, turn=%s', fork, turn)
        super().__init__(parent, pos=pos)
        self._logger.debug('RequestCard.__init__: super() done')

        self._fork = fork
        self._turn = turn
        self._metadata = metadata or {}
        self._on_permission_response = on_permission_response
        self._on_user_toggle = on_user_toggle
        self.is_conditional = is_conditional
        self.conditional_agent_name = conditional_agent_name
        self.is_system_agent = is_system_agent
        self.system_agent_type = system_agent_type
        
        # State tracking
        self._is_pending = True
        self._pending_start_time: Optional[datetime] = None
        self._pending_timer: Optional[wx.Timer] = None
        
        # Subcard references
        self._content_section: Optional[CollapsibleSection] = None
        self._reasoning_section: Optional[CollapsibleSection] = None
        self._tool_sections: List[Dict] = []
        self._pending_permission: Optional[Dict] = None
        self._metadata_label: Optional[wx.StaticText] = None
        self._report_sections: List[CollapsibleSection] = []
        
        # System agent subcard tracking (for scrutinator etc.)
        self._system_agent_subcards: Dict[str, Dict] = {}
        self._build_ui()
        # Logging plan (2026-10-10 resume-desync diagnosis): ONE identity
        # line per card replaces the per-step debug trace — request_seq is
        # the pairing key the desync scrambled, and id() disambiguates the
        # same-label cards it doubled.
        self._logger.info(
            'RequestCard: created id=%s fork=%s turn=%s request_seq=%s '
            'conditional=%s agent=%r system=%r',
            id(self), self._fork, self._turn,
            self._metadata.get('request_seq'),
            self.is_conditional, self.conditional_agent_name,
            self.system_agent_type)
        self._logger.debug('RequestCard.__init__: COMPLETE')

    def _build_ui(self):
        """Build the request card UI."""
        self._logger.debug('RequestCard._build_ui: START')
        self._main_sizer = wx.BoxSizer(wx.VERTICAL)

        # System agents use gray palette; conditional agents use amber; primary agents use yellow/green
        if self.is_system_agent:
            pending_color = wx.Colour(252, 252, 252)     # Very light gray
            response_color = wx.Colour(248, 248, 248)    # Light gray
            header_label = f"⚙ {self.system_agent_type}"
        elif self.is_conditional:
            pending_color = wx.Colour(255, 245, 230)     # Warm amber tint
            response_color = wx.Colour(255, 236, 210)    # Light amber
            header_label = f"→ {self.conditional_agent_name} (Fork {self._fork}, Turn {self._turn})"
        else:
            pending_color = wx.Colour(255, 250, 230)     # Yellow tint
            response_color = wx.Colour(230, 255, 230)    # Green tint
            header_label = f"API Request (Fork {self._fork}, Turn {self._turn})"

        self._pending_color = pending_color
        self._response_color = response_color

        self._logger.debug('RequestCard._build_ui: creating header CollapsibleSection')
        self._header_section = self._new_section(
            'header',
            label=header_label,
            expanded=True,
            indent=0,
            header_color=pending_color,
            auto_collapsible=False,
            # Primary and conditional request cards put the pending/
            # cancelled status on its own centered line under the header
            # text so the right-justified identity span never crowds it;
            # system-agent cards keep the original inline placement.
            extra_below=not self.is_system_agent
        )
        self._logger.debug('RequestCard._build_ui: header created, setting indicator')
        self._header_section.set_indicator("...")
        self._main_sizer.Add(self._header_section, 0, wx.EXPAND)

        self._logger.debug('RequestCard._build_ui: creating content area sizer')
        self._content_area_sizer = wx.BoxSizer(wx.VERTICAL)
        self._main_sizer.Add(self._content_area_sizer, 0, wx.EXPAND)

        self._logger.debug('RequestCard._build_ui: calling SetSizer')
        self.SetSizer(self._main_sizer)
        self.SetBackgroundColour(pending_color)

        # Add colored left border for conditional and system agent cards
        if self.is_conditional:
            self.SetForegroundColour(wx.Colour(210, 150, 60))  # Amber accent
        elif self.is_system_agent:
            self.SetForegroundColour(wx.Colour(160, 160, 160))  # Gray accent

        # Request cards (primary and conditional) show Agent/Provider/Model
        # in a right-justified span at the top of the card. The identity
        # metadata arrives in the request-start metadata, so the span is
        # populated while the card is still pending.
        self._update_header_identity_span()
        self._logger.debug('RequestCard._build_ui: COMPLETE')

    # -- Pending state management --

    def set_pending(self):
        """Set the card to pending state with a timer."""
        self._logger.debug('RequestCard.set_pending: START')
        self._stop_pending_timer()  # Idempotent: drop any previous pending timer
        self._is_pending = True
        self._pending_start_time = datetime.now(timezone.utc)
        self._logger.debug('RequestCard.set_pending: setting indicator')
        self._header_section.set_indicator("...")
        self._logger.debug('RequestCard.set_pending: setting extra text')
        self._header_section.set_extra("pending: 0s")
        self._logger.debug('RequestCard.set_pending: setting header color')
        self._header_section.set_header_color(self._pending_color)
        self.SetBackgroundColour(self._pending_color)
        self._logger.debug('RequestCard.set_pending: creating timer')
        self._pending_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_pending_timer, self._pending_timer)
        self._pending_timer.Start(1000)
        self._logger.debug('RequestCard.set_pending: timer started, calling Layout')
        self.Layout()
        self._logger.debug('RequestCard.set_pending: COMPLETE')

    def _on_pending_timer(self, event: wx.TimerEvent):
        """Update the elapsed time display."""
        if not self._is_pending:
            return
        if self._pending_start_time:
            elapsed = (datetime.now(timezone.utc) - self._pending_start_time).total_seconds()
            self._header_section.set_extra(f"pending: {elapsed:.0f}s")

    def clear_pending(self):
        """Clear the pending state (e.g., when cancel is acknowledged).
        
        Only changes state if the card is still pending. If the response
        has already arrived, this is a no-op.
        """
        if not self._is_pending:
            return
        self._stop_pending_timer()
        self._is_pending = False
        self._header_section.set_indicator("X")
        self._header_section.set_extra("cancelled")
        self._header_section.set_header_color(wx.Colour(255, 230, 230))
        self.SetBackgroundColour(wx.Colour(255, 230, 230))
        self.Layout()

    def _stop_pending_timer(self):
        """Stop the pending timer."""
        if self._pending_timer:
            self._pending_timer.Stop()
            self._pending_timer = None

    def stop_pending_timer(self):
        """Stop the pending timer.

        Public wrapper used when the conversation tab is closing, so a
        running 1-second pending timer cannot fire against a card whose
        window is about to be destroyed.
        """
        self._stop_pending_timer()

    def set_static_pending(self):
        """Set the card to a static pending state (no elapsed-time timer).

        Used at reload finalization (§7.4 step 4): a pending card with no
        running process still shows the pending indicator, but the 1s
        elapsed-time timer is never started (nothing will resolve it live).
        """
        self._stop_pending_timer()
        self._is_pending = True
        self._pending_start_time = None
        self._header_section.set_indicator("...")
        self._header_section.set_extra("pending")
        self._header_section.set_header_color(self._pending_color)
        self.SetBackgroundColour(self._pending_color)
        self.Layout()

    @property
    def is_pending(self) -> bool:
        """Whether this card is still waiting for a response."""
        return self._is_pending

    # -- Response handling --

    def set_response(self, content: str, reasoning: str = "", metadata: dict = None, collapsed: bool = True):
        """Handle the API response - add content and reasoning subcards.
        
        Args:
            content: The response content text
            reasoning: The reasoning text (if any)
            metadata: Response metadata (model, tokens, timing)
            collapsed: If True, create content/reasoning sections as collapsed.
                       Sections are collapsed by default (§8.3); the expansion
                       policy applied by the history controller can override.
        """
        self._logger.debug("RequestCard.set_response: START, content_len=%d, reasoning_len=%d, collapsed=%s", len(content), len(reasoning), collapsed)
        self._stop_pending_timer()
        self._is_pending = False

        # Merge response metadata (model, tokens, timing) into card metadata
        if metadata:
            self._metadata.update(metadata)

        # Update header
        if self.is_system_agent and self._metadata.get('success') is False:
            self._header_section.set_indicator("FAIL")
        else:
            self._header_section.set_indicator("OK")
        self._header_section.set_extra("")
        self._header_section.set_header_color(self._response_color)
        self.SetBackgroundColour(self._response_color)
        # The response metadata may complete the identity span (actual
        # model); keep the span in sync with the merged metadata.
        self._update_header_identity_span()

        # Add content subcard (collapsed by default)
        if content:
            self._logger.debug('RequestCard.set_response: creating content section')
            self._content_section = self._new_section(
                'content',
                label="Content",
                expanded=not collapsed,
                indent=20,
                header_color=wx.Colour(240, 248, 240),
                content_color=wx.Colour(245, 255, 245)
            )
            self._content_section.add_rich_text(content)
            self._content_area_sizer.Add(self._content_section, 0, wx.EXPAND | wx.ALL, 2)

        # Add reasoning subcard (collapsed initially)
        if reasoning:
            self._logger.debug('RequestCard.set_response: creating reasoning section')
            self._reasoning_section = self._new_section(
                'reasoning',
                label="Reasoning",
                expanded=False,
                indent=20,
                header_color=wx.Colour(240, 248, 240),
                content_color=wx.Colour(245, 255, 245)
            )
            self._reasoning_section.add_rich_text(reasoning)
            self._content_area_sizer.Add(self._reasoning_section, 0, wx.EXPAND | wx.ALL, 2)

        # Add metadata footer
        self._add_metadata_footer()

        self.Layout()
        self._logger.debug('RequestCard.set_response: COMPLETE')

    def _add_metadata_footer(self):
        """Add metadata footer with model info, tokens, timing."""
        meta_str = self._format_metadata(self._metadata)
        if meta_str:
            self._metadata_label = wx.StaticText(self, label=meta_str)
            self._metadata_label.SetForegroundColour(wx.Colour(128, 128, 128))
            self._main_sizer.Add(self._metadata_label, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

    def _format_metadata(self, metadata: dict) -> str:
        """Format metadata as a footer string."""
        parts = []
        is_sys = metadata.get('is_system_agent')

        if is_sys:
            # System agent specific metadata
            if metadata.get('system_agent_subdir'):
                parts.append(f"Dir: {metadata['system_agent_subdir']}")
            if metadata.get('success') is not None:
                parts.append("Success" if metadata['success'] else "Failed")
            start = metadata.get('system_agent_start_time')
            end = metadata.get('system_agent_stop_time')
            if start and end:
                try:
                    start_dt = parse_utc_timestamp(start)
                    end_dt = parse_utc_timestamp(end)
                    duration = (end_dt - start_dt).total_seconds()
                    if duration >= 0:
                        parts.append(f"Run: {duration:.1f}s")
                except (ValueError, TypeError):
                    pass
            # System agents (scrutinator/summarizer) keep Provider/Model in
            # the footer — their cards are intentionally left unchanged.
            parts.extend(self._format_model_parts(metadata))
        # else: request cards (primary and conditional agents) render
        # Agent/Provider/Model in the header's right-justified span
        # (_update_header_identity_span) instead of this footer line, which
        # otherwise truncates on narrow cards.

        if metadata.get('prompt_tokens') is not None:
            parts.append(f"Prompt: {metadata['prompt_tokens']} tokens")
        if metadata.get('completion_tokens') is not None:
            parts.append(f"Completion: {metadata['completion_tokens']} tokens")
        if metadata.get('cached_tokens') is not None:
            parts.append(f"Cached: {metadata['cached_tokens']} tokens")
        if metadata.get('reasoning_tokens') is not None:
            parts.append(f"Reasoning: {metadata['reasoning_tokens']} tokens")
        if metadata.get('end_time') and metadata.get('start_time'):
            try:
                end_time = parse_utc_timestamp(metadata['end_time'])
                start_time = parse_utc_timestamp(metadata['start_time'])
                duration = (end_time - start_time).total_seconds()
                if duration >= 0:
                    label = "API" if is_sys else "Time"
                    parts.append(f"{label}: {duration:.1f}s")
            except (ValueError, TypeError):
                pass
        return " | ".join(parts)

    @staticmethod
    def _format_model_parts(metadata: dict) -> List[str]:
        """Format the Provider/Model fragments of the metadata line."""
        parts = []
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
        return parts

    @classmethod
    def _format_identity(cls, metadata: dict) -> str:
        """Format the Agent/Provider/Model span for the card header.

        The combined identity string is too wide for the metadata footer
        (it truncated there), so it renders in its own right-justified
        span at the top of the card instead.
        """
        parts = []
        if metadata.get('agent_type'):
            parts.append(f"Agent: {metadata['agent_type']}")
        parts.extend(cls._format_model_parts(metadata))
        return " | ".join(parts)

    def _update_header_identity_span(self):
        """Update the header's right-justified Agent/Provider/Model span.

        Request cards (primary and conditional agents) show this
        identity in the header span instead of the metadata footer;
        system-agent cards (scrutinator/summarizer) keep their original
        footer and never create the span.
        """
        if self.is_system_agent or self._metadata.get('is_system_agent'):
            return
        self._header_section.set_right_text(
            self._format_identity(self._metadata))

    def add_system_agent_subcard(self, agent_type: str, metadata: dict):
        """Add a system agent subcard within this card's content area.

        Used for system agents like the scrutinator that run during tool
        execution.  The subcard appears as a nested gray section between
        tool sections.
        """
        subcard = self._new_section(
            'system_agent',
            ordinal=len(self._system_agent_subcards),
            label=f"⚙ {agent_type}",
            expanded=True,
            indent=20,
            header_color=wx.Colour(252, 252, 252),
            content_color=wx.Colour(248, 248, 248),
        )
        subcard.set_indicator("...")
        subcard.set_extra("running")

        self._system_agent_subcards[agent_type] = {
            'section': subcard,
            'metadata': dict(metadata),
        }

        # Insert before the last tool section so the subcard appears above
        # the tool use it was run for (scrutinator runs before tool execution).
        insert_pos = self._content_area_sizer.GetItemCount()  # Default: end
        if self._tool_sections:
            last_tool_section = self._tool_sections[-1].get('section')
            if last_tool_section:
                children = self._content_area_sizer.GetChildren()
                for i, child in enumerate(children):
                    if child.GetWindow() is last_tool_section:
                        insert_pos = i
                        break

        self._content_area_sizer.Insert(insert_pos, subcard, 0, wx.EXPAND | wx.ALL, 2)
        self.Layout()

    def update_system_agent_subcard(self, agent_type: str, metadata: dict):
        """Update a system agent subcard with stopped status and metadata."""
        info = self._system_agent_subcards.get(agent_type)
        if not info:
            return

        info['metadata'].update(metadata)
        section = info['section']

        success = metadata.get('success')
        if success is False:
            section.set_indicator("FAIL")
        else:
            section.set_indicator("OK")
        section.set_extra("")

        meta_str = self._format_metadata(info['metadata'])
        if meta_str:
            section.add_rich_text(meta_str)

        self.Layout()

    # -- Tool use handling --

    def add_tool_use(self, tool_message: ConversationMessage):
        """Add a tool use section to the card."""
        self._logger.debug('RequestCard.add_tool_use: START')
        tool_info = {
            'message': tool_message,
            'section': None,
            'permission_section': None,
            'result_section': None,
        }

        # Parse tool call info for label
        tool_calls = tool_message.metadata.get('tool_calls')
        label = "Tool"
        if tool_calls:
            names = [tc[0] for tc in tool_calls]
            label = ", ".join(names)
        elif tool_message.content:
            import json
            try:
                data = json.loads(tool_message.content)
                if isinstance(data, dict) and 'tool_calls' in data:
                    names = [tc.get('name', 'unknown') for tc in data['tool_calls']]
                    label = ", ".join(names)
            except (json.JSONDecodeError, TypeError):
                pass

        self._logger.debug('RequestCard.add_tool_use: creating tool section for %s', label)
        tool_section = self._new_section(
            'tool',
            ordinal=len(self._tool_sections),
            label=label,
            expanded=False,
            indent=20,
            header_color=wx.Colour(255, 250, 200),
            content_color=wx.Colour(255, 255, 245)
        )
        tool_section.set_extra("...")

        if tool_message.content:
            tool_section.add_rich_text(tool_message.content, min_height=50)

        self._content_area_sizer.Add(tool_section, 0, wx.EXPAND | wx.ALL, 2)
        tool_info['section'] = tool_section
        self._tool_sections.append(tool_info)

        # If there's a pending permission, associate it with this tool section
        if self._pending_permission:
            pending = self._pending_permission
            tool_info['permission_section'] = pending['permission_section']
            tool_info['permission_message'] = pending['permission_message']
            tool_info['explanation_ctrl'] = pending['explanation_ctrl']
            tool_info['btn_panel'] = pending.get('btn_panel')
            self._pending_permission = None

        self.Layout()
        self._logger.debug('RequestCard.add_tool_use: COMPLETE')

    def add_permission_request(self, permission_message: ConversationMessage, resolved: bool = False):
        """Add a permission request sub-section below the latest tool section.
        
        If no tool sections exist yet (permission request arrived before tool use,
        which is the normal protocol order), the permission section is added
        directly to the content area and stored as _pending_permission for later
        association when the tool use arrives.
        """
        self._logger.debug('RequestCard.add_permission_request: START, resolved=%s', resolved)
        request: PermissionRequest = permission_message.metadata.get('request')
        if not request:
            return

        # §8.3 per-instance ordinal: this role's sections already on the
        # card (each tool's plus one still-pending) fix the identity the
        # toggle callback will report for THIS section. Keyed by creation
        # identity, a later tool association cannot renumber it (defect B1).
        perm_ordinal = (
            sum(1 for ti in self._tool_sections
                if ti.get('permission_section') is not None)
            + (1 if self._pending_permission is not None else 0)
        )
        permission_section = self._new_section(
            'permission',
            ordinal=perm_ordinal,
            label=f"Permission Request: {request.tool_name}",
            expanded=True,
            indent=40,
            header_color=wx.Colour(255, 240, 230),
            content_color=wx.Colour(255, 250, 245)
        )

        info_text = f"Tool: {request.tool_name}\nReason: {request.reason}"
        permission_section.add_rich_text(info_text, min_height=50)

        if request.details:
            details_str = "\n".join(f"{k}: {v}" for k, v in request.details.items())
            permission_section.add_rich_text(details_str, min_height=50)

        # Add approve/deny buttons
        btn_panel = wx.Panel(permission_section.get_content_panel())
        btn_panel.SetBackgroundColour(wx.Colour(255, 250, 245))
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        explanation_ctrl = wx.TextCtrl(btn_panel, value="", style=wx.TE_MULTILINE, size=(-1, 40))
        explanation_ctrl.SetHint("Explanation (required if denying)")
        btn_sizer.Add(explanation_ctrl, 1, wx.RIGHT, 5)

        approve_btn = wx.Button(btn_panel, label="Approve")
        deny_btn = wx.Button(btn_panel, label="Deny")

        approve_btn.Bind(wx.EVT_BUTTON, 
                        lambda e: self._on_approve(permission_message, explanation_ctrl))
        deny_btn.Bind(wx.EVT_BUTTON,
                      lambda e: self._on_deny(permission_message, explanation_ctrl))

        btn_sizer.Add(approve_btn, 0, wx.RIGHT, 5)
        btn_sizer.Add(deny_btn, 0)
        btn_panel.SetSizer(btn_sizer)
        
        permission_section.get_content_sizer().Add(btn_panel, 0, wx.EXPAND | wx.ALL, 5)

        if resolved:
            permission_section.collapse(notify=False)
            btn_panel.Hide()

        self._content_area_sizer.Add(permission_section, 0, wx.EXPAND | wx.ALL, 2)

        if self._tool_sections:
            # Attach to the latest tool section
            tool_info = self._tool_sections[-1]
            tool_info['permission_section'] = permission_section
            tool_info['permission_message'] = permission_message
            tool_info['explanation_ctrl'] = explanation_ctrl
            tool_info['btn_panel'] = btn_panel
        else:
            # No tool sections yet - permission request arrived before tool use.
            # Store as pending for later association when add_tool_use is called.
            self._pending_permission = {
                'permission_section': permission_section,
                'permission_message': permission_message,
                'explanation_ctrl': explanation_ctrl,
                'btn_panel': btn_panel,
            }

        self.Layout()
        self._logger.debug('RequestCard.add_permission_request: COMPLETE')

    def _on_approve(self, permission_message: ConversationMessage, explanation_ctrl: wx.TextCtrl):
        """Handle approve button."""
        found = False
        for tool_info in self._tool_sections:
            if tool_info.get('permission_message') is permission_message:
                if tool_info.get('permission_section'):
                    tool_info['permission_section'].collapse()
                found = True
                break

        # Also check pending permission (permission request arrived before tool use)
        if not found and self._pending_permission and self._pending_permission.get('permission_message') is permission_message:
            if self._pending_permission.get('permission_section'):
                self._pending_permission['permission_section'].collapse()

        if self._on_permission_response:
            self._on_permission_response(permission_message, approved=True, 
                                         explanation=explanation_ctrl.GetValue().strip())

    def _on_deny(self, permission_message: ConversationMessage, explanation_ctrl: wx.TextCtrl):
        """Handle deny button."""
        found = False
        for tool_info in self._tool_sections:
            if tool_info.get('permission_message') is permission_message:
                if tool_info.get('permission_section'):
                    tool_info['permission_section'].collapse()
                if tool_info.get('section'):
                    tool_info['section'].set_extra("...")
                    tool_info['section'].set_header_color(wx.Colour(255, 250, 200))
                found = True
                break

        # Also check pending permission (permission request arrived before tool use)
        if not found and self._pending_permission and self._pending_permission.get('permission_message') is permission_message:
            if self._pending_permission.get('permission_section'):
                self._pending_permission['permission_section'].collapse()

        if self._on_permission_response:
            self._on_permission_response(permission_message, approved=False,
                                         explanation=explanation_ctrl.GetValue().strip())

    def update_tool_result(self, tool_message: ConversationMessage):
        """Update a tool section with the result."""
        self._logger.debug('RequestCard.update_tool_result: START')
        for tool_info in self._tool_sections:
            if tool_info['message'] is tool_message:
                tool_section = tool_info.get('section')
                if not tool_section:
                    continue

                result = tool_message.metadata.get('result', '')
                execution_successful = tool_message.metadata.get('execution_successful') and result

                if execution_successful:
                    tool_section.set_extra("OK")
                    tool_section.set_header_color(wx.Colour(230, 255, 230))
                else:
                    tool_section.set_extra("FAIL")
                    tool_section.set_header_color(wx.Colour(255, 230, 230))

                # Add result content inside the tool section
                # Include "Result:" as part of the rich text instead of a separate
                # StaticText, so that lazy content creation in CollapsibleSection
                # preserves the correct visual order (tool call -> result label -> result text).
                if result or not execution_successful:
                    result_text = result if result else "[No result returned - execution failed or returned empty]"
                    tool_section.add_rich_text(f"Result:\n{result_text}", min_height=50)

                break

        self.Layout()
        self._logger.debug('RequestCard.update_tool_result: COMPLETE')

    def remove_tool_section(self, tool_message: ConversationMessage):
        """Remove a tool section (e.g., when send_message tool succeeds)."""
        self._logger.debug('RequestCard.remove_tool_section: START')
        for tool_info in self._tool_sections:
            if tool_info['message'] is tool_message:
                for section_key in ('section', 'permission_section', 'result_section'):
                    section = tool_info.get(section_key)
                    if section:
                        self._content_area_sizer.Detach(section)
                        section.Destroy()
                self._tool_sections.remove(tool_info)
                break

        self.Layout()
        self._logger.debug('RequestCard.remove_tool_section: COMPLETE')

    def update_metadata(self, metadata: dict):
        """Update the metadata footer with new information."""
        self._metadata.update(metadata)
        self._update_header_identity_span()
        if self._metadata_label:
            meta_str = self._format_metadata(self._metadata)
            self._metadata_label.SetLabel(meta_str)
        else:
            self._add_metadata_footer()
        self.Layout()

    def add_report_section(self, report_text: str):
        """Add a collapsible report section for conditional agent output.

        Displayed when a conditional agent submits its final report via
        the submit_report tool.  Uses amber styling to match the
        conditional card theme.
        """
        # Choose colors based on conditional/primary styling
        if self.is_conditional:
            header_bg = wx.Colour(255, 240, 217)   # Amber yellow
            content_bg = wx.Colour(255, 245, 230)   # Warm amber tint
        else:
            header_bg = wx.Colour(255, 250, 200)
            content_bg = wx.Colour(255, 255, 245)

        report_section = self._new_section(
            'report',
            ordinal=len(self._report_sections),
            label="Report",
            expanded=True,
            indent=20,
            header_color=header_bg,
            content_color=content_bg
        )
        report_section.add_rich_text(report_text, min_height=50)
        self._content_area_sizer.Add(report_section, 0, wx.EXPAND | wx.ALL, 2)
        self._report_sections.append(report_section)

        self.Layout()

    def collapse_content(self) -> bool:
        """Collapse all expanded sub-sections to free native window handles (HWNDs).

        Called when a new API request starts, so that old cards release their
        HtmlWindow HWNDs.  The user can re-expand sections by clicking on them.
        These programmatic collapses are not user toggles, so they do not
        notify the expansion-policy controller.

        Returns True when any section actually collapsed, False when every
        section was already collapsed (the steady-state repeat pass).

        F-A10 follow-up (the changed-signal): the request-start collapse
        pass walks materialized_item_ids(), which now reaches HIDDEN
        detached-cache cards too, and the caller flags a collapsed
        cached card for one converging (hidden) re-measure ONLY when a
        section actually moved — an unconditional True on the
        steady-state pass would re-measure every cached card on every
        request-start, silently converting the revisit cache's idle
        skip into churn.
        """
        changed = False
        if self._content_section and self._content_section.is_expanded:
            self._content_section.collapse(notify=False)
            changed = True
        if self._reasoning_section and self._reasoning_section.is_expanded:
            self._reasoning_section.collapse(notify=False)
            changed = True
        for tool_info in self._tool_sections:
            section = tool_info.get('section')
            if section and section.is_expanded:
                section.collapse(notify=False)
                changed = True
            perm_section = tool_info.get('permission_section')
            if perm_section and perm_section.is_expanded:
                perm_section.collapse(notify=False)
                changed = True
        if self._pending_permission:
            perm_section = self._pending_permission.get('permission_section')
            if perm_section and perm_section.is_expanded:
                perm_section.collapse(notify=False)
                changed = True
        return changed

    def _new_section(self, section_role: str, ordinal: int = 0,
                     **kwargs) -> CollapsibleSection:
        """Create a CollapsibleSection wired to this card's user-toggle
        callback.

        ``ordinal`` is this section's creation index among the card's
        same-role sections (§8.3) — the per-instance identity the toggle
        callback reports and user_overrides keys on, so one click never
        cascades to same-role siblings (defect B1). Every creation site
        passes its role's running count; iter_sections() yields the
        stored value back, keeping click and policy keyed identically
        even when display order diverges from creation order (a pending
        permission associated with a later tool section).
        """
        kwargs.setdefault('section_role', section_role)
        kwargs.setdefault('section_ordinal', int(ordinal))
        kwargs['on_user_toggle'] = self._on_user_toggle
        return CollapsibleSection(self, **kwargs)

    @staticmethod
    def _section_ordinal(section) -> int:
        """A section's stored creation ordinal (0 for foreign/legacy)."""
        try:
            return int(getattr(section, 'section_ordinal', 0) or 0)
        except (TypeError, ValueError):
            return 0

    def iter_sections(self):
        """Yield (role, ordinal, section) for every CollapsibleSection.

        Pinned order: header -> content -> reasoning -> tool (in order) ->
        permission (each tool's, then a still-pending one) -> report ->
        system_agent. Ordinals are the sections' stored creation
        identities (§8.3) — the same per-instance values the toggle
        callback reports and user_overrides keys on — so policy and
        clicks agree on ONE section instance even where display order
        diverges from creation order (a pending permission later
        associated with a tool section).
        """
        yield ('header', 0, self._header_section)
        if self._content_section is not None:
            yield ('content', self._section_ordinal(self._content_section),
                   self._content_section)
        if self._reasoning_section is not None:
            yield ('reasoning',
                   self._section_ordinal(self._reasoning_section),
                   self._reasoning_section)
        for tool_info in self._tool_sections:
            section = tool_info.get('section')
            if section is not None:
                yield ('tool', self._section_ordinal(section), section)
            perm_section = tool_info.get('permission_section')
            if perm_section is not None:
                yield ('permission', self._section_ordinal(perm_section),
                       perm_section)
        if self._pending_permission is not None:
            perm_section = self._pending_permission.get('permission_section')
            if perm_section is not None:
                yield ('permission', self._section_ordinal(perm_section),
                       perm_section)
        for report_section in self._report_sections:
            yield ('report', self._section_ordinal(report_section),
                   report_section)
        for info in self._system_agent_subcards.values():
            section = info.get('section')
            if section is not None:
                yield ('system_agent', self._section_ordinal(section),
                       section)

    def deactivate_permission(self):
        """Deactivate (hide) all permission button panels on this card.

        Called when the turn becomes terminal or a permission is resolved.
        """
        for tool_info in self._tool_sections:
            panel = tool_info.get('btn_panel')
            if panel is not None:
                try:
                    panel.Hide()
                except RuntimeError:
                    pass
        if self._pending_permission is not None:
            panel = self._pending_permission.get('btn_panel')
            if panel is not None:
                try:
                    panel.Hide()
                except RuntimeError:
                    pass
        self.Layout()

    def activate_permission(self):
        """Re-activate (show + expand) all permission surfaces on this card.

        Inverse of ``deactivate_permission``; called at reload finalization
        (§5.5 restore) to re-arm genuinely live approve/deny buttons for a
        non-terminal, unanswered permission streamed during a reload.
        """
        for tool_info in self._tool_sections:
            section = tool_info.get('permission_section')
            if section is not None:
                try:
                    section.expand(notify=False)
                except RuntimeError:
                    pass
            panel = tool_info.get('btn_panel')
            if panel is not None:
                try:
                    panel.Show()
                except RuntimeError:
                    pass
        if self._pending_permission is not None:
            section = self._pending_permission.get('permission_section')
            if section is not None:
                try:
                    section.expand(notify=False)
                except RuntimeError:
                    pass
            panel = self._pending_permission.get('btn_panel')
            if panel is not None:
                try:
                    panel.Show()
                except RuntimeError:
                    pass
        self.Layout()

    # -- Layout helpers --

