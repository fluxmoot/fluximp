"""Registry of toggleable API Request metadata items.

Single source of truth shared by the View > API Request Card Metadata menu and
RequestCard metadata formatters.  Deliberately free of wx imports so
AppSettings (constructed before the GUI toolkit is fully initialized)
can import the item keys and defaults without pulling in wx.

Each entry is ``(key, menu label, default_visible)``.  The menu label
matches the label rendered on the cards except for the token-count items,
which are suffixed with " Tokens" ("Prompt Tokens" instead of "Prompt").

Only Prompt Tokens is visible by default.  The non-API system-agent items
(subdirectory name, overall success/failure and overall run time) are
intentionally absent here — they are always shown and are not
user-toggleable.
"""

API_METADATA_ITEMS = [
    ("agent", "Agent", False),
    ("provider", "Provider", False),
    ("model", "Model", False),
    ("prompt_tokens", "Prompt Tokens", True),
    ("completion_tokens", "Completion Tokens", False),
    ("cached_tokens", "Cached Tokens", False),
    ("reasoning_tokens", "Reasoning Tokens", False),
    ("time", "Time", False),
]

API_METADATA_KEYS = [key for key, _label, _default in API_METADATA_ITEMS]

API_METADATA_DEFAULTS = {
    key: default for key, _label, default in API_METADATA_ITEMS
}
