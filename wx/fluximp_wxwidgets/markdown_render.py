"""Markdown -> HTML fragment rendering for wx.html.HtmlWindow.

Assistant cards whose model_message_to_user record carries
format='markdown' route their message body through markdown_to_html()
instead of the plain escape+<br> path. mistune (3.x) supplies the parser
and escaper; this module is the wxHTML policy layer:

- the renderer subclass constrains output to the tag subset
  wx.html.HtmlWindow supports (mistune's defaults for headings, lists,
  links, blockquotes, tables, and emphasis already do -- blockquote,
  h1-h6, ul/ol/li, table/tr/th/td, del/em/strong, hr, a, p, br),
- raw inline/block HTML and markdown images NEVER emit live markup
  (they render as escaped literal text instead), so no sanitizer pass
  or extra dependency is needed,
- the format's failure modes degrade to the plain-text path instead of
  raising: a missing mistune package renders '' from markdown_to_html()
  (the card then falls back to its plain escape+<br> body), and any
  internal rendering error degrades to escaped plain text.

This module imports only mistune + stdlib (no wx), so the unit tests in
test/tool_tests/markdown_render_tests.py run outside a GUI process. The
renderer overrides are deliberately signature-tolerant (**kwargs /
*args forwarding) to absorb mistune minor-version call-shape drift; the
unit tests are the tripwire that fails first when the installed
version's behavior changes.
"""

import html as html_lib
import logging

try:
    import mistune
    MARKDOWN_AVAILABLE = True
except ImportError:  # tolerate a machine without the dependency
    mistune = None  # type: ignore[assignment]
    MARKDOWN_AVAILABLE = False

logger = logging.getLogger(__name__)

if MARKDOWN_AVAILABLE:

    class _WxHtmlRenderer(mistune.HTMLRenderer):
        """Renderer constrained to wxHTML-safe output.

        Every override must fail safe: any drift in the installed
        mistune's call shapes degrades to escaped/literal output (or a
        logged fall-through in markdown_to_html), never an exception
        escaping into a card build.
        """

        def text(self, text, **kwargs):
            # Explicit rather than trust in the default's escaping: plain
            # text tokens are the bulk of LLM output, and the
            # markdown-rendering tests pin this behavior.
            return html_lib.escape(text)

        def softbreak(self):
            # Approved rendering semantics: a lone \n inside a paragraph
            # soft-joins (space), like real markdown rendering. Flip to
            # '<br>\n' if chat-style per-line breaks are ever wanted.
            return " "

        def codespan(self, text, **kwargs):
            # Code spans are escaped here rather than left to the
            # default, same policy as text().
            return "<code>" + html_lib.escape(text) + "</code>"

        def block_code(self, code, info=None, **kwargs):
            # Fenced code content is fully escaped; the info string is
            # consumed (wxHTML has no syntax-highlighting stack).
            return "<pre><code>" + html_lib.escape(code) + "</code></pre>"

        def inline_html(self, html, **kwargs):
            # LLM-dropped raw inline HTML must appear as literal text; it
            # must never reach wxHTML as markup.
            return html_lib.escape(html)

        def block_html(self, html, **kwargs):
            # Same policy at block level (wrapped in <p> so a multi-line
            # raw fragment keeps its visual separation).
            return "<p>" + html_lib.escape(html) + "</p>"

        def image(self, *args, **kwargs):
            # No <img> emission: wxHTML would attempt local file:// loads
            # for LLM-emitted tags, and remote references dangle. The
            # default renderer's tag is shown as escaped literal text so
            # the user still sees the alt text and source URL. Blind
            # forwarding keeps this working across mistune's argument
            # reorderings (v2-style image(text, url, title) vs v3-style
            # image(url, text, title)).
            try:
                tag = super().image(*args, **kwargs)
            except Exception:
                return html_lib.escape("[image]")
            return html_lib.escape(tag)

    # Module-level singleton pipeline. The GUI thread is the only caller
    # (reload replay is marshaled onto it), so no locking is needed.
    _MD = mistune.create_markdown(
        renderer=_WxHtmlRenderer(),
        plugins=["table", "strikethrough"],
    )

else:
    _MD = None  # type: ignore[assignment]


def markdown_to_html(md: str) -> str:
    """Render ``md`` as an HTML fragment for a wxHTML SetPage body.

    Returns '' when mistune is unavailable or the input is empty (the
    caller falls back to its plain-text rendering path) and never
    raises: any internal failure degrades to escaped plain text so a
    card build cannot be broken by a malformed message.
    """
    if not MARKDOWN_AVAILABLE or not md:
        return ""
    try:
        return _MD(md)
    except Exception:
        logger.exception("markdown_to_html failed; degrading to plain text")
        return html_lib.escape(md).replace("\n", "<br>\n")
