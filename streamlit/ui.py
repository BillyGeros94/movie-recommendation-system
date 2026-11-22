"""The app's look: monochrome and type-led (ink on off-white, the inverse in dark), hairlines instead of boxes,
small mono labels. The colours are the same tokens as .streamlit/config.toml."""
import streamlit as st

PALETTE = {
    "light": {"bg": "#fbfbfa", "ink": "#0b0b0b", "ink2": "#4f4e4a", "muted": "#74736e",
              "line": "rgba(11,11,11,.10)", "soft": "#f1f1ef"},
    "dark": {"bg": "#0d0d0d", "ink": "#ededea", "ink2": "#bdbcb6", "muted": "#8f8e88",
             "line": "rgba(237,237,234,.12)", "soft": "#1a1a19"},
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root { --bg: {bg}; --ink: {ink}; --ink2: {ink2}; --muted: {muted}; --line: {line}; --soft: {soft};
  --mono: 'JetBrains Mono', ui-monospace, monospace; }
.stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp input, .stApp textarea, .stApp button p,
.hero, .hero .title, .brand, .lead, .note { font-family: 'Inter', system-ui, sans-serif; }
.block-container { max-width: {width}; padding-top: .6rem; padding-bottom: 3rem; }
/* Streamlit's header is invisible but lies over the page: let clicks through, except on its own buttons */
header[data-testid="stHeader"] { background: transparent; pointer-events: none; }
header[data-testid="stHeader"] * { pointer-events: none; }
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a { pointer-events: auto; }
.mono { font-family: var(--mono); font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
/* the top bar, pinned */
.mark .bg { fill: var(--ink); } .mark .s { stroke: var(--bg); fill: none; } .mark .f { fill: var(--bg); }
.brand { display: flex; align-items: center; gap: .55rem; height: 2.4rem; }
.brand b { font-size: 1.02rem; font-weight: 600; letter-spacing: -.02em; color: var(--ink); white-space: nowrap; }
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar) { position: sticky; top: 0; z-index: 1000; background: var(--bg); }
.st-key-topbar { border-bottom: 1px solid var(--line); padding: .6rem 0; }
/* the opening */
.hero { margin: 5vh 0 2rem; }
.hero .mono { margin-bottom: 1.1rem; }
.hero .title { font-size: 2.9rem; font-weight: 600; letter-spacing: -.045em; line-height: 1.04; margin: 0 0 1.1rem;
  color: var(--ink); }
.hero .title span { color: var(--muted); }
.hero p { color: var(--ink2); font-size: 1rem; line-height: 1.6; margin: 0; max-width: 580px; }
.label { font-family: var(--mono); font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
  color: var(--muted); margin: 2rem 0 .6rem; }
.lead { color: var(--ink2); font-size: .92rem; line-height: 1.55; margin: -.2rem 0 .8rem; }
.note { color: var(--muted); font-size: .82rem; line-height: 1.5; }
.foot { margin-top: 2.6rem; padding-top: .9rem; border-top: 1px solid var(--line); }
/* controls: pills, hairline inputs */
.stButton button, .stFormSubmitButton button, .stDownloadButton button { border-radius: 999px; }
button[kind^="primary"] p { color: var(--bg) !important; }
.stButton button[kind="secondary"], .stFormSubmitButton button[kind="secondary"] { background: transparent;
  border: 1px solid var(--line); }
.stButton button[kind="secondary"]:hover { border-color: var(--ink); color: var(--ink); }
[data-testid="stTextInputRootElement"], [data-testid="stTextInput"] div[data-baseweb="input"] {
  border-radius: 12px; border: 1px solid var(--line); background: var(--soft); }
[data-testid="stForm"] { border: 0; padding: 0; }
[data-testid="stDataFrame"], [data-testid="stImage"] img { border-radius: 10px; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); overflow: hidden; padding: .2rem; }
[data-testid="stExpander"] details { border: 0; border-top: 1px solid var(--line); border-radius: 0; }
[data-testid="stExpander"] summary p { font-size: .92rem; }
[data-testid="stAlert"] > div, [data-testid="stAlertContainer"] { background: var(--soft) !important;
  color: var(--ink) !important; border-radius: 10px; }
/* numbered hairline rows (keys "row_0", "row_1", ...) */
[class*="st-key-rows"] { gap: 0; }
[class*="st-key-row_"] button { width: 100%; justify-content: flex-start; text-align: left; min-height: auto;
  border: 0 !important; border-top: 1px solid var(--line) !important; border-radius: 0 !important;
  padding: .68rem 2rem .68rem .1rem; background: transparent; position: relative; }
[class*="st-key-row_"]:last-child button { border-bottom: 1px solid var(--line) !important; }
[class*="st-key-row_"] button:hover { background: var(--soft); color: var(--ink); }
[class*="st-key-row_"] button > div, [class*="st-key-row_"] button span,
[class*="st-key-row_"] button [data-testid="stMarkdownContainer"] { width: 100%; justify-content: flex-start;
  text-align: left; }
[class*="st-key-row_"] button::after { content: "\\2197"; position: absolute; right: .6rem; color: var(--muted);
  font-size: .95rem; }
[class*="st-key-row_"] button:hover::after { color: var(--ink); }
[class*="st-key-row_"] button p { font-size: .95rem; line-height: 1.4; color: var(--ink); }
[class*="st-key-row_"] button code { font-family: var(--mono); font-size: .72rem; color: var(--muted);
  background: none; padding: 0; margin-right: 1.1rem; }
[class*="st-key-row_"] button strong { display: inline-block; min-width: 8.2rem; font-weight: 400; color: var(--muted); }
@media (max-width: 640px) {
  .hero { margin-top: 3vh; } .hero .title { font-size: 2.2rem; }
  [class*="st-key-row_"] button strong { display: block; min-width: 0; }
  .st-key-topbar [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: .5rem; }
  .st-key-topbar [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 0 0 auto !important; }
  .st-key-topbar [data-testid="stColumn"]:first-child { flex: 1 1 auto !important; }
}
{extra}
</style>
"""


def apply(width: str = "780px", extra: str = ""):
    """The page's styles. Each colour is light-dark(light, dark): the browser picks by the theme Streamlit is showing
    (the device's, or Light / Dark chosen in Streamlit's menu), at once and without a rerun."""
    css = CSS
    tokens = {k: f"light-dark({PALETTE['light'][k]}, {PALETTE['dark'][k]})" for k in PALETTE["light"]}
    for k, v in {**tokens, "width": width, "extra": extra}.items():
        css = css.replace("{" + k + "}", v)
    st.markdown(css, unsafe_allow_html=True)


def mark(shapes: str) -> str:
    """A 26px rounded ink square with the app's own glyph drawn in the background colour."""
    return (f'<svg class="mark" width="26" height="26" viewBox="0 0 26 26" xmlns="http://www.w3.org/2000/svg" '
            f'aria-hidden="true"><rect class="bg" width="26" height="26" rx="7"/>{shapes}</svg>')


def top_bar(name: str, glyph: str, widths=(6, 1.6)):
    """The pinned bar: the mark and name on the left; returns the right-hand column for the app's controls."""
    with st.container(key="topbar"):
        left, right = st.columns(list(widths), vertical_alignment="center")
        left.markdown(f'<div class="brand">{mark(glyph)}<b>{name}</b></div>', unsafe_allow_html=True)
        return right


def hero(eyebrow: str, head: str, tail: str, lead: str):
    st.markdown(f'<div class="hero"><div class="mono">{eyebrow}</div><div class="title">{head}<br><span>{tail}</span>'
                f'</div><p>{lead}</p></div>', unsafe_allow_html=True)


def label(text: str):
    st.markdown(f'<div class="label">{text}</div>', unsafe_allow_html=True)


def foot(text: str):
    st.markdown(f'<div class="mono foot">{text}</div>', unsafe_allow_html=True)
