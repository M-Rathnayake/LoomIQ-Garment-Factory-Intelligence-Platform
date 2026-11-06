"""Light / dark mode for the dashboard.

How it works:
    * ``.streamlit/config.toml`` defines a light and a dark Streamlit theme, so every
      built-in element (tables, menus, inputs, chat) is styled natively in both modes.
    * The sidebar toggle is the single control. A small script switches Streamlit's
      own theme to match, without reloading the page, so the simulated day is kept.
    * **Light is the default.** The choice is remembered in a cookie, so it survives
      a page reload or a new visit.
    * Plotly charts and the custom card styling read :func:`palette`, so they switch too.
"""

from typing import Any

import streamlit as st

COOKIE = "loomiq_theme"

# Colours per mode. Categorical and status steps come from the validated palette
# (see loomiq.utils.plotting); dark mode uses the steps tuned for a dark surface.
PALETTES: dict[str, dict[str, Any]] = {
    "light": {
        "background": "#f5f7fa",
        "card": "#ffffff",
        "text": "#1b1f24",
        "text_muted": "#5b626d",
        "grid": "#e6e9ee",
        "border": "#e2e6ec",
        "sewing": "#2a78d6",
        "finishing": "#eb6834",
        "neutral": "#9aa0a8",
        "positive": "#2a78d6",
        "negative": "#e34948",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
        "shadow": "0 1px 2px rgba(16, 24, 40, 0.06), 0 1px 3px rgba(16, 24, 40, 0.08)",
        "shadow_hover": "0 8px 24px rgba(16, 24, 40, 0.10)",
        "hero": "linear-gradient(135deg, #e8f1fc 0%, #f5f7fa 60%)",
    },
    "dark": {
        "background": "#111317",
        "card": "#1b1e24",
        "text": "#eef0f3",
        "text_muted": "#a3a9b3",
        "grid": "#2a2e36",
        "border": "#2a2e36",
        "sewing": "#3987e5",
        "finishing": "#d95926",
        "neutral": "#6b717b",
        "positive": "#3987e5",
        "negative": "#e66767",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
        "shadow": "0 1px 3px rgba(0, 0, 0, 0.45)",
        "shadow_hover": "0 8px 24px rgba(0, 0, 0, 0.55)",
        "hero": "linear-gradient(135deg, #172437 0%, #111317 60%)",
    },
}


def get_mode() -> str:
    """Current mode, "light" or "dark". Light unless the user chose dark before.

    Stored in a plain session value (not the toggle's own state), because Streamlit
    clears widget state when you move to another page.
    """
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark" if st.context.cookies.get(COOKIE) == "dark" else "light"
    return st.session_state.theme_mode


def palette() -> dict[str, Any]:
    """Colours for the current mode."""
    return PALETTES[get_mode()]


def _css(colors: dict[str, Any]) -> str:
    return f"""
<style>
/* One control for light/dark: hide Streamlit's own theme items in the ⋮ menu */
[data-testid^="stMainMenuItem-theme"] {{ display: none !important; }}

.block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1440px; }}
h1 {{ font-weight: 700 !important; letter-spacing: -0.02em; }}
h2, h3 {{ font-weight: 650 !important; letter-spacing: -0.01em; }}

/* Cards: metrics and bordered containers lift slightly on hover */
[data-testid="stMetric"] {{
    background: {colors["card"]};
    box-shadow: {colors["shadow"]};
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
[data-testid="stMetric"]:hover {{ transform: translateY(-2px); box-shadow: {colors["shadow_hover"]}; }}
[data-testid="stMetricValue"] {{ font-weight: 700; letter-spacing: -0.02em; }}
[data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    background: {colors["card"]};
    box-shadow: {colors["shadow"]};
}}

/* Page header */
.loomiq-hero {{
    background: {colors["hero"]};
    border: 1px solid {colors["border"]};
    border-radius: 1rem;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.4rem;
}}
.loomiq-hero h1 {{ margin: 0; padding: 0; font-size: 2rem; color: {colors["text"]}; }}
.loomiq-hero p {{ margin: 0.35rem 0 0 0; color: {colors["text_muted"]}; font-size: 0.98rem; }}
.loomiq-brand {{ font-weight: 700; font-size: 1.25rem; letter-spacing: -0.01em; color: {colors["text"]}; }}
.loomiq-brand span {{ color: {colors["sewing"]}; }}
.loomiq-muted {{ color: {colors["text_muted"]}; font-size: 0.85rem; }}
</style>
"""


def _sync_script(mode: str) -> str:
    """Make Streamlit's own theme match ``mode`` ("Light" or "Dark") and remember it."""
    return f"""
<script>
(() => {{
  const mode = "{mode}";
  document.cookie = "{COOKIE}=" + mode.toLowerCase() + "; path=/; max-age=31536000; SameSite=Lax";
  const storedKey = "stActiveTheme-" + window.location.pathname + "-v2";
  let attempts = 0;
  const sync = () => {{
    if (attempts++ > 60) return;  // give up quietly after ~6 s
    const menuButton = document.querySelector('[data-testid="stMainMenuButton"]');
    if (!menuButton) {{ setTimeout(sync, 100); return; }}
    const item = document.querySelector('[data-testid="stMainMenuItem-theme-' + mode + '"]');
    if (item) {{
      if (item.getAttribute("aria-checked") !== "true") item.click();
      // Close the menu we opened (picking a theme item leaves it open)
      setTimeout(() => {{
        if (document.querySelector('[data-testid^="stMainMenuItem-theme"]')) menuButton.click();
      }}, 120);
      return;
    }}
    if (window.localStorage.getItem(storedKey) === JSON.stringify(mode)) return;  // already right
    menuButton.click();  // open the menu, then pick the item on the next tick
    setTimeout(sync, 150);
  }};
  sync();
}})();
</script>
"""


def apply_theme() -> dict[str, Any]:
    """Apply the styling for the current mode and return its palette.

    Call once at the top of every page. The toggle itself is drawn by :func:`mode_toggle`.
    """
    from loomiq.config import PROJECT_ROOT

    mode = get_mode()
    colors = PALETTES[mode]
    st.logo(str(PROJECT_ROOT / "app" / "assets" / "logo.svg"), size="large")
    st.html(_css(colors))
    st.html(_sync_script(mode.title()), unsafe_allow_javascript=True)
    return colors


def _flip_mode() -> None:
    st.session_state.theme_mode = "dark" if st.session_state.dark_mode_toggle else "light"


def mode_toggle() -> None:
    """The light / dark switch (place it in the sidebar)."""
    st.toggle(
        ":material/dark_mode: Dark mode", value=get_mode() == "dark", key="dark_mode_toggle",
        on_change=_flip_mode, help="Switch between light and dark mode. Your choice is remembered.",
    )


def page_header(title: str, subtitle: str, icon: str = "") -> None:
    """A clean header card at the top of a page."""
    st.html(f'<div class="loomiq-hero"><h1>{icon} {title}</h1><p>{subtitle}</p></div>')
