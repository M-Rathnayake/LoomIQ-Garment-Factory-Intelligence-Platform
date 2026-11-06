"""Plotly charts for the dashboard, styled for the active light or dark mode.

Every function takes ``colors`` (from :func:`loomiq.dashboard.theme.palette`).
Backgrounds are transparent so charts sit on the page's cards in either mode.
Status colours (green / amber / red) always appear with an icon and a text label,
so meaning never depends on colour alone.
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from loomiq.dashboard.live import STATUS_AT_RISK, STATUS_BEHIND, STATUS_ON_TRACK


def _light() -> dict[str, Any]:
    from loomiq.dashboard.theme import PALETTES

    return PALETTES["light"]


def status_fill(colors: dict[str, Any]) -> dict[str, str]:
    return {STATUS_ON_TRACK: colors["good"], STATUS_AT_RISK: colors["warning"], STATUS_BEHIND: colors["critical"]}


def _style(fig: go.Figure, title: str, height: int, colors: dict[str, Any]) -> go.Figure:
    """Shared look: transparent background, recessive grid, left-aligned title."""
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15, "color": colors["text"]}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": colors["text_muted"], "size": 13, "family": "Inter, sans-serif"},
        height=height,
        margin={"l": 10, "r": 20, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1,
                "font": {"color": colors["text"]}},
        hoverlabel={"bgcolor": colors["card"], "bordercolor": colors["border"], "font": {"color": colors["text"]}},
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=colors["grid"], zeroline=False, linecolor=colors["grid"],
                     tickfont={"color": colors["text_muted"]}, automargin=True)
    fig.update_yaxes(gridcolor=colors["grid"], zeroline=False, linecolor=colors["grid"],
                     tickfont={"color": colors["text"]}, automargin=True, ticksuffix="  ")
    return fig


def line_overview(view: pd.DataFrame, colors: dict[str, Any] | None = None) -> go.Figure:
    """Projected end-of-day output as % of target, one bar per line. Click a bar to open the line."""
    colors = colors or _light()
    data = view.iloc[::-1]  # first line at the top
    share = data["projected_share"] * 100
    fig = go.Figure(
        go.Bar(
            x=share,
            y=data["line_id"],
            orientation="h",
            marker={"color": data["status"].map(status_fill(colors)), "cornerradius": 4},
            text=[f"{s:.0f}%  {label}" for s, label in zip(share, data["status_label"])],
            textposition="outside",
            textfont={"color": colors["text"]},
            cliponaxis=False,
            customdata=data[["pieces_so_far", "projected_pieces", "target_pieces", "risk_probability"]],
            hovertemplate=(
                "<b>%{y}</b><br>Projected: %{x:.0f}% of target"
                "<br>Pieces so far: %{customdata[0]:,.0f}"
                "<br>Projected: %{customdata[1]:,.0f} / target %{customdata[2]:,.0f}"
                "<br>Model risk score: %{customdata[3]:.2f}<br><i>Click to open this line</i><extra></extra>"
            ),
        )
    )
    fig.add_vline(x=100, line={"color": colors["text_muted"], "dash": "dash", "width": 1})
    fig.add_annotation(x=100, y=1.02, yref="paper", text="target", showarrow=False,
                       font={"size": 11, "color": colors["text_muted"]})
    fig.update_xaxes(range=[0, max(150, share.max() + 45)], ticksuffix="%")  # room for the status labels
    fig.update_yaxes(showgrid=False)
    fig.update_layout(clickmode="event+select", dragmode=False)
    return _style(fig, "Projected end-of-day output (% of target)", 70 + 34 * len(view), colors)


def line_progress(
    history: pd.DataFrame, line: pd.Series, hours_per_shift: int, colors: dict[str, Any] | None = None
) -> go.Figure:
    """Cumulative pieces vs target through the day, with the projection to the end."""
    colors = colors or _light()
    color = colors.get(str(line["department"]), colors["positive"])
    hours = [0] + history["hour"].tolist()
    pieces = [0] + history["pieces_so_far"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, hours_per_shift], y=[0, line["target_pieces"]], name="Target pace", mode="lines",
        line={"color": colors["neutral"], "dash": "dash", "width": 2},
        hovertemplate="Target pace<br>Hour %{x}: %{y:,.0f} pieces<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=pieces, name="Actual", mode="lines+markers", fill="tozeroy",
        fillcolor=_fade(color, 0.12), line={"color": color, "width": 2.5},
        marker={"size": 8, "line": {"color": colors["card"], "width": 2}},
        hovertemplate="Actual<br>Hour %{x}: %{y:,.0f} pieces<extra></extra>",
    ))
    if hours[-1] < hours_per_shift:
        fig.add_trace(go.Scatter(
            x=[hours[-1], hours_per_shift], y=[pieces[-1], line["projected_pieces"]], name="Projection",
            mode="lines", line={"color": color, "dash": "dot", "width": 2},
            hovertemplate="Projected end of day: %{y:,.0f} pieces<extra></extra>",
        ))
    fig.update_xaxes(title="hour of shift", range=[0, hours_per_shift + 0.2], dtick=1)
    fig.update_yaxes(title="pieces (cumulative)", tickfont={"color": colors["text_muted"]})
    return _style(fig, f"{line['line_id']}: output through the day", 380, colors)


def drivers_chart(drivers: list[dict], colors: dict[str, Any] | None = None) -> go.Figure:
    """SHAP drivers for one line: red bars raise the risk of missing, blue bars lower it."""
    colors = colors or _light()
    data = pd.DataFrame(drivers).iloc[::-1]
    labels = [f"{row.label} = {row.value}" for row in data.itertuples()]
    fig = go.Figure(go.Bar(
        x=data["shap_value"], y=labels, orientation="h",
        marker={"color": [colors["negative"] if v > 0 else colors["positive"] for v in data["shap_value"]], "cornerradius": 4},
        text=data["direction"], textposition="outside", textfont={"color": colors["text_muted"]}, cliponaxis=False,
        hovertemplate="%{y}<br>%{text}<br>SHAP: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line={"color": colors["text_muted"], "width": 1})
    span = data["shap_value"].abs().max() * 1.7 or 1
    fig.update_xaxes(range=[-span, span], title="← lowers risk   |   raises risk →")
    fig.update_yaxes(showgrid=False)
    return _style(fig, "Why the model rates this line's risk this way", 90 + 45 * len(data), colors)


def plan_chart(table: pd.DataFrame, colors: dict[str, Any] | None = None) -> go.Figure:
    """Pieces short per at-risk line, before and after the recommended plan."""
    colors = colors or _light()
    data = table[table["shortfall_pieces"] > 0].sort_values("shortfall_pieces")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data["shortfall_pieces"], y=data["line_id"], orientation="h", name="Before",
        marker={"color": colors["neutral"], "cornerradius": 4},
        hovertemplate="%{y}<br>Before: %{x:,.0f} short<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=data["shortfall_pieces_after"], y=data["line_id"], orientation="h", name="After the plan",
        marker={"color": colors["positive"], "cornerradius": 4},
        text=["fixed ✓" if v < 0.5 else "" for v in data["shortfall_pieces_after"]],
        textposition="outside", textfont={"color": colors["text_muted"]},
        hovertemplate="%{y}<br>After: %{x:,.0f} short<extra></extra>",
    ))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="expected pieces short of target")
    fig.update_yaxes(showgrid=False)
    return _style(fig, "Expected shortfall: before vs after the plan", 120 + 55 * max(len(data), 1), colors)


def _fade(hex_color: str, alpha: float) -> str:
    """Hex colour -> rgba with transparency, for area fills."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"
