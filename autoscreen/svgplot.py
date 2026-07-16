"""Minimal dependency-free SVG line-chart writer.

matplotlib's native (Agg) renderer segfaults on some Windows setups, so this
module emits clean SVG directly. Supports multiple series with optional shaded
std bands, axis ticks, gridlines, and a legend.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Series:
    label: str
    xs: list[float]
    ys: list[float]
    color: str
    lo: list[float] = field(default_factory=list)
    hi: list[float] = field(default_factory=list)


PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    import math

    if vmax <= vmin:
        return [vmin]
    raw = (vmax - vmin) / n
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if (vmax - vmin) / step <= n + 1e-9:
            break
    start = math.ceil(vmin / step) * step
    ticks = []
    t = start
    while t <= vmax + step * 1e-6:
        ticks.append(round(t, 6))
        t += step
    return ticks


def line_chart(
    series: list[Series],
    title: str,
    xlabel: str,
    ylabel: str,
    width: int = 720,
    height: int = 500,
) -> str:
    ml, mr, mt, mb = 70, 150, 50, 60
    pw = width - ml - mr
    ph = height - mt - mb

    all_x = [x for s in series for x in s.xs]
    all_y = [y for s in series for y in s.ys]
    all_y += [v for s in series for v in s.lo]
    all_y += [v for s in series for v in s.hi]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    ypad = (ymax - ymin) * 0.05 or 0.01
    ymin -= ypad
    ymax += ypad

    def sx(x: float) -> float:
        return ml + (x - xmin) / (xmax - xmin) * pw if xmax > xmin else ml

    def sy(y: float) -> float:
        return mt + (ymax - y) / (ymax - ymin) * ph if ymax > ymin else mt + ph

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI, Arial, sans-serif">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append(
        f'<text x="{ml + pw / 2}" y="26" text-anchor="middle" '
        f'font-size="17" font-weight="600">{title}</text>'
    )

    # gridlines + y ticks
    for ty in _nice_ticks(ymin, ymax):
        y = sy(ty)
        parts.append(
            f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#444">{ty:g}</text>'
        )
    # x ticks
    for tx in _nice_ticks(xmin, xmax):
        x = sx(tx)
        parts.append(
            f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" '
            f'stroke="#f0f0f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{mt + ph + 20}" text-anchor="middle" '
            f'font-size="12" fill="#444">{int(tx)}</text>'
        )

    # axes
    parts.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#333" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" '
        f'stroke="#333" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{ml + pw / 2}" y="{height - 18}" text-anchor="middle" '
        f'font-size="13">{xlabel}</text>'
    )
    parts.append(
        f'<text x="20" y="{mt + ph / 2}" text-anchor="middle" font-size="13" '
        f'transform="rotate(-90 20 {mt + ph / 2})">{ylabel}</text>'
    )

    for s in series:
        # std band
        if s.lo and s.hi:
            top = " ".join(f"{sx(x):.1f},{sy(h):.1f}" for x, h in zip(s.xs, s.hi))
            bot = " ".join(
                f"{sx(x):.1f},{sy(l):.1f}" for x, l in zip(reversed(s.xs), reversed(s.lo))
            )
            parts.append(
                f'<polygon points="{top} {bot}" fill="{s.color}" fill-opacity="0.12" stroke="none"/>'
            )
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(s.xs, s.ys))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{s.color}" stroke-width="2.2"/>'
        )
        for x, y in zip(s.xs, s.ys):
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{s.color}"/>')

    # legend
    lx = ml + pw + 20
    ly = mt + 10
    for i, s in enumerate(series):
        yy = ly + i * 22
        parts.append(
            f'<line x1="{lx}" y1="{yy}" x2="{lx + 24}" y2="{yy}" '
            f'stroke="{s.color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{lx + 30}" y="{yy + 4}" font-size="13" fill="#222">{s.label}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)
