#!/usr/bin/env python3
"""Generate the animated terminal-style profile banner (dark + light SVGs).

Reads assets/profile-banner/portrait.png (RGBA, transparent background) and
dithers it at native 300x340 panel resolution (Floyd-Steinberg, serpentine)
after histogram-equalization + unsharp-masking, so facial detail survives
against the dark hair/jacket. Most of those dots are static (merged into
compact horizontal-run SVG paths for a small file); a small "traveller"
subset is animated (cx/cy, index-paired via the Hungarian algorithm) so it
periodically morphs into a generic "</>" glyph and back while the static
bulk fades out/in around it. Fully self contained: no JS, no external
resources.

Run with the venv created for this script (has numpy + Pillow + scipy):
    ./.venv-banner/bin/python scripts/banner/generate.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "assets" / "profile-banner"
SOURCE_IMAGE = ASSETS_DIR / "portrait.png"

PANEL_W, PANEL_H = 300, 340
TRAVELLER_COUNT = 800
SEED = 20260922

CANVAS_W = 920
CANVAS_H = 520

INFO_ROWS = [
    ("Subject", "Gaurav"),
    ("Role", "Software Engineer"),
    ("Languages", "Python · Java · C++ · TypeScript"),
    ("Frontend", "React · Next.js · Tailwind"),
    ("Backend", "Node · Express · FastAPI"),
    ("AI", "ML · RAG · LLMs · AI Agents"),
    ("Data", "PostgreSQL · MongoDB · Vector DBs"),
    ("Infra", "Git · Docker · GitHub Actions"),
    ("GitHub", "orewagaurav"),
]

MONO_STACK = "'JetBrains Mono','Fira Code',ui-monospace,Menlo,Consolas,monospace"


@dataclass
class Theme:
    name: str
    bg: str
    panel_bg: str
    border: str
    border_soft: str
    text_primary: str
    text_secondary: str
    text_dim: str
    accent_cyan: str
    dot_color: str
    ok_color: str
    live_color: str
    shadow: str
    select_bright: bool  # dark theme draws dots on the photo's LIT pixels


DARK = Theme(
    name="dark", bg="#060810", panel_bg="#0a0d18", border="#232a3d",
    border_soft="#161b2b", text_primary="#e7e9f5", text_secondary="#8b93b0",
    text_dim="#5a6182", accent_cyan="#39e6e0", dot_color="#b0a0ff",
    ok_color="#4ade80", live_color="#ff4d5a", shadow="#02050b", select_bright=True,
)

LIGHT = Theme(
    name="light", bg="#f8f8fc", panel_bg="#ffffff", border="#dcdce6",
    border_soft="#ececf3", text_primary="#1a1b2c", text_secondary="#565b76",
    text_dim="#9a9fb8", accent_cyan="#0e9aa3", dot_color="#221c3d",
    ok_color="#1f9d55", live_color="#e0293a", shadow="#aab7c4", select_bright=False,
)


# ---------------------------------------------------------------------------
# Photo -> dithered dot grid
# ---------------------------------------------------------------------------

def crop_to_bust(im: Image.Image) -> Image.Image:
    """Head-to-upper-chest framing, then fit to the panel aspect."""
    w, h = im.size
    target_ratio = PANEL_H / PANEL_W
    bust_h = int(h * 0.60)
    sub_w = int(bust_h / target_ratio)
    if sub_w > w:
        sub_w = w
        bust_h = int(sub_w * target_ratio)
    left = (w - sub_w) // 2
    return im.crop((left, 0, left + sub_w, bust_h)).resize((PANEL_W, PANEL_H), Image.LANCZOS)


def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    """Serpentine 1-bit dither; True where a pixel quantizes to white."""
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    h, w = work.shape
    for y in range(h):
        left_to_right = y % 2 == 0
        xs = range(w) if left_to_right else range(w - 1, -1, -1)
        step = 1 if left_to_right else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + step
            if 0 <= nx < w:
                work[y, nx] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - step < w:
                    work[y + 1, x - step] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < w:
                    work[y + 1, nx] += err * 1 / 16
    return out


def dither_portrait(theme: Theme) -> np.ndarray:
    """Return an (N, 2) array of active-dot grid coordinates for one theme."""
    crop = crop_to_bust(Image.open(SOURCE_IMAGE).convert("RGBA"))
    alpha = np.asarray(crop.getchannel("A"), dtype=np.float32) / 255.0

    if theme.select_bright:
        gray = np.asarray(ImageOps.grayscale(crop.convert("RGB")), dtype=np.float32)
        prepared = Image.fromarray(np.uint8(np.clip(gray * alpha, 0, 255)), "L")
        mask = Image.fromarray(np.uint8((alpha > 0.08) * 255), "L")
        prepared = ImageOps.equalize(prepared, mask=mask)
    else:
        white = Image.new("RGBA", crop.size, "white")
        white.alpha_composite(crop)
        prepared = ImageOps.grayscale(white.convert("RGB"))
        prepared = ImageOps.autocontrast(prepared, cutoff=1)

    prepared = ImageEnhance.Contrast(prepared).enhance(1.35)
    prepared = prepared.filter(ImageFilter.UnsharpMask(radius=2, percent=170, threshold=1))

    bits = floyd_steinberg(np.asarray(prepared))
    active = bits if theme.select_bright else ~bits
    if theme.select_bright:
        active &= alpha > 0.08

    ys, xs = np.where(active)
    return np.column_stack((xs, ys)).astype(np.float32)


def code_glyph_points(count: int, seed: int) -> np.ndarray:
    """Uniform-random sample inside a hand-drawn '</>' glyph mask."""
    mask = Image.new("L", (PANEL_W, PANEL_H), 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = PANEL_W / 2, PANEL_H / 2
    stroke = int(PANEL_W * 0.052)
    span_y = PANEL_H * 0.16

    def chevron(vertex_x: float, arm_x: float) -> None:
        draw.line([(arm_x, cy - span_y), (vertex_x, cy), (arm_x, cy + span_y)],
                   fill=255, width=stroke, joint="curve")

    chevron(cx - PANEL_W * 0.32, cx - PANEL_W * 0.17)
    chevron(cx + PANEL_W * 0.32, cx + PANEL_W * 0.17)
    draw.line([(cx - PANEL_W * 0.045, cy + PANEL_H * 0.23), (cx + PANEL_W * 0.045, cy - PANEL_H * 0.23)],
               fill=255, width=int(stroke * 0.85))

    ys, xs = np.where(np.asarray(mask) > 127)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(xs), count, replace=len(xs) < count)
    return np.column_stack((xs[chosen], ys[chosen])).astype(np.float32)


def hungarian_pair(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reorder target to minimize total squared travel distance from source."""
    rows, cols = linear_sum_assignment(cdist(source, target, metric="sqeuclidean"))
    ordered = np.empty_like(target)
    ordered[rows] = target[cols]
    return ordered


# ---------------------------------------------------------------------------
# Compact static-dot path encoding
# ---------------------------------------------------------------------------

def point_path(points: np.ndarray, ox: float, oy: float) -> str:
    """Merge same-row adjacent dots into 'Mx yh<run>' commands."""
    if not len(points):
        return ""
    ordered = sorted({(int(x), int(y)) for x, y in points}, key=lambda p: (p[1], p[0]))
    chunks: list[str] = []
    i = 0
    while i < len(ordered):
        x0, y = ordered[i]
        x1 = x0
        i += 1
        while i < len(ordered) and ordered[i][1] == y and ordered[i][0] <= x1 + 1:
            x1 = ordered[i][0]
            i += 1
        chunks.append(f"M{ox + x0:.1f} {oy + y:.1f}h{x1 - x0 + 1}")
    return "".join(chunks)


# ---------------------------------------------------------------------------
# SVG assembly
# ---------------------------------------------------------------------------

def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text_width(text: str, font_size: float) -> float:
    """Approximate width of monospace text, for textLength + leader placement."""
    return len(text) * font_size * 0.6


def dotted_leader(x1: float, x2: float, y: float, step: float = 5.0) -> str:
    if x2 <= x1:
        return ""
    xs = np.arange(x1, x2, step)
    return "".join(f"M{x:.1f} {y:.1f}h1" for x in xs)


def build_svg(theme: Theme, portrait: np.ndarray, icon: np.ndarray, cycle_seconds: float) -> str:
    left_x, left_y = 18, 60
    portrait_x = left_x + 12
    portrait_y = left_y + 28
    right_x = left_x + 336
    header_h = 40
    footer_y = int(portrait_y + PANEL_H + 46)

    rng = np.random.default_rng(SEED)
    n = min(TRAVELLER_COUNT, len(portrait))
    traveller_idx = rng.choice(len(portrait), n, replace=False)
    traveller_src = portrait[traveller_idx]
    bulk = np.delete(portrait, traveller_idx, axis=0)
    traveller_dst = hungarian_pair(traveller_src, icon[:n])

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
        f'width="{CANVAS_W}" height="{CANVAS_H}" role="img" '
        f'aria-label="Gaurav Kumar developer profile, terminal-style animated banner">'
        f'<title>profile.sh --live — Gaurav Kumar</title>'
        f'<style>text{{font-family:{MONO_STACK};}}</style>'
        '<defs>'
        f'<filter id="shadow-{theme.name}" x="-20%" y="-20%" width="140%" height="150%">'
        f'<feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="{theme.shadow}" flood-opacity=".3"/>'
        '</filter>'
        f'<filter id="glow-{theme.name}" x="-100%" y="-100%" width="300%" height="300%">'
        '<feGaussianBlur stdDeviation="2.5" result="b"/>'
        f'<feFlood flood-color="{theme.live_color}" flood-opacity=".45"/>'
        '<feComposite in2="b" operator="in"/>'
        '<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>'
        '</filter>'
        '</defs>'
    )
    parts.append(
        f'<rect x="1" y="1" width="{CANVAS_W - 2}" height="{CANVAS_H - 2}" rx="10" '
        f'fill="{theme.bg}" stroke="{theme.border}" stroke-width="1.5" filter="url(#shadow-{theme.name})"/>'
    )

    parts.append(f'<line x1="0" y1="{header_h}" x2="{CANVAS_W}" y2="{header_h}" stroke="{theme.border_soft}"/>')
    for i, color in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        parts.append(f'<circle cx="{24 + i * 18}" cy="{header_h / 2}" r="5.5" fill="{color}"/>')
    parts.append(f'<text x="88" y="{header_h / 2 + 5}" font-size="13" fill="{theme.text_secondary}">profile.sh --live</text>')
    parts.append(
        f'<text x="{CANVAS_W - 20}" y="{header_h / 2 + 5}" font-size="12" fill="{theme.text_dim}" text-anchor="end">@orewagaurav</text>'
    )
    parts.append(
        f'<g filter="url(#glow-{theme.name})"><circle cx="{CANVAS_W - 190}" cy="{header_h / 2 + 1}" r="3.5" fill="{theme.live_color}">'
        f'<animate attributeName="opacity" values="1;.35;1" dur="1.6s" repeatCount="indefinite"/>'
        f'</circle></g>'
        f'<text x="{CANVAS_W - 180}" y="{header_h / 2 + 5}" font-size="11.5" fill="{theme.live_color}" '
        f'font-weight="700" letter-spacing="1">LIVE</text>'
    )

    parts.append(
        f'<line x1="{right_x - 18}" y1="{header_h + 1}" x2="{right_x - 18}" y2="{footer_y}" stroke="{theme.border_soft}"/>'
    )
    parts.append(f'<line x1="0" y1="{footer_y}" x2="{CANVAS_W}" y2="{footer_y}" stroke="{theme.border_soft}"/>')
    parts.append(f'<text x="{left_x}" y="{left_y + 12}" font-size="11" letter-spacing="1.5" fill="{theme.accent_cyan}">VISUAL.MAP</text>')

    clip_id = f"portraitClip-{theme.name}"
    parts.append(
        f'<clipPath id="{clip_id}"><rect x="{portrait_x}" y="{portrait_y}" width="{PANEL_W}" height="{PANEL_H}" rx="4"/></clipPath>'
    )
    parts.append(
        f'<rect x="{portrait_x}" y="{portrait_y}" width="{PANEL_W}" height="{PANEL_H}" rx="4" '
        f'fill="{theme.panel_bg}" stroke="{theme.border}" stroke-width="1"/>'
    )

    # Timeline (fractions of the cycle): hold portrait, morph out, hold icon, morph back.
    kt = [0.0, 0.34, 0.40, 0.66, 0.72, 1.0]
    key_times = ";".join(str(t) for t in kt)
    bulk_opacity = "1;1;0;0;1;1"

    parts.append(f'<g clip-path="url(#{clip_id})" shape-rendering="crispEdges">')

    # Static bulk layer: everything except the animated traveller subset.
    parts.append(
        f'<path d="{point_path(bulk, portrait_x, portrait_y)}" fill="none" '
        f'stroke="{theme.dot_color}" stroke-width="1">'
        f'<animate attributeName="opacity" keyTimes="{key_times}" values="{bulk_opacity}" '
        f'dur="{cycle_seconds}s" repeatCount="indefinite"/>'
        f'</path>'
    )

    # Animated traveller layer: morphs into the glyph and back.
    parts.append(f'<g fill="{theme.dot_color}">')
    for (px, py), (ix, iy) in zip(traveller_src, traveller_dst):
        cx0, cy0 = portrait_x + px + 0.5, portrait_y + py + 0.5
        cx1, cy1 = portrait_x + ix + 0.5, portrait_y + iy + 0.5
        vx = f"{cx0:.1f};{cx0:.1f};{cx1:.1f};{cx1:.1f};{cx0:.1f};{cx0:.1f}"
        vy = f"{cy0:.1f};{cy0:.1f};{cy1:.1f};{cy1:.1f};{cy0:.1f};{cy0:.1f}"
        parts.append(
            f'<circle cx="{cx0:.1f}" cy="{cy0:.1f}" r="1.1">'
            f'<animate attributeName="cx" keyTimes="{key_times}" values="{vx}" dur="{cycle_seconds}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" keyTimes="{key_times}" values="{vy}" dur="{cycle_seconds}s" repeatCount="indefinite"/>'
            f'</circle>'
        )
    parts.append('</g>')
    parts.append('</g>')

    bracket = 14
    for cx, cy, hx, hy in [
        (portrait_x, portrait_y, 1, 1),
        (portrait_x + PANEL_W, portrait_y, -1, 1),
        (portrait_x, portrait_y + PANEL_H, 1, -1),
        (portrait_x + PANEL_W, portrait_y + PANEL_H, -1, -1),
    ]:
        parts.append(
            f'<path d="M{cx + hx * bracket:.1f} {cy:.1f} L{cx:.1f} {cy:.1f} L{cx:.1f} {cy + hy * bracket:.1f}" '
            f'fill="none" stroke="{theme.accent_cyan}" stroke-width="1.6"/>'
        )

    caption_y = portrait_y + PANEL_H + 18
    parts.append(
        f'<text x="{portrait_x + PANEL_W / 2:.1f}" y="{caption_y}" font-size="10.5" text-anchor="middle" '
        f'fill="{theme.text_dim}" letter-spacing="1">300&#215;340 / 1-BIT</text>'
    )
    parts.append(
        f'<text x="{left_x}" y="{footer_y - 12}" font-size="10" fill="{theme.text_dim}">'
        f'PTS {len(portrait)} &#183; FS/SERPENTINE</text>'
    )

    parts.append(f'<text x="{right_x}" y="{left_y + 12}" font-size="11" letter-spacing="1.5" fill="{theme.accent_cyan}">SYSTEM.INFO</text>')
    row_y = left_y + 46
    row_gap = (portrait_y + PANEL_H - row_y) / (len(INFO_ROWS) - 1)
    value_right = CANVAS_W - 18
    font_size = 13
    for label, value in INFO_ROWS:
        value_w = text_width(value, font_size)
        label_w = text_width(label, font_size)
        leader_start = right_x + label_w + 10
        leader_end = value_right - value_w - 10
        parts.append(
            f'<text x="{right_x}" y="{row_y}" font-size="{font_size}" fill="{theme.text_secondary}">{esc(label)}</text>'
        )
        parts.append(
            f'<path d="{dotted_leader(leader_start, leader_end, row_y - 4)}" '
            f'fill="none" stroke="{theme.border}" stroke-width="1" shape-rendering="crispEdges"/>'
        )
        parts.append(
            f'<text x="{value_right}" y="{row_y}" text-anchor="end" fill="{theme.text_primary}" '
            f'font-size="{font_size}" textLength="{value_w:.1f}" lengthAdjust="spacingAndGlyphs">{esc(value)}</text>'
        )
        row_y += row_gap

    parts.append(
        f'<circle cx="20" cy="{footer_y + 17}" r="3.5" fill="{theme.ok_color}"/>'
        f'<text x="30" y="{footer_y + 21}" font-size="11" fill="{theme.text_secondary}" letter-spacing="0.5">ALL SYSTEMS NOMINAL</text>'
    )
    parts.append(
        f'<text x="{CANVAS_W - 20}" y="{footer_y + 21}" font-size="11" fill="{theme.text_dim}" text-anchor="end" letter-spacing="1">INDIA NODE</text>'
    )
    parts.append('</svg>')
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-seconds", type=float, default=15.0)
    args = parser.parse_args()

    if not SOURCE_IMAGE.exists():
        raise SystemExit(f"Source photo not found: {SOURCE_IMAGE}")

    for theme in (DARK, LIGHT):
        print(f"dithering {theme.name}...")
        portrait = dither_portrait(theme)
        icon = code_glyph_points(max(TRAVELLER_COUNT, 1), seed=SEED + 1)
        print(f"  {len(portrait)} portrait dots, {min(TRAVELLER_COUNT, len(portrait))} travellers")

        svg = build_svg(theme, portrait, icon, args.cycle_seconds)
        out_path = ASSETS_DIR / f"banner-{theme.name}.svg"
        out_path.write_text(svg, encoding="utf-8")
        print(f"wrote {out_path} ({len(svg) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
