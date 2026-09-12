"""Render the architecture diagram used in the README.

The counts in the diagram are read from the panel rather than typed in, so the
picture cannot quietly describe a system that no longer exists. Output is SVG:
it stays crisp in a README at any width, and it carries its own dark ground so
it reads the same in either GitHub theme.
"""

import argparse
import pathlib
import shutil
import subprocess
import tempfile
from xml.sax.saxutils import escape

import pandas as pd

from f1pred.cli import _load_panel
from f1pred.config import DEFAULT_MODEL_CONFIG, IMAGES_DIR, RAW_DIR, ensure_dirs
from f1pred.sim.race import SimulationSettings

GROUND = "#0e1117"
PANEL = "#171c26"
PANEL_2 = "#1d2430"
LINE = "#2b3342"
INK = "#e9ecf3"
INK2 = "#aab4c4"
MUTED = "#79839a"
ACCENT = "#ff5a37"
GATE = "#e0a63a"
DEAD = "#5b6577"

SANS = "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

W, H = 1240, 1010


def _box(x, y, w, h, fill=PANEL, stroke=LINE, rx=3, width=1):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def _text(x, y, s, size=13, fill=INK, family=SANS, weight=400, anchor="start", spacing=0):
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>{escape(s)}</text>'
    )


def _arrow(x, y1, y2, colour=LINE):
    return (
        f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 7}" stroke="{colour}" '
        f'stroke-width="1.5"/><path d="M{x - 4},{y2 - 7} L{x + 4},{y2 - 7} L{x},{y2} Z" '
        f'fill="{colour}"/>'
    )


#: Where a headless browser usually lives. SVG is the source of truth; the PNG
#: exists because social platforms will not accept vector files.
BROWSERS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
)


def _browser() -> str | None:
    for candidate in BROWSERS:
        if pathlib.Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def rasterise(svg: pathlib.Path, png: pathlib.Path, *, scale: int = 2) -> bool:
    """Screenshot the SVG through a headless browser at ``scale``x.

    Rendering the real SVG rather than redrawing the diagram means the PNG can
    never disagree with the vector version; there is only one implementation.
    """
    browser = _browser()
    if browser is None:
        print("no headless browser found; skipping PNG")
        return False
    width, height = W, H
    with tempfile.TemporaryDirectory() as work:
        page = pathlib.Path(work) / "page.html"
        page.write_text(
            '<!doctype html><meta charset="utf-8">'
            f"<style>html,body{{margin:0;padding:0;background:{GROUND}}}"
            "svg{display:block}</style>\n" + svg.read_text()
        )
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--force-device-scale-factor={scale}",
                f"--window-size={width},{height}",
                f"--screenshot={png}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    return png.exists()


def build(counts: dict) -> str:
    cfg = DEFAULT_MODEL_CONFIG
    sim = SimulationSettings()
    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="How the forecast is built">',
        f'<rect width="{W}" height="{H}" fill="{GROUND}"/>',
    ]
    L, R = 40, W - 40

    # ---- header ---------------------------------------------------------
    p.append(_text(L, 46, "HOW THE FORECAST IS BUILT", 25, INK, SANS, 700, spacing=1.2))
    p.append(
        _text(
            L,
            70,
            "Every fitting function takes an explicit as-of index, so the live forecast "
            "and the backtest run through identical code.",
            13.5,
            MUTED,
        )
    )
    p.append(f'<line x1="{L}" y1="88" x2="{R}" y2="88" stroke="{INK}" stroke-width="2"/>')

    def band(y, label):
        p.append(_text(L, y, label, 10.5, ACCENT, MONO, 500, spacing=2.2))

    # ---- 1. sources -----------------------------------------------------
    band(120, "01  SOURCES")
    sources = [
        (
            "Jolpica",
            [
                "results, qualifying",
                "driver standings",
                f"{counts['with_results']} races with results, 2014-2026",
            ],
        ),
        ("OpenF1 + F1 timing", ["practice laps", "on-track passes", "pit-lane times"]),
        ("Reference", ["circuit specifications", "for venues with no", "racing history"]),
    ]
    bw = (R - L - 2 * 18) / 3
    for i, (title, lines) in enumerate(sources):
        x = L + i * (bw + 18)
        p.append(_box(x, 134, bw, 86))
        p.append(_text(x + 16, 158, title, 14.5, INK, SANS, 600))
        for j, line in enumerate(lines):
            p.append(_text(x + 16, 178 + j * 16, line, 11.5, MUTED, MONO))
    p.append(_arrow(W / 2, 220, 244))

    # ---- 2. panel -------------------------------------------------------
    band(262, "02  PANEL")
    p.append(_box(L, 276, R - L, 74, PANEL_2))
    p.append(_text(L + 18, 302, "One chronological panel", 15, INK, SANS, 600))
    p.append(
        _text(
            L + 18,
            322,
            "A single authoritative event index on every table, plus when each result "
            "became available (assumed: race day + 1).",
            12.5,
            INK2,
        )
    )
    p.append(
        _text(
            R - 18,
            302,
            f"{counts['observations']:,} observations",
            13,
            INK,
            MONO,
            500,
            anchor="end",
        )
    )
    p.append(
        _text(
            R - 18,
            322,
            f"{counts['qual']:,} qualifying · {counts['race']:,} race",
            11.5,
            MUTED,
            MONO,
            anchor="end",
        )
    )
    p.append(_arrow(W / 2, 350, 374))

    # ---- 3. leakage gates ----------------------------------------------
    p.append(_box(L, 374, R - L, 52, PANEL, GATE))
    p.append(_text(L + 18, 398, "LEAKAGE GATES", 11, GATE, MONO, 600, spacing=1.8))
    p.append(
        _text(
            L + 18,
            416,
            "Fails closed: an unregistered feature family is illegal by default.",
            11.5,
            MUTED,
        )
    )
    gates = "stage gate  ·  cutoff timestamp  ·  historical features use past races only"
    p.append(_text(R - 18, 404, gates, 11.5, INK2, MONO, anchor="end"))
    p.append(_arrow(W / 2, 426, 452))

    # ---- 4. fitted models ----------------------------------------------
    band(470, "03  MODELS FITTED ONLY ON INFORMATION AVAILABLE BEFORE THE TARGET RACE")
    models = [
        (
            "Strength",
            "driver + team strength estimated separately",
            [
                "team effect + driver effect",
                f"recent races count more ({cfg.recency_half_life_races:g}-race half-life)",
                "older rule eras count less",
                "retirements excluded",
            ],
        ),
        (
            "Reliability",
            "historical retirement / reliability model",
            [
                "team hazard (shared car)",
                "driver excess vs. team-mate",
                "circuit attrition multiplier",
                "modelled separately from pace",
            ],
        ),
        (
            "Circuit",
            "measured, then transferred",
            [
                "overtaking difficulty",
                "retirement multiplier",
                "new venues inherit from",
                "nearest speed-profile analogues",
            ],
        ),
    ]
    for i, (title, sub, lines) in enumerate(models):
        x = L + i * (bw + 18)
        p.append(_box(x, 484, bw, 122))
        p.append(f'<rect x="{x}" y="484" width="3" height="122" fill="{ACCENT}"/>')
        p.append(_text(x + 16, 508, title, 15, INK, SANS, 600))
        p.append(_text(x + 16, 525, sub, 11, MUTED, MONO))
        for j, line in enumerate(lines):
            p.append(_text(x + 16, 548 + j * 15, line, 11.5, INK2))
    p.append(_arrow(W / 2, 606, 632))

    # ---- 5. simulator ---------------------------------------------------
    band(650, "04  SIMULATOR")
    p.append(_box(L, 664, R - L, 128, PANEL_2))
    p.append(
        _text(
            L + 18,
            690,
            "100,000 Monte Carlo races",
            15,
            INK,
            SANS,
            600,
        )
    )
    p.append(
        _text(
            R - 18,
            690,
            f"seed {cfg.seed}",
            12,
            MUTED,
            MONO,
            anchor="end",
        )
    )
    steps = [
        (
            "Simulate starting grid",
            f"qualifying is unknown, so it is\nsampled: σ = {sim.qualifying_noise}",
        ),
        ("Draw race pace", f"independent of the grid draw,\nσ = {sim.race_pace_noise}"),
        (
            "Combine grid + race pace",
            f"track position vs. pace, plus\n{sim.start_swing_slots}/{sim.strategy_swing_slots} slot swings",
        ),
        (
            "Simulate retirements & shared incidents",
            f"{sim.chaos_probability:.0%} of races chaotic at ×{sim.chaos_dnf_multiplier},\nmarginals preserved",
        ),
    ]
    sw = (R - L - 36 - 3 * 14) / 4
    for i, (title, detail) in enumerate(steps):
        x = L + 18 + i * (sw + 14)
        p.append(_box(x, 704, sw, 72, PANEL, LINE))
        p.append(_text(x + 12, 724, f"{i + 1}", 10.5, ACCENT, MONO, 600))
        # 11.5 keeps the longest step title ("Simulate retirements & shared
        # incidents") inside its box; the four stay consistent at one size.
        p.append(_text(x + 28, 724, title, 11.5, INK, SANS, 600))
        for j, line in enumerate(detail.split("\n")):
            p.append(_text(x + 12, 744 + j * 14, line, 10.5, MUTED, MONO))
    p.append(_arrow(W / 2, 792, 816))

    # ---- 6. output ------------------------------------------------------
    band(834, "05  PUBLISHED")
    p.append(_box(L, 848, R - L, 56, PANEL, ACCENT))
    outputs = [
        "win",
        "podium",
        "top 10",
        "retirement",
        "expected finish",
        "P10-P90 band",
    ]
    ox = L + 18
    for o in outputs:
        p.append(_text(ox, 882, o, 13, INK, SANS, 500))
        ox += len(o) * 7.6 + 34
    p.append(
        _text(
            R - 18,
            882,
            "probabilities come directly from simulation outcomes",
            11.5,
            MUTED,
            MONO,
            anchor="end",
        )
    )

    # ---- evaluation rail ------------------------------------------------
    rail = R + 4
    p.append(
        f'<path d="M{rail - 6},876 L{rail + 14},876 L{rail + 14},498 L{rail - 6},498" '
        f'fill="none" stroke="{LINE}" stroke-width="1.5"/>'
        f'<path d="M{rail - 6},494 L{rail - 6},502 L{rail - 14},498 Z" fill="{LINE}"/>'
    )
    p.append(
        f'<text transform="translate({rail + 30},687) rotate(-90)" text-anchor="middle" '
        f'font-family="{MONO}" font-size="10.5" fill="{MUTED}" letter-spacing="1.6">'
        f"RACE-BY-RACE TESTING  ·  ADOPTION GATE</text>"
    )

    # ---- rejected --------------------------------------------------------
    p.append(f'<line x1="{L}" y1="926" x2="{R}" y2="926" stroke="{LINE}" stroke-width="1"/>')
    p.append(_text(L, 946, "INGESTED, MEASURED, NOT USED", 10.5, DEAD, MONO, 500, spacing=2))
    dead = [
        (f"{counts['practice']:,} practice laps", "no validated incremental lift yet"),
        (f"{counts['passes']:,} counted passes", "inert at this stage"),
        ("track-car compatibility", "rejected"),
        ("corner telemetry", "never backtested"),
    ]
    cw = (R - L) / 4
    for i, (what, verdict) in enumerate(dead):
        x = L + i * cw
        p.append(_text(x, 968, what, 11.5, INK2, SANS, 500))
        p.append(_text(x, 984, verdict, 11, DEAD, MONO))

    p.append("</svg>")
    return "\n".join(p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="architecture.svg")
    parser.add_argument(
        "--png",
        action="store_true",
        help="also write a 2x PNG, for platforms that will not take SVG",
    )
    args = parser.parse_args()
    ensure_dirs()

    panel = _load_panel()
    kinds = panel.observations["kind"].value_counts()
    practice = RAW_DIR / "practice.parquet"
    counts = {
        "races": int(panel.event_order.shape[0]),
        "with_results": int(panel.results["race_id"].nunique()),
        "observations": int(len(panel.observations)),
        "qual": int(kinds.get("qualifying", 0)),
        "race": int(kinds.get("race", 0)),
        "practice": int(len(pd.read_parquet(practice))) if practice.exists() else 0,
        "passes": int(len(panel.overtakes)),
    }
    out = IMAGES_DIR / args.out
    out.write_text(build(counts))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
    if args.png:
        png = out.with_suffix(".png")
        if rasterise(out, png):
            print(f"wrote {png}  ({png.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
