# The Verstappen Decade

A data-driven story built from the Ergast Formula 1 database: twelve seasons of
Max Verstappen, 2015–2026, rendered as a single self-contained HTML infographic.

**The thesis:** the wins collapsed, but the driver metrics didn't. The
teammate-qualifying head-to-head — the closest thing motorsport has to a
controlled experiment — holds at 82–100% straight through the decline.

## Getting the data

The raw CSVs are not committed. Download the dataset from Kaggle and unzip it
into `data/raw/`:

    https://www.kaggle.com/datasets/jtrotman/formula-1-race-data

`data/raw/` should end up containing `races.csv`, `results.csv`, `lap_times.csv`,
`qualifying.csv`, `pit_stops.csv`, `drivers.csv`, `constructors.csv`,
`status.csv` and the rest of the Ergast tables.

## Build

    pip install -r requirements.txt
    python3 pipeline/build.py

This reads `data/raw/`, computes the season metrics, writes `site/data.json`,
and injects the payload into `site/template.html` to produce a standalone
`site/index.html` with no runtime data fetch.

To preview:

    python3 -m http.server 8899 --directory site

## Layout

| Path | Purpose |
|---|---|
| `pipeline/build.py` | CSVs → metrics → `site/index.html` |
| `site/template.html` | Page markup, styles, and the SVG chart code |
| `site/data.json` | Generated payload (also inlined into the built page) |

Edit `site/template.html`, never `site/index.html` — the latter is generated and
is overwritten on every build.

## Method notes

**Classifying retirements.** Ergast marks a retirement with
`positionText = "R"`. In this snapshot the 2026 rows *also* carry a numeric
`position` for those retirements, while earlier seasons leave it null.
Classifying on `position` scores a lap-zero engine failure as "finished 22nd":
2026's net position change swings from −34 to +9 depending on which column you
trust. Everything here classifies on `positionText`.

**Overtakes.** There is no overtake column in this data. Places gained and lost
are derived from lap-by-lap classified position, dropping each pit in-lap and
the following out-lap so a pit stop is never counted as a pass. It remains a
proxy — it cannot separate a pass on track from a rival's slow lap.

**Laps led.** A lead lap is one lap of one race with a car classified first on
that lap. The season denominator is every such lap that year, so "share of laps
led" is comparable across seasons of different length.

**Qualifying gap in seconds.** Each driver's best lap of the weekend is parsed
from the `q1`/`q2`/`q3` time strings; Verstappen's is differenced against the
sister car's, and the season figure is the median over weekends where both set a
comparable time. Negative means Verstappen was quicker on aggregate. It is
coarser than the session head-to-head (tyre/fuel state, traffic, deleted laps)
but the direction is clear — the margin widens as the wins fall away.

**The 2025 split.** 2025 ran 24 rounds; the halves are rounds 1–12 and 13–24,
points include sprint points. The standout-drives chart records the 2025 São
Paulo start as P19 (Ergast); Verstappen was knocked out in Q1 and started from
the pit lane, so that grid-to-finish gain is a lower bound. All eight comeback
drives were checked grid-and-finish against the race reports.

**No telemetry.** This dataset carries timing and classification only — no
throttle, brake or speed traces.

## Charts

Colours follow a CVD-validated two-hue palette (orange `#d9540a`/`#d95926` for
Verstappen, blue `#2a78d6`/`#3987e5` for the field), validated against both the
light and dark chart surfaces. SVG fills are set through CSS custom properties
rather than baked hex, so the charts follow a live theme change.

## Source

Ergast Developer API database, via the Kaggle mirror
`jtrotman/formula-1-race-data`. Results complete through 2026-08-23.
