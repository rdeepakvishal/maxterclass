"""Build the Verstappen-decade infographic from the Ergast/Kaggle F1 CSVs.

Reads data/raw/*.csv, computes the season metrics the story needs, and injects
them into site/template.html to produce a single self-contained site/index.html.

Run:  python3 pipeline/build.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

VER = 830                     # Max Verstappen's Ergast driverId
START_YEAR = 2015             # his debut season
ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SITE = ROOT / "site"

# Ergast encodes nulls as the literal two-character string \N.
NA = ["\\N"]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv", na_values=NA)


def classified(results: pd.DataFrame) -> pd.Series:
    """True when the driver was classified with a finishing position.

    IMPORTANT: classify on positionText, not position. In the 2026 rows of this
    dataset `position` is populated even for retirements (positionText 'R'),
    while pre-2026 rows correctly leave it null. Using `position` would score a
    lap-0 engine failure as "finished 22nd" and corrupt every position-gain
    metric downstream.
    """
    return results.positionText.astype(str).str.isdigit()


def main() -> int:
    results = read("results")
    races = read("races")
    drivers = read("drivers")
    status = read("status")
    quali = read("qualifying")
    constructors = read("constructors")
    lap_times = read("lap_times")
    sprint = read("sprint_results")
    pit_stops = read("pit_stops")

    sprint_pts = (
        sprint.merge(races[["raceId", "year"]], on="raceId")
        .query("driverId == @VER")
        .groupby("year")
        .points.sum()
    )

    name_of = drivers.set_index("driverId").apply(
        lambda r: f"{r.forename} {r.surname}", axis=1
    )

    race_cols = ["raceId", "year", "round", "name", "date"]
    res = (
        results.merge(races[race_cols], on="raceId")
        .merge(status, on="statusId")
        .merge(
            constructors[["constructorId", "name"]],
            on="constructorId",
            suffixes=("_race", "_team"),
        )
    )
    res["classified"] = classified(res)
    res["fin"] = np.where(
        res.classified, pd.to_numeric(res.positionText, errors="coerce"), np.nan
    )
    res["grid"] = pd.to_numeric(res.grid, errors="coerce").replace(0, np.nan)
    res["gain"] = np.where(res.classified, res.grid - res.fin, np.nan)
    res["fastest_lap"] = pd.to_numeric(res["rank"], errors="coerce") == 1

    # Only seasons that actually have results (the file carries a future calendar).
    scored_years = sorted(res.loc[res.year >= START_YEAR, "year"].unique())

    ver = res[res.driverId == VER].copy()

    # ---- lap-by-lap: laps led, and on-track places gained/lost -------------
    laps = lap_times.merge(races[["raceId", "year"]], on="raceId")
    laps = laps[laps.year >= START_YEAR].sort_values(["raceId", "driverId", "lap"])

    led = laps[laps.position == 1].groupby(["year", "driverId"]).size().rename("laps_led")
    season_lead_laps = laps[laps.position == 1].groupby("year").size()

    # A pit stop swaps track position without an overtake, so drop the in-lap
    # and the following out-lap before counting position changes.
    pit = pit_stops[["raceId", "driverId", "lap"]].drop_duplicates().assign(pit_in=1)
    f = laps.merge(pit, on=["raceId", "driverId", "lap"], how="left")
    f["pit_in"] = f.pit_in.fillna(0)
    grp = f.groupby(["raceId", "driverId"])
    f["pit_out"] = grp.pit_in.shift(1).fillna(0)
    f["prev"] = grp.position.shift(1)
    clean = f[(f.pit_in == 0) & (f.pit_out == 0) & f.prev.notna()].copy()
    clean["delta"] = clean.prev - clean.position
    clean["gained"] = clean.delta.clip(lower=0)
    clean["lost"] = (-clean.delta).clip(lower=0)
    moves = clean.groupby(["year", "driverId"])[["gained", "lost"]].sum()

    # ---- qualifying: poles and the teammate head-to-head -------------------
    q = quali.merge(races[["raceId", "year"]], on="raceId")
    vq = q[q.driverId == VER][["raceId", "year", "constructorId", "position"]].rename(
        columns={"position": "ver_pos"}
    )
    mate = q.merge(vq, on=["raceId", "year", "constructorId"])
    mate = mate[mate.driverId != VER].copy()
    mate["ver_ahead"] = mate.position > mate.ver_pos
    h2h = mate.groupby("year").agg(
        n=("ver_ahead", "size"),
        won=("ver_ahead", "sum"),
        mates=("driverId", lambda s: sorted({name_of[i] for i in s})),
    )
    # Public records count a pole as starting P1, not as setting the fastest
    # qualifying lap: five of Verstappen's qualifying-fastest weekends carried a
    # grid penalty (three of them Spa), and in 2021 the sprint winner was awarded
    # pole. Counting qualifying P1 gives 51; counting grid P1 gives 48, which is
    # the figure public sources report, season by season.
    poles = ver[ver.grid == 1].groupby("year").size()
    fastest_quali = q[(q.driverId == VER) & (q.position == 1)].groupby("year").size()
    avg_quali = q[q.driverId == VER].groupby("year").position.mean()

    # ---- qualifying gap to teammate, in seconds ---------------------------
    # The head-to-head count says who was faster; this says by how much. Parse
    # the lap strings (m:ss.mmm) in q1/q2/q3, take each driver's best of the
    # weekend, and difference Verstappen against his teammate. Negative = MV
    # faster. Reported as the season median so one wet outlier can't swing it.
    def _to_seconds(t):
        m = re.match(r"(?:(\d+):)?(\d+)\.(\d+)$", str(t).strip())
        if not m:
            return np.nan
        return int(m.group(1) or 0) * 60 + int(m.group(2)) + int(m.group(3)) / 1000

    qt = quali.merge(races[["raceId", "year"]], on="raceId").copy()
    for c in ("q1", "q2", "q3"):
        qt[c + "s"] = qt[c].map(_to_seconds)
    qt["best"] = qt[["q1s", "q2s", "q3s"]].min(axis=1, skipna=True)
    vqt = qt[qt.driverId == VER][["raceId", "year", "constructorId", "best"]].rename(
        columns={"best": "ver_best"}
    )
    mqt = qt.merge(vqt, on=["raceId", "year", "constructorId"])
    mqt = mqt[(mqt.driverId != VER) & mqt.best.notna() & mqt.ver_best.notna()].copy()
    # keep the faster teammate on the rare weekend a seat was shared
    mqt = mqt.sort_values("best").groupby(["raceId", "year"], as_index=False).first()
    mqt["gap"] = mqt.ver_best - mqt.best
    quali_gap = mqt.groupby("year").agg(
        gap_median=("gap", "median"), gap_n=("gap", "size")
    )

    # ---- season table ------------------------------------------------------
    seasons = []
    for yr in scored_years:
        d = ver[ver.year == yr]
        if d.empty:
            continue
        gl = moves.loc[(yr, VER)] if (yr, VER) in moves.index else pd.Series({"gained": 0, "lost": 0})
        gained, lost = float(gl.gained), float(gl.lost)
        ll = int(led.get((yr, VER), 0))
        total_lead = int(season_lead_laps.get(yr, 0))
        hh = h2h.loc[yr] if yr in h2h.index else None
        seasons.append(
            {
                "year": int(yr),
                "team": " / ".join(sorted(set(d.name_team))),
                "races": int(len(d)),
                "wins": int((d.fin == 1).sum()),
                "podiums": int((d.fin <= 3).sum()),
                "poles": int(poles.get(yr, 0)),
                "fastest_quali": int(fastest_quali.get(yr, 0)),
                "points": float(d.points.sum()) + float(sprint_pts.get(yr, 0.0)),
                "race_points": float(d.points.sum()),
                "sprint_points": float(sprint_pts.get(yr, 0.0)),
                "dnf": int((~d.classified).sum()),
                "finish_rate": round(float(d.classified.mean()) * 100, 1),
                "avg_grid": round(float(d.grid.mean()), 2),
                "avg_finish": round(float(d.fin.mean()), 2),
                "avg_quali": round(float(avg_quali.get(yr, np.nan)), 2),
                "net_gain": int(np.nansum(d.gain)),
                "fastest_laps": int(d.fastest_lap.sum()),
                "laps_led": ll,
                "season_lead_laps": total_lead,
                "pct_laps_led": round(ll / total_lead * 100, 1) if total_lead else 0.0,
                "gained": gained,
                "lost": lost,
                "gain_ratio": round(gained / lost, 2) if lost else None,
                "h2h_n": int(hh.n) if hh is not None else 0,
                "h2h_won": int(hh.won) if hh is not None else 0,
                "h2h_pct": round(float(hh.won) / float(hh.n) * 100) if hh is not None and hh.n else None,
                "teammates": hh.mates if hh is not None else [],
                "quali_gap_s": (
                    round(float(quali_gap.loc[yr, "gap_median"]), 3)
                    if yr in quali_gap.index else None
                ),
                "quali_gap_n": (
                    int(quali_gap.loc[yr, "gap_n"]) if yr in quali_gap.index else 0
                ),
            }
        )

    # ---- who else led laps, per season (the field closing in) --------------
    rivals = {}
    for yr in scored_years:
        s = led.loc[yr] if yr in led.index.get_level_values(0) else pd.Series(dtype=int)
        total = int(season_lead_laps.get(yr, 0)) or 1
        top = s.sort_values(ascending=False).head(5)
        rivals[int(yr)] = [
            {
                "driver": name_of[i],
                "laps_led": int(v),
                "pct": round(v / total * 100, 1),
                "is_ver": bool(i == VER),
            }
            for i, v in top.items()
        ]

    # ---- race-by-race strip and the standout drives ------------------------
    timeline = [
        {
            "year": int(r.year),
            "round": int(r.round),
            "race": r.name_race,
            "grid": None if pd.isna(r.grid) else int(r.grid),
            "finish": None if pd.isna(r.fin) else int(r.fin),
            "points": float(r.points),
            "status": r.status,
            "classified": bool(r.classified),
        }
        for r in ver.sort_values(["year", "round"]).itertuples()
    ]

    comebacks = [
        {
            "year": int(r.year),
            "race": r.name_race,
            "grid": int(r.grid),
            "finish": int(r.fin),
            "gain": int(r.gain),
        }
        for r in ver[ver.classified & ver.gain.notna()]
        .nlargest(8, "gain")
        .itertuples()
    ]


    # ---- 2025 split by half: the second-half recovery --------------------
    # 2025 ran 24 rounds; split 1-12 / 13-24. Points include sprint points.
    v25 = ver[ver.year == 2025]
    led25 = (
        laps[(laps.year == 2025) & (laps.position == 1)]
        .merge(races[["raceId", "round"]], on="raceId")
    )
    led25 = led25[led25.driverId == VER]
    spr25 = sprint.merge(races[["raceId", "year", "round"]], on="raceId")
    spr25 = spr25[(spr25.year == 2025) & (spr25.driverId == VER)]
    split_2025 = {"mid": 12, "halves": []}
    for lo, hi, lab in ((1, 12, "Rounds 1–12"), (13, 24, "Rounds 13–24")):
        d = v25[(v25["round"] >= lo) & (v25["round"] <= hi)]
        race_pts = float(d.points.sum())
        sp_pts = float(spr25[(spr25["round"] >= lo) & (spr25["round"] <= hi)].points.sum())
        ll = int(len(led25[(led25["round"] >= lo) & (led25["round"] <= hi)]))
        split_2025["halves"].append(
            {
                "label": lab,
                "races": int(len(d)),
                "wins": int((d.fin == 1).sum()),
                "podiums": int((d.fin <= 3).sum()),
                "dnf": int((~d.classified).sum()),
                "points": round(race_pts + sp_pts, 1),
                "avg_finish": round(float(d.fin.mean()), 2),
                "avg_grid": round(float(d.grid.mean()), 2),
                "laps_led": ll,
            }
        )

    # ---- the overtaking illusion: raw volume vs gained-per-lost -----------
    FOCUS = 2023
    starts = f.groupby(["year", "driverId"]).raceId.nunique().rename("starts")
    grids = (
        res[res.classified | ~res.classified]
        .groupby(["year", "driverId"])
        .grid.mean()
        .rename("avg_grid")
    )
    ot = moves.join(starts).join(grids).reset_index()
    ot = ot[(ot.year == FOCUS) & (ot.starts >= 10)].copy()
    ot["ratio"] = ot.gained / ot.lost.replace(0, np.nan)

    def ot_rows(frame):
        return [
            {
                "driver": name_of[r.driverId],
                "gained": int(r.gained),
                "lost": int(r.lost),
                "ratio": round(float(r.ratio), 2),
                "avg_grid": round(float(r.avg_grid), 1),
                "is_ver": bool(r.driverId == VER),
            }
            for r in frame.itertuples()
        ]

    by_gained = ot.sort_values("gained", ascending=False)
    ver_rank = int((by_gained.driverId.values == VER).argmax()) + 1
    top_raw = by_gained.head(10)
    if VER not in top_raw.driverId.values:
        top_raw = pd.concat([top_raw, by_gained[by_gained.driverId == VER]])
    top_ratio = ot.sort_values("ratio", ascending=False).head(8)

    wins = ver[ver.fin == 1]
    payload = {
        "meta": {
            "source": "Ergast / Kaggle jtrotman/formula-1-race-data",
            "through": str(res.date.max()),
            "driver": "Max Verstappen",
        },
        "career": {
            "races": int(len(ver)),
            "wins": int((ver.fin == 1).sum()),
            "podiums": int((ver.fin <= 3).sum()),
            "poles": int(poles.sum()),
            "fastest_laps": int(ver.fastest_lap.sum()),
            "points": float(ver.points.sum()) + float(sprint_pts.sum()),
            "laps_led": int(sum(s["laps_led"] for s in seasons)),
            "wins_from_pole": int((wins.grid == 1).sum()),
            "wins_off_pole": int(len(wins)) - int((wins.grid == 1).sum()),
        },
        "seasons": seasons,
        "rivals": rivals,
        "timeline": timeline,
        "comebacks": comebacks,
        "split_2025": split_2025,
        "overtakes_2023_raw": ot_rows(top_raw),
        "overtakes_2023_ratio": ot_rows(top_ratio),
        "overtakes_2023_ver_rank": ver_rank,
        "overtakes_2023_pool": int(len(ot)),
        "overtakes_2023_focus": FOCUS,
        "wins_by_grid": {
            int(k): int(v) for k, v in wins.grid.value_counts().sort_index().items()
        },
    }

    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(payload, indent=2))

    template = (SITE / "template.html").read_text()
    marker = "/*__DATA__*/"
    if marker not in template:
        print(f"error: {marker} not found in site/template.html", file=sys.stderr)
        return 1
    html = template.replace(marker, json.dumps(payload, separators=(",", ":")))
    (SITE / "index.html").write_text(html)

    print(f"seasons {seasons[0]['year']}-{seasons[-1]['year']}  "
          f"races {payload['career']['races']}  wins {payload['career']['wins']}  "
          f"poles {payload['career']['poles']}")
    print(f"VER 2023 rank by raw places gained: {ver_rank}")
    print(f"wrote site/data.json and site/index.html ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
