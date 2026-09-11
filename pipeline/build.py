"""Build the Verstappen-decade infographic from the Ergast/Kaggle F1 CSVs.

Reads data/raw/*.csv, computes the season metrics the story needs, and injects
them into docs/template.html to produce a single self-contained docs/index.html.

Run:  python3 pipeline/build.py
"""
from __future__ import annotations

import base64
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
SITE = ROOT / "docs"   # GitHub Pages serves main:/docs

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
    vqt = qt[qt.driverId == VER][
        ["raceId", "year", "constructorId", "q1s", "q2s", "q3s"]
    ].rename(columns={"q1s": "v1", "q2s": "v2", "q3s": "v3"})
    mqt = qt.merge(vqt, on=["raceId", "year", "constructorId"])
    mqt = mqt[mqt.driverId != VER].copy()
    mqt["v_best"] = mqt[["v1", "v2", "v3"]].min(axis=1)
    mqt["m_best"] = mqt[["q1s", "q2s", "q3s"]].min(axis=1)
    mqt = mqt[mqt.v_best.notna() & mqt.m_best.notna()]
    # keep the faster teammate on the rare weekend a seat was shared
    mqt = mqt.sort_values("m_best").groupby(["raceId", "year"], as_index=False).first()

    def _same_session(r):
        """Gap taken from the deepest session BOTH cars set a time in.

        Comparing each driver's best lap of the weekend silently compares a Q3
        lap against a Q1 lap whenever the teammate is eliminated early — lower
        fuel, fresher tyres and an evolved track, worth a few tenths on its own.
        Verstappen out-qualified his teammate into a deeper session on 71% of
        2025 weekends and 38% of 2024, so that mismatch inflated exactly the
        seasons the story leaned on (2025 read -0.805s; same-session is -0.540s).
        """
        for mine, theirs in ((r.v3, r.q3s), (r.v2, r.q2s), (r.v1, r.q1s)):
            if pd.notna(mine) and pd.notna(theirs):
                return mine - theirs
        return np.nan

    mqt["gap"] = mqt.apply(_same_session, axis=1)
    mqt["gap_best_of_weekend"] = mqt.v_best - mqt.m_best
    quali_gap = mqt.dropna(subset=["gap"]).groupby("year").agg(
        gap_median=("gap", "median"),
        gap_n=("gap", "size"),
        gap_best_median=("gap_best_of_weekend", "median"),
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
                "quali_gap_best_s": (
                    round(float(quali_gap.loc[yr, "gap_best_median"]), 3)
                    if yr in quali_gap.index else None
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

    # ---- championship years, from the final standings of each season ------
    standings = read("driver_standings").merge(
        races[["raceId", "year", "round"]], on="raceId"
    )
    finals = standings[
        standings["round"] == standings.groupby("year")["round"].transform("max")
    ]
    champion_years = sorted(
        int(y) for y in finals[(finals.driverId == VER) & (finals.position == 1)].year
    )

    # ---- 2025 title fight: cumulative championship points, round by round --
    TITLE_2025 = {VER: "Verstappen", 846: "Norris", 857: "Piastri"}
    r25 = races[races.year == 2025][["raceId", "round", "name"]]
    race_pts = results[["raceId", "driverId", "points"]]
    sp_pts = sprint[["raceId", "driverId", "points"]]
    both = pd.concat([race_pts, sp_pts])
    both = both[both.driverId.isin(TITLE_2025)].merge(r25, on="raceId")
    per_round = (
        both.groupby(["driverId", "round"], as_index=False).points.sum()
        .sort_values(["driverId", "round"])
    )
    per_round["cum"] = per_round.groupby("driverId").points.cumsum()
    rounds_25 = sorted(r25["round"].unique())
    title_2025 = {
        "rounds": [int(r) for r in rounds_25],
        "races": [
            r25[r25["round"] == r].name.iloc[0].replace(" Grand Prix", "")
            for r in rounds_25
        ],
        "drivers": [
            {
                "driver": label,
                "is_ver": bool(did == VER),
                "cum": [
                    float(
                        per_round[(per_round.driverId == did)
                                  & (per_round["round"] <= r)].points.sum()
                    )
                    for r in rounds_25
                ],
            }
            for did, label in TITLE_2025.items()
        ],
    }

    # ---- grand slams: pole + win + fastest lap + led every lap -------------
    # Lap-by-lap data is complete from 1982, so this is computable for Senna,
    # Prost, Mansell and Schumacher as well as the modern era — but NOT before
    # 1982, which is why Jim Clark's all-time record of eight cannot appear here.
    # The fastest lap is derived from lap_times rather than results.rank, which
    # only exists from 2004 and so would silently exclude three decades.
    SLAM_MIN_LAPS = 20
    lap_era = lap_times.merge(races[["raceId", "year", "name"]], on="raceId")
    lap_era = lap_era[lap_era.year >= 1982]
    race_laps = lap_era.groupby("raceId").lap.max().rename("n_laps")
    leader = (
        lap_era[lap_era.position == 1]
        .groupby(["raceId", "driverId"]).lap.agg(["count", "min", "max"])
        .join(race_laps, on="raceId")
    )
    leader["led_all"] = (
        (leader["count"] == leader.n_laps)
        & (leader["min"] == 1)
        & (leader["max"] == leader.n_laps)
    )
    led_all = leader[leader.led_all].reset_index()[["raceId", "driverId"]]
    led_all["led_all"] = True
    fastest = lap_era.loc[lap_era.groupby("raceId").milliseconds.idxmin()][
        ["raceId", "driverId"]
    ].assign(fastest=True)

    slam_src = res[res.year >= 1982].copy()
    slam_src["won"] = slam_src.positionText.astype(str) == "1"
    slam_src = (
        slam_src.merge(led_all, on=["raceId", "driverId"], how="left")
        .merge(fastest, on=["raceId", "driverId"], how="left")
        .merge(race_laps, on="raceId", how="left")
    )
    for col in ("led_all", "fastest"):
        slam_src[col] = slam_src[col].notna() & (slam_src[col] == True)  # noqa: E712
    # A race abandoned behind the safety car satisfies "led every lap" trivially:
    # Spa 2021 ran a single lap in this data and would otherwise score as a slam
    # for Verstappen. Real slams in the era run 51+ laps, so the cut is clean.
    slam_src["short_race"] = slam_src.n_laps < SLAM_MIN_LAPS
    slam_src["slam"] = (
        slam_src.won
        & (slam_src.grid == 1)
        & slam_src.led_all
        & slam_src.fastest
        & ~slam_src.short_race
    )
    slams = slam_src[slam_src.slam]
    slam_counts = slams.groupby("driverId").size().sort_values(ascending=False)
    grand_slams = {
        "min_laps": SLAM_MIN_LAPS,
        "since": 1982,
        "total": int(len(slams)),
        "seasons_covered": int(res[res.year >= 1982].year.nunique()),
        "leaders": [
            {
                "driver": name_of[did],
                "slams": int(n),
                "first": int(slams[slams.driverId == did].year.min()),
                "last": int(slams[slams.driverId == did].year.max()),
                # Starts and seasons are career totals, so the comparison is
                # "how many attempts did each slam take", not "how long was
                # the career" — the two are easy to conflate.
                "starts": int((results.driverId == did).sum()),
                "seasons": int(res[res.driverId == did].year.nunique()),
                # Chronological, so the nth block in the chart is the nth slam.
                "races": [
                    {"year": int(r.year), "race": r.name_race.replace(" Grand Prix", "")}
                    for r in slams[slams.driverId == did]
                    .sort_values("date").itertuples()
                ],
                "is_ver": bool(did == VER),
            }
            for did, n in slam_counts.head(8).items()
        ],
        "ver_slams": [
            {"year": int(r.year), "race": r.name_race, "laps": int(r.n_laps)}
            for r in slams[slams.driverId == VER].sort_values("date").itertuples()
        ],
        "excluded": [
            {"year": int(r.year), "race": r.name_race, "laps": int(r.n_laps)}
            for r in slam_src[
                slam_src.won & (slam_src.grid == 1) & slam_src.led_all
                & slam_src.fastest & slam_src.short_race
            ].itertuples()
        ],
    }

    # ---- driver profiles for the parallel-coordinates chart ---------------
    # Pool: the 30 winningest drivers in F1 history, so the chart spans Fangio
    # to Verstappen rather than one era.
    #
    # Every axis here is complete for 1950-2026, which is why "laps led" is NOT
    # among them: lap-by-lap timing only begins in 1982, and eight of these
    # thirty (Fangio, Ascari, Clark, Moss, Brabham, Graham Hill, Stewart,
    # Fittipaldi) raced entirely before it. Including it would either exclude
    # them or leave holes, and a brush over a half-empty axis means nothing.
    #
    # Pole rate comes from the grid, not the qualifying table, which starts in
    # 1994 and is patchy to 2000.
    PROFILE_N = 30
    win_counts = res[res.fin == 1].groupby("driverId").size()
    pool = list(win_counts.nlargest(PROFILE_N).index)
    titles = finals[finals.position == 1].groupby("driverId").size()

    # Ordered as a rarity funnel — turning up, scoring, the podium, pole, the
    # win, the title. Only adjacent axes can be read against each other in a
    # parallel-coordinates plot, so this puts the related pairs side by side and
    # makes the career-shape crossings (a long career at modest rates against a
    # short one at high rates) run the length of the chart.
    profile_axes = [
        {"key": "starts", "label": "Starts", "unit": ""},
        {"key": "points", "label": "Points finishes", "unit": "%"},
        {"key": "podium", "label": "Podium rate", "unit": "%"},
        {"key": "pole", "label": "Pole rate", "unit": "%"},
        {"key": "win", "label": "Win rate", "unit": "%"},
        {"key": "titles", "label": "Titles", "unit": ""},
    ]
    driver_profiles = []
    for did in pool:
        x = res[res.driverId == did]
        if x.empty:
            continue
        driver_profiles.append({
            "driver": name_of[did],
            "last": name_of[did].split()[-1],
            "starts": int(len(x)),
            "span": f"{int(x.year.min())}\u2013{int(x.year.max())}",
            "era": int(x.year.min()),
            "is_ver": bool(did == VER),
            "win": round(float((x.fin == 1).mean()) * 100, 1),
            "pole": round(float((x.grid == 1).mean()) * 100, 1),
            "podium": round(float((x.fin <= 3).mean()) * 100, 1),
            # Era caveat, stated on the page: the scoring system paid the top 5
            # in the 1950s, top 6 to 2002, top 8 to 2009 and top 10 since 2010,
            # so this axis structurally favours modern drivers.
            "points": round(float((x.points > 0).mean()) * 100, 1),
            "titles": int(titles.get(did, 0)),
        })
    driver_profiles.sort(key=lambda r: -r["win"])

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
        "champion_years": champion_years,
        "grand_slams": grand_slams,
        "profile_axes": profile_axes,
        "driver_profiles": driver_profiles,
        "title_2025": title_2025,
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

    # Inline the hero artwork as a data URI: the published page must be
    # self-contained, and its CSP blocks external image hosts.
    hero = ROOT / "assets" / "max-hero.jpg"
    if "__HERO_IMG__" in template:
        if hero.exists():
            template = template.replace(
                "__HERO_IMG__", base64.b64encode(hero.read_bytes()).decode()
            )
        else:
            print(f"warning: {hero} missing — hero image will not render",
                  file=sys.stderr)
    marker = "/*__DATA__*/"
    if marker not in template:
        print(f"error: {marker} not found in docs/template.html", file=sys.stderr)
        return 1
    html = template.replace(marker, json.dumps(payload, separators=(",", ":")))
    (SITE / "index.html").write_text(html)

    print(f"seasons {seasons[0]['year']}-{seasons[-1]['year']}  "
          f"races {payload['career']['races']}  wins {payload['career']['wins']}  "
          f"poles {payload['career']['poles']}")
    print(f"VER 2023 rank by raw places gained: {ver_rank}")
    print(f"wrote docs/data.json and docs/index.html ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
