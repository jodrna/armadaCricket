# PROCESS: stage 3 - fills player_ratings.t20_lineups_updated_w. Run BEFORE 2_adjData. Writes to the DB.
# See PROCESS.txt at the top of prod_model for the whole run order.
# project root = the first folder above this script holding paths.py, so paths / db / shared modules come from THIS
# project wherever it is (OneDrive, GitHub, a copy) - no PYTHONPATH needed (20 Oct 2026)
import sys as _sys
from pathlib import Path as _Path
_root = str(next(p for p in _Path(__file__).resolve().parents if (p / 'paths.py').exists()))
if _root in _sys.path: _sys.path.remove(_root)
_sys.path.insert(0, _root)
import pandas as pd
from paths import PROJECT_ROOT
import sqlalchemy

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# TOGGLE: set to False to disable writing to database (for testing)
# ──────────────────────────────────────────────
WRITE_TO_DB = True

from sqlalchemy import text
from db import engine  # connection details come from .env via db.py (no password in the repo)
# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
# ball by ball data straight from 1_getData in this repo (run it first) - was an old OneDrive PythonData copy that
# stopped in July, so newer matches never got lineups and 2_adjData dropped them. The repo file uses camelCase names
bbb = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/Cleaned_t20bbb3_new_w.csv', parse_dates=['date'])
bbb = bbb.rename(columns={'matchID': 'matchid', 'battingTeam': 'battingteam', 'batsmanName': 'batsman', 'inningNumber': 'innings'})
test = bbb[bbb.matchid == 1000162728]
with engine.connect() as conn:
    lineups = pd.read_sql("SELECT * FROM match_data.t20_lineups_w", conn)


bbb["date"] = pd.to_datetime(bbb["date"])

# ──────────────────────────────────────────────
# FIND MATCHES IN BBB NOT IN LINEUPS
# ──────────────────────────────────────────────
existing_match_ids = set(lineups["matchid"].unique())
missing_match_ids = [m for m in bbb["matchid"].unique() if m not in existing_match_ids]
print(f"Found {len(missing_match_ids)} matches to process.")

# ──────────────────────────────────────────────
# PRECOMPUTE: match dates, average batting positions, name lookups
# ──────────────────────────────────────────────
match_dates = bbb.groupby("matchid")["date"].first()

avg_batting_pos = (
    bbb.dropna(subset=["batterid", "ord"])
    .drop_duplicates(subset=["batterid", "matchid"])
    .groupby("batterid")["ord"]
    .mean()
)

batter_names = (
    bbb.dropna(subset=["batterid", "batsman"])
    .groupby("batterid")["batsman"]
    .agg(lambda x: x.value_counts().index[0])
)
bowler_names = (
    bbb.dropna(subset=["bowlerid", "bowler"])
    .groupby("bowlerid")["bowler"]
    .agg(lambda x: x.value_counts().index[0])
)
player_names = pd.concat([batter_names, bowler_names])
player_names = player_names[~player_names.index.duplicated(keep="first")]

# ──────────────────────────────────────────────
# PRECOMPUTE: deduplicated batting table split by team for fast lookup
# ──────────────────────────────────────────────
bbb_batting_deduped = (
    bbb.dropna(subset=["batterid", "ord", "battingteam", "matchid"])
    .drop_duplicates(subset=["batterid", "matchid"])[["matchid", "battingteam", "batterid", "ord"]]
)
bbb_by_team = {team: grp for team, grp in bbb_batting_deduped.groupby("battingteam")}

# ──────────────────────────────────────────────
# HELPER: infer missing players from nearby matches
# ──────────────────────────────────────────────
def infer_from_nearby_matches(team, known_ids, missing_positions, match_date, match_id, match_dates):

    team_df = bbb_by_team.get(team, pd.DataFrame())

    if team_df.empty:
        return {pos: ("unknown", None) for pos in missing_positions}

    team_df = team_df[team_df["matchid"] != match_id].copy()

    if team_df.empty:
        return {pos: ("unknown", None) for pos in missing_positions}

    # Attach date proximity and exclude known players
    team_df["date_diff"] = team_df["matchid"].map(match_dates).sub(match_date).abs()
    team_df = team_df[~team_df["batterid"].isin(known_ids)]

    result = {}
    already_assigned = set(known_ids)

    for pos in sorted(missing_positions):
        pos_df = team_df[
            (team_df["ord"].between(pos - 1, pos + 1)) &
            (~team_df["batterid"].isin(already_assigned))
        ].copy()

        if pos_df.empty:
            result[pos] = ("unknown", None)
            continue

        pos_df["days_diff"] = pos_df["date_diff"].dt.days.clip(lower=1)
        pos_df["weight"] = 1 / pos_df["days_diff"]

        scores = pos_df.groupby("batterid")["weight"].sum()
        best_id = scores.idxmax()
        best_name = player_names.get(best_id, "unknown")

        result[pos] = (best_name, best_id)
        already_assigned.add(best_id)

    return result

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
new_rows = []

for match_id in missing_match_ids:
    match_df = bbb[bbb["matchid"] == match_id]
    match_date = match_dates[match_id]

    batting_teams = match_df["battingteam"].dropna().unique()

    for team in batting_teams:
        team_df = match_df[match_df["battingteam"] == team]

        bat_inn = team_df["innings"].mode().iloc[0]

        # Players who actually batted, in order
        batted = (
            team_df.dropna(subset=["batterid", "ord"])
            .drop_duplicates(subset=["batterid"])
            .sort_values("ord")[["batterid", "batsman", "ord"]]
        )
        confirmed_ids = list(batted["batterid"])
        confirmed_names = list(batted["batsman"])
        confirmed_positions = list(batted["ord"].astype(int))

        # Bowlers for this team who didn't bat
        opposition_df = match_df[match_df["battingteam"] != team]
        team_bowlers = (
            opposition_df.dropna(subset=["bowlerid", "bowler"])
            .drop_duplicates(subset=["bowlerid"])[["bowlerid", "bowler"]]
        )
        didnt_bat = team_bowlers[~team_bowlers["bowlerid"].isin(confirmed_ids)]

        # Build lineup from confirmed batters
        full_lineup = {}
        for pos, pid, name in zip(confirmed_positions, confirmed_ids, confirmed_names):
            full_lineup[pos] = (name, pid)

        batted_flag = {pos: 1 for pos in confirmed_positions}

        # Remaining unfilled positions
        all_positions = list(range(1, 12))
        remaining_positions = [p for p in all_positions if p not in full_lineup]

        # Estimate positions for non-batters using global average
        estimated = []
        for _, row in didnt_bat.iterrows():
            pid = row["bowlerid"]
            name = row["bowler"]
            est_pos = avg_batting_pos.get(pid, 10)
            estimated.append((pid, name, est_pos))

        # Match each non-batter to closest remaining position
        remaining_positions_copy = remaining_positions.copy()

        for pid, name, est_pos in sorted(estimated, key=lambda x: x[2]):
            if not remaining_positions_copy:
                break
            closest = min(remaining_positions_copy, key=lambda p: abs(p - est_pos))
            full_lineup[closest] = (name, pid)
            batted_flag[closest] = 0
            remaining_positions_copy.remove(closest)

        # Fill still-missing positions using nearby matches
        still_missing = [p for p in all_positions if p not in full_lineup]

        if still_missing:
            print(f"Match {match_id}, team {team}: {len(still_missing)} position(s) "
                  f"unfilled — checking nearby matches...")

            known_ids = {v[1] for v in full_lineup.values() if v[1] is not None}
            inferred = infer_from_nearby_matches(
                team=team,
                known_ids=known_ids,
                missing_positions=still_missing,
                match_date=match_date,
                match_id=match_id,
                match_dates=match_dates
            )
            full_lineup.update(inferred)
            for pos in still_missing:
                batted_flag[pos] = 0

        # Final safety net
        for pos in all_positions:
            if pos not in full_lineup:
                full_lineup[pos] = ("unknown", None)
                batted_flag[pos] = 0

        for pos in all_positions:
            name, pid = full_lineup[pos]
            new_rows.append({
                "matchid": match_id,
                "playerid": pid,
                "player": name,
                "carded": pos,
                "team": team,
                "inns": bat_inn,
                "in_original_lineups": 0,
                "actually_batted": batted_flag.get(pos, 0)
            })

# ──────────────────────────────────────────────
# TAG ORIGINAL LINEUPS ROWS AND COMBINE
# ──────────────────────────────────────────────
lineups["in_original_lineups"] = 1
lineups["actually_batted"] = 1

new_df = pd.DataFrame(new_rows)
lineups_updated = pd.concat([lineups, new_df], ignore_index=True)

print(f"\nOriginal lineups rows: {len(lineups)}")
print(f"New rows added:        {len(new_df)}")
print(f"Total rows:            {len(lineups_updated)}")



if WRITE_TO_DB:
    # the table is owned by another role (dbadmin), so its rows are replaced rather than the table itself (keeps owner and grants;
    # one transaction, so nothing changes unless the whole write goes through)
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM player_ratings.t20_lineups_updated_w'))
        lineups_updated.to_sql("t20_lineups_updated_w", con=conn, schema="player_ratings",
                               if_exists="append", index=False)
        conn.execute(sqlalchemy.text("GRANT SELECT ON TABLE player_ratings.t20_lineups_updated_w TO jordan;"))
    print("Done! Written to player_ratings.t20_lineups_updated_w")
else:
    print("WRITE_TO_DB is disabled — skipping database write.")

# ══════════════════════════════════════════════
# TESTING & VALIDATION
# ══════════════════════════════════════════════

bbb["year"] = bbb["date"].dt.year

# ──────────────────────────────────────────────
# 1. OVERALL MATCH COUNTS
# ──────────────────────────────────────────────
bbb_match_count = bbb["matchid"].nunique()
lineups_match_count = lineups["matchid"].nunique()
new_match_count = new_df["matchid"].nunique()
total_covered = lineups_match_count + new_match_count

print("\n")
print("=" * 50)
print("OVERALL MATCH COUNTS")
print("=" * 50)
print(f"Matches in bbb:                   {bbb_match_count}")
print(f"Matches in original lineups:      {lineups_match_count}")
print(f"Matches newly added:              {new_match_count}")
print(f"Total matches now covered:        {total_covered}")
print(f"Matches still missing:            {bbb_match_count - total_covered}")

# ──────────────────────────────────────────────
# 2. BY YEAR
# ──────────────────────────────────────────────
match_years = bbb.drop_duplicates(subset=["matchid"])[["matchid", "year"]]

original_ids = set(lineups["matchid"].unique())
new_ids = set(new_df["matchid"].unique())

match_years["in_original"] = match_years["matchid"].isin(original_ids).astype(int)
match_years["newly_added"] = match_years["matchid"].isin(new_ids).astype(int)
match_years["still_missing"] = (
    (~match_years["matchid"].isin(original_ids)) &
    (~match_years["matchid"].isin(new_ids))
).astype(int)

year_summary = match_years.groupby("year").agg(
    total_in_bbb=("matchid", "count"),
    in_original_lineups=("in_original", "sum"),
    newly_added=("newly_added", "sum"),
    still_missing=("still_missing", "sum")
).reset_index()

year_summary["total_covered"] = year_summary["in_original_lineups"] + year_summary["newly_added"]
year_summary["pct_covered"] = (
    year_summary["total_covered"] / year_summary["total_in_bbb"] * 100
).round(1)

print("\n")
print("=" * 50)
print("MATCH COUNTS BY YEAR")
print("=" * 50)
print(year_summary.to_string(index=False))

# ──────────────────────────────────────────────
# 3. SPOT CHECK: 5 random newly added matches
# ──────────────────────────────────────────────
print("\n")
print("=" * 50)
print("SPOT CHECK: 5 RANDOM NEWLY ADDED MATCHES")
print("=" * 50)

sample_ids = pd.Series(list(new_ids)).sample(min(5, len(new_ids)), random_state=42)

for match_id in sample_ids:
    match_bbb = bbb[bbb["matchid"] == match_id]
    match_date = match_bbb["date"].iloc[0].date()
    teams = match_bbb["battingteam"].dropna().unique()

    print(f"\nMatch ID: {match_id}  |  Date: {match_date}  |  Teams: {' vs '.join(teams)}")
    print("-" * 40)

    match_rows = lineups_updated[lineups_updated["matchid"] == match_id].sort_values(
        ["inns", "carded"]
    )

    for inns, inn_df in match_rows.groupby("inns"):
        print(f"  Innings {inns}:")
        for _, row in inn_df.iterrows():
            batted_label = "batted" if row["actually_batted"] == 1 else "inferred"
            pid = row["playerid"] if pd.notna(row["playerid"]) else "no id"
            print(f"    {int(row['carded']):>2}. {str(row['player']):<30} (id: {pid}) [{batted_label}]")

# ──────────────────────────────────────────────
# 4. QUALITY SUMMARY
# ──────────────────────────────────────────────
total = len(new_df)
confirmed = (new_df["actually_batted"] == 1).sum()
inferred = (new_df["actually_batted"] == 0).sum()
unknown = (new_df["player"] == "unknown").sum()

print("\n")
print("=" * 50)
print("QUALITY SUMMARY: NEWLY ADDED ROWS")
print("=" * 50)
print(f"Total player-rows added:     {total}")
print(f"Confirmed (actually batted): {confirmed} ({confirmed/total*100:.1f}%)")
print(f"Inferred:                    {inferred} ({inferred/total*100:.1f}%)")
print(f"Of which 'unknown':          {unknown} ({unknown/total*100:.1f}%)")

# ──────────────────────────────────────────────
# 5. MATCHES WITH HIGH NUMBER OF UNKNOWNS
# ──────────────────────────────────────────────
print("\n")
print("=" * 50)
print("MATCHES WITH HIGH UNKNOWNS")
print("=" * 50)

unknown_counts = (
    new_df[new_df["player"] == "unknown"]
    .groupby("matchid")
    .size()
    .reset_index(name="unknown_count")
    .sort_values("unknown_count", ascending=False)
)

# Flag matches with more than 2 unknowns (adjust threshold as needed)
UNKNOWN_THRESHOLD = 2
high_unknown = unknown_counts[unknown_counts["unknown_count"] > UNKNOWN_THRESHOLD]

if high_unknown.empty:
    print(f"No matches with more than {UNKNOWN_THRESHOLD} unknowns.")
else:
    print(f"Found {len(high_unknown)} matches with more than {UNKNOWN_THRESHOLD} unknowns:\n")
    for _, row in high_unknown.iterrows():
        match_id = row["matchid"]
        unknown_count = row["unknown_count"]
        match_date = bbb[bbb["matchid"] == match_id]["date"].iloc[0].date()
        teams = bbb[bbb["matchid"] == match_id]["battingteam"].dropna().unique()

        print(f"  Match ID: {match_id}  |  Date: {match_date}  |  "
              f"Teams: {' vs '.join(teams)}  |  Unknowns: {unknown_count}")