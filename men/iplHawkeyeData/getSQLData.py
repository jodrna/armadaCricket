# import sqlalchemy
# import pandas as pd
# from urllib.parse import quote
# from db import engine
# from paths import PROJECT_ROOT

# connection = engine.connect()

#
# # load your local file (must contain a match id column)
# df = pd.read_csv("/Users/jordan/Documents/ArmadaCricket/iplHawkeyeData/iplData.csv")
#
# # # ---- build list of ids for SQL IN (...) ----
# # match_ids = (
# #     df['matchid']
# #     .dropna()
# #     .astype(int)
# #     .drop_duplicates()
# #     .tolist()
# # )
# #
# # match_ids_sql = ",".join(str(x) for x in match_ids)
# #
# # # ---- query venue per matchid ----
# # allData = pd.read_sql_query(
# #     f"""
# #     select
# #         tb.matchid,
# #         max(tb.venue) as venue
# #     from match_data.t20_bbb tb
# #     where tb.matchid in ({match_ids_sql})
# #     group by tb.matchid
# #     """,
# #     con=connection
# # )
# #
# # # ---- merge venue back to df ----
# # df = df.merge(
# #     allData,
# #     how="left", on="matchid"
# # )
#
#
#
#
# # # -----------------------
# # # Wicket-ball flag (0/1) + innings ball number (resets on Phase 3 -> 1)
# # # Assumes df is ALREADY in true delivery order. NO SORTING.
# # # -----------------------
# #
# # df["wicket_ball"] = (
# #     df
# #     .groupby(["matchid"])["Wickets"]
# #     .diff()
# #     .fillna(0)
# #     .eq(1)
# #     .astype(int)
# # )
# #
# # df["new_innings"] = (
# #     (df["matchid"] != df["matchid"].shift()) |
# #     ((df["Phase"] == 1) & (df["Phase"].shift() == 3) & (df["matchid"] == df["matchid"].shift()))
# # )
# #
# # df["innings_id"] = df["new_innings"].cumsum()
# #
# # df["inningBallNumber"] = (
# #     df
# #     .groupby("innings_id")
# #     .cumcount() + 1
# # )
#
# df = df.drop(columns=["new_innings", 'ballAge'])
#
# df.to_csv("/Users/jordan/Documents/ArmadaCricket/iplHawkeyeData/iplData.csv", index=False)
#
#
#
# for column in df.columns:
#     print(column)



