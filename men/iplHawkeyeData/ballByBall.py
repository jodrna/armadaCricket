import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


# -----------------------
# 1) Load + clean
# -----------------------
df = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/iplHawkeyeData/iplData.csv')
df = df.dropna()

num_cols = [
    "Pre Bounce Velocity_x",
    "Pre Bounce Velocity_y",
    "Post Bounce Velocity_x",
    "Post Bounce Velocity_y",
    "Batter Runs",
    "Ovrexpr",
    "inningBallNumber",
    "match.delivery.trajectory.dropAngle",
]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(
    subset=[
        "matchid",
        "venue",
        "Phase",
        "inningBallNumber",
        "match.delivery.trajectory.dropAngle",
        "Pre Bounce Velocity_x",
        "Pre Bounce Velocity_y",
        "Post Bounce Velocity_x",
        "Post Bounce Velocity_y",
        "Batter Runs",
        "Ovrexpr",
    ]
).copy()


# -----------------------
# 2) Per-ball retention metrics
# -----------------------
df = df[
    (df["Pre Bounce Velocity_x"] != 0) &
    (df["Pre Bounce Velocity_y"] != 0)
].copy()

df["pace_retention"] = (
    df["Post Bounce Velocity_x"] /
    df["Pre Bounce Velocity_x"]
)

df["bounce_retention"] = (
    df["Post Bounce Velocity_y"] /
    df["Pre Bounce Velocity_y"]
)

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(
    subset=[
        "pace_retention",
        "bounce_retention",
    ]
).copy()


# -----------------------
# 3) Remove 3-sigma outliers
# -----------------------
for col in ["pace_retention", "bounce_retention"]:
    mu = df[col].mean()
    sd = df[col].std(ddof=0)
    df = df[(df[col] >= mu - 3 * sd) & (df[col] <= mu + 3 * sd)]


# -----------------------
# 4) Continuous physics features
# -----------------------
df["drop_angle"] = df["match.delivery.trajectory.dropAngle"]

df["speed_x_drop_angle"] = (
    df["Pre Bounce Velocity_x"] *
    df["drop_angle"]
)

df["inningBallNumber_log"] = np.log(df["inningBallNumber"])
df["drop_angle_sq"] = df["drop_angle"] ** 2


# -----------------------
# 5) CV by match
# -----------------------
match_table = (
    df
    .groupby(["matchid", "venue"], dropna=False)
    .agg(
        deliveries=("Batter Runs", "size"),
        ovrexpr=("Ovrexpr", "mean"),
        total_batter_runs=("Batter Runs", "sum"),
    )
    .reset_index()
)

match_table["avg_runs_per_ball"] = (
    match_table["total_batter_runs"] /
    match_table["deliveries"]
)

kf = KFold(n_splits=20, shuffle=True, random_state=42)

oof_pred_base = np.full(len(match_table), np.nan)
oof_pred_full = np.full(len(match_table), np.nan)
oof_fold = np.full(len(match_table), np.nan)

rows = []


for fold, (train_idx, test_idx) in enumerate(kf.split(match_table), start=1):

    train_matchids = set(match_table.iloc[train_idx]["matchid"])
    test_matchids = set(match_table.iloc[test_idx]["matchid"])

    train_balls = df[df["matchid"].isin(train_matchids)].copy()
    test_balls = df[df["matchid"].isin(test_matchids)].copy()

    # -----------------------
    # 5a) predict expected bounce and pace based on dropAngle, speed, innings ball
    # -----------------------
    base_feats = [
        "Pre Bounce Velocity_x",
        # "drop_angle",
        "speed_x_drop_angle",
        "inningBallNumber"
    ]

    Xp_train = train_balls[base_feats]
    Xp_test = test_balls[base_feats]

    # pace
    pace_model = LinearRegression()
    pace_model.fit(Xp_train.values, train_balls["pace_retention"].values)

    train_balls["pitch_pace_ball"] = (
        train_balls["pace_retention"] -
        pace_model.predict(Xp_train.values)
    )

    test_balls["pitch_pace_ball"] = (
        test_balls["pace_retention"] -
        pace_model.predict(Xp_test.values)
    )

    # bounce
    bounce_model = LinearRegression()
    bounce_model.fit(Xp_train.values, train_balls["bounce_retention"].values)

    train_balls["pitch_bounce_ball"] = (
        train_balls["bounce_retention"] -
        bounce_model.predict(Xp_train.values)
    )

    test_balls["pitch_bounce_ball"] = (
        test_balls["bounce_retention"] -
        bounce_model.predict(Xp_test.values)
    )

    # -----------------------
    # 5b) Aggregate pitch features
    # -----------------------
    train_pitch = (
        train_balls
        .groupby("matchid")
        .agg(
            pitch_pace=("pitch_pace_ball", "median"),
            pitch_bounciness=("pitch_bounce_ball", "median"),
        )
        .reset_index()
    )

    test_pitch = (
        test_balls
        .groupby("matchid")
        .agg(
            pitch_pace=("pitch_pace_ball", "median"),
            pitch_bounciness=("pitch_bounce_ball", "median"),
        )
        .reset_index()
    )

    train_df = (
        match_table[match_table["matchid"].isin(train_matchids)]
        .merge(train_pitch, on="matchid", how="inner")
    )

    test_df = (
        match_table[match_table["matchid"].isin(test_matchids)]
        .merge(test_pitch, on="matchid", how="inner")
    )

    # -----------------------
    # 5c) Runs model
    # -----------------------
    y_train = train_df["avg_runs_per_ball"].values
    y_test = test_df["avg_runs_per_ball"].values

    # base = ovrexpr
    pred_base = test_df["ovrexpr"].values
    oof_pred_base[test_idx] = pred_base

    X_train = train_df[["ovrexpr", "pitch_pace", "pitch_bounciness"]].values
    X_test = test_df[["ovrexpr", "pitch_pace", "pitch_bounciness"]].values

    m_full = LinearRegression()
    m_full.fit(X_train, y_train)

    pred_full = m_full.predict(X_test)
    oof_pred_full[test_idx] = pred_full
    oof_fold[test_idx] = fold

    rows.append(
        {
            "fold": fold,
            "r2_base": r2_score(y_test, pred_base),
            "r2_engineered": r2_score(y_test, pred_full),
            "mae_base": mean_absolute_error(y_test, pred_base),
            "mae_engineered": mean_absolute_error(y_test, pred_full),
        }
    )


cv_results = pd.DataFrame(rows)

cv_results["r2_gain_eng_minus_base"] = (
    cv_results["r2_engineered"] -
    cv_results["r2_base"]
)

cv_results["mae_improve_eng_minus_base"] = (
    cv_results["mae_base"] -
    cv_results["mae_engineered"]
)


# -----------------------
# 6) Attach OOF preds
# -----------------------
pitch_pivot = match_table.copy()

pitch_pivot["cv_fold"] = oof_fold
pitch_pivot["oof_pred_base"] = oof_pred_base
pitch_pivot["oof_pred_engineered"] = oof_pred_full

pitch_pivot["oof_abs_err_base"] = (
    pitch_pivot["avg_runs_per_ball"] -
    pitch_pivot["oof_pred_base"]
).abs()

pitch_pivot["oof_abs_err_engineered"] = (
    pitch_pivot["avg_runs_per_ball"] -
    pitch_pivot["oof_pred_engineered"]
).abs()


# -----------------------
# 7) Prints (same style)
# -----------------------
print(cv_results)
print()

print("Mean R2 base:", cv_results["r2_base"].mean())
print("Mean R2 engineered:", cv_results["r2_engineered"].mean())
print("Mean R2 gain (eng - base):", cv_results["r2_gain_eng_minus_base"].mean())
print()

print("Mean MAE base:", cv_results["mae_base"].mean())
print("Mean MAE engineered:", cv_results["mae_engineered"].mean())
print("Mean MAE improve (eng - base):", cv_results["mae_improve_eng_minus_base"].mean())


# -----------------------
# Helper: quantile-binned mean curve
# -----------------------
def binned_mean(x: pd.Series, y: pd.Series, n_bins: int = 40) -> pd.DataFrame:
    tmp = pd.DataFrame({"x": x.astype(float), "y": y.astype(float)}).dropna()
    tmp["bin"] = pd.qcut(tmp["x"], q=n_bins, duplicates="drop")
    out = (
        tmp
        .groupby("bin", observed=True)
        .agg(
            x_mid=("x", "median"),
            y_mean=("y", "mean"),
            n=("y", "size"),
        )
        .reset_index(drop=True)
        .sort_values("x_mid")
    )
    return out


# -----------------------
# Build curves
# -----------------------
x_order = ["drop_angle", "Pre Bounce Velocity_x", "inningBallNumber"]
y_order = ["pace_retention", "bounce_retention"]

titles = {
    ("drop_angle", "pace_retention"): "Drop Angle vs Pace Retention",
    ("Pre Bounce Velocity_x", "pace_retention"): "Speed (Pre Vx) vs Pace Retention",
    ("inningBallNumber", "pace_retention"): "Inning Ball Number vs Pace Retention",
    ("drop_angle", "bounce_retention"): "Drop Angle vs Bounce Retention",
    ("Pre Bounce Velocity_x", "bounce_retention"): "Speed (Pre Vx) vs Bounce Retention",
    ("inningBallNumber", "bounce_retention"): "Inning Ball Number vs Bounce Retention",
}

xlabels = {
    "drop_angle": "dropAngle (radians)",
    "Pre Bounce Velocity_x": "Pre Bounce Velocity_x",
    "inningBallNumber": "inningBallNumber",
}

ylabels = {
    "pace_retention": "pace_retention",
    "bounce_retention": "bounce_retention",
}

curves = {}
for yname in y_order:
    for xname in x_order:
        curves[(xname, yname)] = binned_mean(df[xname], df[yname], n_bins=40)


# -----------------------
# Plot: 2 rows x 3 cols = 6 plots
# -----------------------
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8), sharey=False)

for r, yname in enumerate(y_order):
    for c, xname in enumerate(x_order):

        ax = axes[r, c]
        d = curves[(xname, yname)]

        ax.plot(d["x_mid"].values, d["y_mean"].values)
        ax.set_title(titles[(xname, yname)])
        ax.set_xlabel(xlabels[xname])
        ax.set_ylabel(ylabels[yname])
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
