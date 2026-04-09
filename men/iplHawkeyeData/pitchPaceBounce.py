import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from paths import PROJECT_ROOT


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
]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(
    subset=[
        "matchid",
        "Phase",
        "Pre Bounce Velocity_x",
        "Pre Bounce Velocity_y",
        "Post Bounce Velocity_x",
        "Post Bounce Velocity_y",
        "Batter Runs",
        "Ovrexpr",
    ]
)







# -----------------------
# 2) Per-ball retention metrics
# -----------------------
df["pace_retention"] = (
    df["Post Bounce Velocity_x"] /
    df["Pre Bounce Velocity_x"]
)

df["bounce_retention"] = (
    df["Post Bounce Velocity_y"] /
    (df["Pre Bounce Velocity_y"])
)


df = df.dropna(
    subset=[
        "pace_retention",
        "bounce_retention",
    ]
)





# -----------------------
# 3) Remove 3 std outliers
# -----------------------
pace_mean = df["pace_retention"].mean()
pace_std = df["pace_retention"].std(ddof=0)

bounce_mean = df["bounce_retention"].mean()
bounce_std = df["bounce_retention"].std(ddof=0)

df = df[
    (df["pace_retention"] >= pace_mean - 3 * pace_std) &
    (df["pace_retention"] <= pace_mean + 3 * pace_std) &
    (df["bounce_retention"] >= bounce_mean - 3 * bounce_std) &
    (df["bounce_retention"] <= bounce_mean + 3 * bounce_std)
].copy()


# -----------------------
# 4) Remove bowler effects via bins (speed + angle + Phase)
# -----------------------
df["angle_in"] = np.arctan2(
    df["Pre Bounce Velocity_y"],
    df["Pre Bounce Velocity_x"]
)

df["v_pre_bin"] = pd.qcut(
    df["Pre Bounce Velocity_x"],
    q=4,
    labels=False,
    duplicates="drop",
)

df["angle_bin"] = pd.qcut(
    df["angle_in"],
    q=4,
    labels=False,
    duplicates="drop",
)

df["pace_bin_median"] = (
    df
    .groupby(["v_pre_bin", "angle_bin", "Phase"])["pace_retention"]
    .transform("median")
)

df["bounce_bin_median"] = (
    df
    .groupby(["v_pre_bin", "angle_bin", "Phase"])["bounce_retention"]
    .transform("median")
)

df["pitch_pace_ball"] = (
    df["pace_retention"] -
    df["pace_bin_median"]
)

df["pitch_bounce_ball"] = (
    df["bounce_retention"] -
    df["bounce_bin_median"]
)


# -----------------------
# 5) Match-level pivot (target + engineered features)
# -----------------------
pitch_pivot = (
    df
    .groupby(["matchid", "venue"], dropna=False)
    .agg(
        deliveries=("pitch_pace_ball", "size"),
        pitch_pace=("pitch_pace_ball", "median"),
        pitch_bounciness=("pitch_bounce_ball", "median"),
        ovrexpr=("Ovrexpr", "mean"),
        total_batter_runs=("Batter Runs", "sum"),
    )
    .reset_index()
)


pitch_pivot["avg_runs_per_ball"] = (
    pitch_pivot["total_batter_runs"] /
    pitch_pivot["deliveries"]
)


# -----------------------
# 6) Add raw-input match medians (for raw model)
# -----------------------
raw_cols = [
    "Pre Bounce Velocity_x",
    "Pre Bounce Velocity_y",
    "Post Bounce Velocity_x",
    "Post Bounce Velocity_y",
    "angle_in",
]

raw_match = (
    df
    .groupby("matchid")[raw_cols]
    .median()
    .reset_index()
)

pitch_pivot = pitch_pivot.merge(
    raw_match,
    on="matchid",
    how="left"
)


# -----------------------
# 7) One OOF CV loop producing base / engineered / raw predictions
# -----------------------
y = pitch_pivot["avg_runs_per_ball"].values

X_base = pitch_pivot[["ovrexpr"]].values
X_full = pitch_pivot[["ovrexpr", "pitch_pace", "pitch_bounciness"]].values
X_raw = pitch_pivot[["ovrexpr"] + raw_cols].values

kf = KFold(n_splits=20, shuffle=True, random_state=42)

oof_pred_base = np.full(len(pitch_pivot), np.nan, dtype=float)
oof_pred_full = np.full(len(pitch_pivot), np.nan, dtype=float)
oof_pred_raw = np.full(len(pitch_pivot), np.nan, dtype=float)
oof_fold = np.full(len(pitch_pivot), np.nan, dtype=float)

rows = []

for fold, (train_idx, test_idx) in enumerate(kf.split(pitch_pivot), start=1):
    oof_fold[test_idx] = fold

    y_train = y[train_idx]
    y_test = y[test_idx]

    # base = pure ovrexpr
    pred_base = X_base[test_idx].ravel()
    oof_pred_base[test_idx] = pred_base

    # engineered
    m_full = LinearRegression()
    m_full.fit(X_full[train_idx], y_train)
    pred_full = m_full.predict(X_full[test_idx])
    oof_pred_full[test_idx] = pred_full

    # raw inputs
    m_raw = LinearRegression()
    m_raw.fit(X_raw[train_idx], y_train)
    pred_raw = m_raw.predict(X_raw[test_idx])
    oof_pred_raw[test_idx] = pred_raw

    rows.append(
        {
            "fold": fold,

            "r2_base": r2_score(y_test, pred_base),
            "r2_engineered": r2_score(y_test, pred_full),
            "r2_raw_inputs": r2_score(y_test, pred_raw),

            "mae_base": mean_absolute_error(y_test, pred_base),
            "mae_engineered": mean_absolute_error(y_test, pred_full),
            "mae_raw_inputs": mean_absolute_error(y_test, pred_raw),
        }
    )


cv_results = pd.DataFrame(rows)

cv_results["r2_gain_eng_minus_base"] = cv_results["r2_engineered"] - cv_results["r2_base"]
cv_results["r2_gain_eng_minus_raw"] = cv_results["r2_engineered"] - cv_results["r2_raw_inputs"]

cv_results["mae_improve_eng_minus_base"] = cv_results["mae_base"] - cv_results["mae_engineered"]
cv_results["mae_improve_eng_minus_raw"] = cv_results["mae_raw_inputs"] - cv_results["mae_engineered"]


# -----------------------
# 8) Attach OOF preds + errors to pitch_pivot (Excel-friendly)
# -----------------------
pitch_pivot["cv_fold"] = oof_fold

pitch_pivot["oof_pred_base"] = oof_pred_base
pitch_pivot["oof_pred_engineered"] = oof_pred_full
pitch_pivot["oof_pred_raw_inputs"] = oof_pred_raw

pitch_pivot["oof_resid_base"] = pitch_pivot["avg_runs_per_ball"] - pitch_pivot["oof_pred_base"]
pitch_pivot["oof_resid_engineered"] = pitch_pivot["avg_runs_per_ball"] - pitch_pivot["oof_pred_engineered"]
pitch_pivot["oof_resid_raw_inputs"] = pitch_pivot["avg_runs_per_ball"] - pitch_pivot["oof_pred_raw_inputs"]

pitch_pivot["oof_abs_err_base"] = pitch_pivot["oof_resid_base"].abs()
pitch_pivot["oof_abs_err_engineered"] = pitch_pivot["oof_resid_engineered"].abs()
pitch_pivot["oof_abs_err_raw_inputs"] = pitch_pivot["oof_resid_raw_inputs"].abs()

pitch_pivot["oof_err_improve_eng_minus_base"] = (
    pitch_pivot["oof_abs_err_base"] -
    pitch_pivot["oof_abs_err_engineered"]
)

pitch_pivot["oof_err_improve_eng_minus_raw"] = (
    pitch_pivot["oof_abs_err_raw_inputs"] -
    pitch_pivot["oof_abs_err_engineered"]
)

pitch_pivot["cv_mean_r2_base"] = cv_results["r2_base"].mean()
pitch_pivot["cv_mean_r2_engineered"] = cv_results["r2_engineered"].mean()
pitch_pivot["cv_mean_r2_raw_inputs"] = cv_results["r2_raw_inputs"].mean()

pitch_pivot["cv_mean_mae_base"] = cv_results["mae_base"].mean()
pitch_pivot["cv_mean_mae_engineered"] = cv_results["mae_engineered"].mean()
pitch_pivot["cv_mean_mae_raw_inputs"] = cv_results["mae_raw_inputs"].mean()

pitch_pivot["cv_mean_r2_gain_eng_minus_base"] = cv_results["r2_gain_eng_minus_base"].mean()
pitch_pivot["cv_mean_r2_gain_eng_minus_raw"] = cv_results["r2_gain_eng_minus_raw"].mean()

pitch_pivot["cv_mean_mae_improve_eng_minus_base"] = cv_results["mae_improve_eng_minus_base"].mean()
pitch_pivot["cv_mean_mae_improve_eng_minus_raw"] = cv_results["mae_improve_eng_minus_raw"].mean()


# -----------------------
# 9) Prints
# -----------------------
print(cv_results)
print()

print("Mean R2 base:", cv_results["r2_base"].mean())
print("Mean R2 engineered:", cv_results["r2_engineered"].mean())
print("Mean R2 raw inputs:", cv_results["r2_raw_inputs"].mean())
print("Mean R2 gain (eng - base):", cv_results["r2_gain_eng_minus_base"].mean())
print("Mean R2 gain (eng - raw):", cv_results["r2_gain_eng_minus_raw"].mean())
print()

print("Mean MAE base:", cv_results["mae_base"].mean())
print("Mean MAE engineered:", cv_results["mae_engineered"].mean())
print("Mean MAE raw inputs:", cv_results["mae_raw_inputs"].mean())
print("Mean MAE improve (eng - base):", cv_results["mae_improve_eng_minus_base"].mean())
print("Mean MAE improve (eng - raw):", cv_results["mae_improve_eng_minus_raw"].mean())



venues = pd.pivot_table(df, values=['pitch_pace_ball', 'pitch_bounce_ball'], index=['venue'], aggfunc=['mean', 'count']).reset_index()
bowler_types = pd.pivot_table(df, values=['venue'], index=['Bowler Style'], aggfunc=['count']).reset_index()

