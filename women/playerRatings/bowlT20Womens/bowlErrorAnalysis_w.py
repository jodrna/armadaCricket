import numpy as np
import pandas as pd
from bowlFunctions_w import qualityMethodBins, newMethodBins
from paths import PROJECT_ROOT


# -------------------------
# Load data
# -------------------------
ratingsJungle = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle3_w.csv', parse_dates=['date'])
ratingsRasoi = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsRasoi3_w.csv', parse_dates=['date'])
bowl_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/bowlDataCombinedClean_w.csv', parse_dates=['date', 'dob'])
bowl_weightings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/auxiliaries/bowlWeightings_w.csv')


# -------------------------
# Match optimiser
# -------------------------
bowl_data['competition'] = np.where(bowl_data['format'] == 'odi', np.where(bowl_data['ballsremaining'] < 84, 'ODI2', 'ODI1'), bowl_data['competition'])
bowl_data['format'] = bowl_data['format'].fillna('t20')

bowl_data = bowl_data.merge(bowl_weightings, on='balls_bowled_career', how='left')
bowl_data['runs_weight_curve'] = bowl_data['runs_weight_curve'].fillna(1)
bowl_data['wkts_weight_curve'] = bowl_data['wkts_weight_curve'].fillna(1)


# -------------------------
# Overall
# -------------------------
bowl_data['ovr_run_ratio'] = bowl_data['runs'].sum() / bowl_data['realexprbowl'].sum()
bowl_data['ovr_wkt_ratio'] = bowl_data['wkt'].sum() / bowl_data['realexpwbowl'].sum()


# -------------------------
# Scoring
# -------------------------
def score_cols(
        bowl_data,
        ratings_df,
        rating_cols,
        target,
        exp_col,
        actual_col,
        weight_curve_col,
        output_method,
        quality_bin_rating_col=None,
        bin_size=60,
        min_balls_per_bin=30,
):
    out_rows = []

    merge_keys = ['playerid', 'matchid', 'host', 'date', 'competition']

    rating_cols_to_merge = [
        c for c in rating_cols
        if c in ratings_df.columns and c not in bowl_data.columns
    ]

    if quality_bin_rating_col is not None:
        if quality_bin_rating_col in ratings_df.columns and quality_bin_rating_col not in rating_cols_to_merge:
            rating_cols_to_merge.append(quality_bin_rating_col)

    df_base = bowl_data.merge(
        ratings_df.loc[:, merge_keys + rating_cols_to_merge].drop_duplicates(merge_keys),
        how='left',
        on=merge_keys,
    )

    df_base = df_base[(df_base['competition'] != 'ODI') & (df_base['balls_bowled'] > 0)].copy()

    if output_method == 'jungleMethod':
        df_method = qualityMethodBins(
            df_base.dropna(subset=[quality_bin_rating_col]).copy(),
            bin_size=bin_size,
            rating_col=quality_bin_rating_col,
            out_col='binid',
        )
        binning_method = 'qualityMethod'

    elif output_method == 'rasoiMethod':
        df_method = newMethodBins(df_base.copy())
        binning_method = 'newMethod'

    else:
        raise ValueError("output_method must be either 'jungleMethod' or 'rasoiMethod'")

    for rating_col in rating_cols:
        df = df_method.dropna(subset=[rating_col, actual_col, exp_col, weight_curve_col]).copy()

        df['pred'] = df[exp_col] * df[rating_col]
        df['pred_w'] = df['pred'] * df[weight_curve_col]
        df['act_w'] = df[actual_col] * df[weight_curve_col]

        pivot = (
            df.groupby('binid', as_index=False)
            .agg(
                rating_avg=(rating_col, 'mean'),
                balls_bowled=('balls_bowled', 'sum'),
                exp_sum=(exp_col, 'sum'),
                pred_sum=('pred', 'sum'),
                actual_sum=(actual_col, 'sum'),
                weight_curve_avg=(weight_curve_col, 'mean'),
                pred_w=('pred_w', 'sum'),
                act_w=('act_w', 'sum'),
            )
            .assign(bin_residual=lambda x: x['pred_w'] - x['act_w'])
            .sort_values('binid')
        )

        pivot = pivot[pivot['balls_bowled'] > min_balls_per_bin]

        residual = pivot['bin_residual'].to_numpy(dtype=float)

        if len(residual) > 0:
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            mae = float(np.mean(np.abs(residual)))
            sse_mean = float(np.mean(residual ** 2))
            sse_total = float(np.sum(residual ** 2))
        else:
            rmse = np.nan
            mae = np.nan
            sse_mean = np.nan
            sse_total = np.nan

        out_rows.append({
            'output_method': output_method,
            'target': target,
            'binning_method': binning_method,
            'rating_col': rating_col,
            'quality_bin_rating_col': quality_bin_rating_col if output_method == 'jungleMethod' else np.nan,
            'bin_size': bin_size if output_method == 'jungleMethod' else np.nan,
            'min_balls_per_bin': min_balls_per_bin,
            'n_bins_used': int(pivot.shape[0]),
            'balls_used': float(pivot['balls_bowled'].sum()),
            'rmse': rmse,
            'mae': mae,
            'sse_mean': sse_mean,
            'sse_total': sse_total,
        })

    return pd.DataFrame(out_rows)


# -------------------------
# Columns
# -------------------------
run_rating_cols = ['ovr_run_ratio', 'z_run_ratio', 'run_rating_0', 'run_rating', 'rep_run_ratio', 'run_rating_3']
wkt_rating_cols = ['ovr_wkt_ratio', 'z_wkt_ratio', 'wkt_rating_0', 'wkt_rating', 'rep_wkt_ratio', 'wkt_rating_3']


# -------------------------
# Score Jungle using quality method
# -------------------------
jungle_run_errors = score_cols(
    bowl_data=bowl_data,
    ratings_df=ratingsJungle,
    rating_cols=run_rating_cols,
    target='runs',
    exp_col='realexprbowl',
    actual_col='runs',
    weight_curve_col='runs_weight_curve',
    output_method='jungleMethod',
    quality_bin_rating_col='run_rating_3',
    bin_size=60,
    min_balls_per_bin=30,
)

jungle_wkt_errors = score_cols(
    bowl_data=bowl_data,
    ratings_df=ratingsJungle,
    rating_cols=wkt_rating_cols,
    target='wickets',
    exp_col='realexpwbowl',
    actual_col='wkt',
    weight_curve_col='wkts_weight_curve',
    output_method='jungleMethod',
    quality_bin_rating_col='wkt_rating_3',
    bin_size=60,
    min_balls_per_bin=30,
)


# -------------------------
# Score Rasoi using new method
# -------------------------
rasoi_run_errors = score_cols(
    bowl_data=bowl_data,
    ratings_df=ratingsRasoi,
    rating_cols=run_rating_cols,
    target='runs',
    exp_col='realexprbowl',
    actual_col='runs',
    weight_curve_col='runs_weight_curve',
    output_method='rasoiMethod',
    quality_bin_rating_col=None,
    bin_size=60,
    min_balls_per_bin=30,
)

rasoi_wkt_errors = score_cols(
    bowl_data=bowl_data,
    ratings_df=ratingsRasoi,
    rating_cols=wkt_rating_cols,
    target='wickets',
    exp_col='realexpwbowl',
    actual_col='wkt',
    weight_curve_col='wkts_weight_curve',
    output_method='rasoiMethod',
    quality_bin_rating_col=None,
    bin_size=60,
    min_balls_per_bin=30,
)


# -------------------------
# Final output
# -------------------------
errors_df = pd.concat(
    [
        jungle_run_errors,
        jungle_wkt_errors,
        rasoi_run_errors,
        rasoi_wkt_errors,
    ],
    axis=0,
).reset_index(drop=True)

errors_df = errors_df.loc[:, [
    'output_method',
    'target',
    'binning_method',
    'rating_col',
    'quality_bin_rating_col',
    'bin_size',
    'min_balls_per_bin',
    'n_bins_used',
    'balls_used',
    'rmse',
    'mae',
    'sse_mean',
    'sse_total',
]]


