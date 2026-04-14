import numpy as np
import pandas as pd
from batFunctions import qualityMethodBins, newMethodBins
from paths import PROJECT_ROOT


# --- Load Data ---
ratingsJungle = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsJungle3.csv')
ratingsRasoi = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsRasoi3.csv')
bat_data = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/data/combinedBatDataClean.csv', parse_dates=['date', 'dob'])

# --- Load and merge weights ---
bat_weightings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batWeightings.csv')
bat_data = bat_data.merge(bat_weightings, on='balls_faced_career', how='left')

# bat_data['runs_weight_curve'] = bat_data['runs_weight_curve'].fillna(1)
# bat_data['wkts_weight_curve'] = bat_data['wkts_weight_curve'].fillna(1)

bat_data['runs_weight_curve'] = 1
bat_data['wkts_weight_curve'] = 1

# Calculate overall baseline ratios and add them as columns to bat_data
ovr_run_ratio_val = bat_data['runs'].sum() / bat_data['realexprbat'].sum()
ovr_wkt_ratio_val = bat_data['wkt'].sum() / bat_data['realexpwbat'].sum()

bat_data['ovr_run_ratio'] = ovr_run_ratio_val
bat_data['ovr_wkt_ratio'] = ovr_wkt_ratio_val

def score_cols_cross_methods(
        bat_data: pd.DataFrame,
        ratings_jungle: pd.DataFrame,
        ratings_rasoi: pd.DataFrame,
        run_rating_cols: list[str],
        wkt_rating_cols: list[str],
        bin_size: int = 40,
        min_balls_per_bin: int = 30,
) -> pd.DataFrame:
    """
    Scores Jungle outputs using jungleMethod and Rasoi outputs using rasoiMethod.
    Bins are fixed based on the final reverted rating (rating_3) to ensure consistency.
    """
    ratings_jungle['date'] = pd.to_datetime(ratings_jungle['date'])
    ratings_rasoi['date'] = pd.to_datetime(ratings_rasoi['date'])

    out_rows = []

    configs = [
        {
            'method': 'jungleMethod',
            'outputs': ratings_jungle,
            'target': 'runs',
            'rating_cols': run_rating_cols,
            'primary_rating_col': 'run_rating_3',
            'exp_col': 'realexprbat',
            'actual_col': 'runs',
            'weight_curve_col': 'runs_weight_curve'
        },
        {
            'method': 'jungleMethod',
            'outputs': ratings_jungle,
            'target': 'wickets',
            'rating_cols': wkt_rating_cols,
            'primary_rating_col': 'wkt_rating_3',
            'exp_col': 'realexpwbat',
            'actual_col': 'wkt',
            'weight_curve_col': 'wkts_weight_curve'
        },
        {
            'method': 'rasoiMethod',
            'outputs': ratings_rasoi,
            'target': 'runs',
            'rating_cols': run_rating_cols,
            'primary_rating_col': 'run_rating_3',
            'exp_col': 'realexprbat',
            'actual_col': 'runs',
            'weight_curve_col': 'runs_weight_curve'
        },
        {
            'method': 'rasoiMethod',
            'outputs': ratings_rasoi,
            'target': 'wickets',
            'rating_cols': wkt_rating_cols,
            'primary_rating_col': 'wkt_rating_3',
            'exp_col': 'realexpwbat',
            'actual_col': 'wkt',
            'weight_curve_col': 'wkts_weight_curve'
        },
    ]

    for config in configs:
        method = config['method']
        ratings_df = config['outputs']
        rating_cols = config['rating_cols']
        primary_col = config['primary_rating_col']
        exp_col = config['exp_col']
        actual_col = config['actual_col']
        target = config['target']
        weight_curve_col = config['weight_curve_col']

        # Only merge columns from the outputs files that aren't already in bat_data
        cols_to_merge = [c for c in rating_cols if c not in bat_data.columns]

        bat_i = bat_data.merge(
            ratings_df.loc[:, ['playerid', 'date', 'matchid'] + cols_to_merge]
            .drop_duplicates(subset=['playerid', 'date', 'matchid']),
            how='left',
            on=['playerid', 'date', 'matchid'],
        )

        # 1. Cleaning & Binning (Done ONCE per config using the primary_col)
        df_base = bat_i.dropna(subset=[primary_col, exp_col, actual_col, weight_curve_col]).copy()
        df_base = df_base[df_base['balls_faced'] > 0]

        if method == 'jungleMethod':
            df_base = qualityMethodBins(
                df_base,
                bin_size=bin_size,
                rating_col=primary_col,
                out_col='binid',
            )
        elif method == 'rasoiMethod':
            df_base = newMethodBins(df_base)

        # 2. Iterate through all columns using the fixed binid from step 1
        for rating_col in rating_cols:
            df = df_base.dropna(subset=[rating_col]).copy()

            # Row-level prediction
            df['pred'] = df[exp_col] * df[rating_col]
            df['pred_w'] = df['pred'] * df[weight_curve_col]
            df['act_w'] = df[actual_col] * df[weight_curve_col]

            pivot = (
                df.groupby('binid', as_index=False)
                .agg(
                    balls_faced=('balls_faced', 'sum'),
                    pred_w_sum=('pred_w', 'sum'),
                    act_w_sum=('act_w', 'sum'),
                )
            )
            pivot = pivot[pivot['balls_faced'] > min_balls_per_bin]

            residual = (pivot['pred_w_sum'] - pivot['act_w_sum']).to_numpy(dtype=float)

            if len(residual) > 0:
                rmse = float(np.sqrt(np.mean(residual ** 2)))
                mae = float(np.mean(np.abs(residual)))
                sse_mean = float(np.mean(residual ** 2))
                sse_total = float(np.sum(residual ** 2))
            else:
                rmse = mae = sse_mean = sse_total = np.nan

            out_rows.append({
                'target': target,
                'method': method,
                'rating_col': rating_col,
                'bin_size': int(bin_size) if method == 'jungleMethod' else np.nan,
                'min_balls_per_bin': int(min_balls_per_bin),
                'n_bins_used': int(pivot.shape[0]),
                'rmse': rmse,
                'mae': mae,
                'sse_mean': sse_mean,
                'sse_total': sse_total,
            })

    return pd.DataFrame(out_rows)


# --- Execution ---
run_rating_cols = ['ovr_run_ratio', 'z_run_ratio', 'run_rating_0', 'run_rating', 'run_rating_2', 'rep_run_ratio', 'run_rating_3']
wkt_rating_cols = ['ovr_wkt_ratio', 'z_wkt_ratio', 'wkt_rating_0', 'wkt_rating', 'wkt_rating_2', 'rep_wkt_ratio', 'wkt_rating_3']

errors_df = score_cols_cross_methods(
    bat_data=bat_data,
    ratings_jungle=ratingsJungle,
    ratings_rasoi=ratingsRasoi,
    run_rating_cols=run_rating_cols,
    wkt_rating_cols=wkt_rating_cols,
    bin_size=40,
    min_balls_per_bin=30,
)


