import pandas as pd
import numpy as np
from sklearn import preprocessing
import statsmodels.api as sm
from paths import PROJECT_ROOT


def make_ohe(values, cats, prefix):
    encoder = preprocessing.OneHotEncoder(sparse_output=False, categories=[cats], drop='first', handle_unknown='ignore')
    encoded = encoder.fit_transform(values)
    columns = [f'{prefix}__{cat}' for cat in cats[1:]]

    return pd.DataFrame(encoded, columns=columns)


def rep_weight(faced, rating, rep_ratio, mode='run'):
    # Constants optimised for the basic reversion used for oppo param
    if mode == 'run':
        k, a, x, y = 0.001296, 0.611901, 0.000757, 0.02
    else:
        k, a, x, y = 0.000942, 0.8874, 0.000957, 0.02

    weight = np.maximum(y, np.maximum((1 - k) ** faced, a - (x * faced)))
    rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)

    return weight, rating_2


def build_training_features_bat(bat_data, transformers):
    # Competition encodings
    is_global_comp = bat_data['competition'].isin(['WT20I', 'tier_2'])
    competition = np.where(is_global_comp, bat_data['competition'], bat_data['competition'] + ' ' + bat_data['H/A_competition']).reshape(-1, 1)
    competition_cats = sorted(np.unique(competition).tolist())
    transformers['competition_cats'] = competition_cats
    competition_encodings = make_ohe(competition, competition_cats, 'competition')

    # Nationality encodings
    nationality = np.array(bat_data['battingteam']).reshape(-1, 1)
    nationality_cats = ['England', 'India', 'Australia', 'New Zealand', 'South Africa']
    transformers['nationality_cats'] = nationality_cats
    nationality_encodings = make_ohe(nationality, nationality_cats, 'nationality')

    # Age poly
    age = pd.DataFrame(bat_data.loc[:, ['age']])
    age_poly = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(age), columns=['age_x', 'age_x^2']).reset_index(drop=True)

    # Order poly
    ord_col = 'ord_r' if 'ord_r' in bat_data.columns else ('ord_w' if 'ord_w' in bat_data.columns else 'ord')
    order = pd.DataFrame(bat_data.loc[:, [ord_col]])
    order_poly = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(order), columns=['order_x', 'order_x^2']).reset_index(drop=True)

    # Experience transform
    experience = pd.DataFrame(bat_data.loc[:, ['balls_faced_career']])
    transformer = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    transformer.fit(experience)
    transformers['experience_transformer'] = transformer
    experience = pd.DataFrame(transformer.transform(experience), columns=['experience']).reset_index(drop=True)

    # Oppo feature RUN
    oppo_run = bat_data['wkt_rating_2'].copy()
    mask = bat_data[ord_col] > 7
    oppo_run[mask] = (((1 - oppo_run[mask]) / 2) * np.minimum(2, abs(bat_data.loc[mask, ord_col] - 7))) + oppo_run[mask]
    oppo_run = pd.DataFrame(oppo_run).reset_index(drop=True)
    oppo_run.columns = ['oppo']

    # Oppo feature WKT
    oppo_wkt = bat_data['run_rating_2'].copy()
    mask = bat_data[ord_col] > 7
    oppo_wkt[mask] = (((1 - oppo_wkt[mask]) / 2) * np.minimum(2, abs(bat_data.loc[mask, ord_col] - 7))) + oppo_wkt[mask]
    oppo_wkt = pd.DataFrame(oppo_wkt).reset_index(drop=True)
    oppo_wkt.columns = ['oppo']

    X_run = pd.concat([competition_encodings, nationality_encodings, age_poly, experience, order_poly, oppo_run], axis=1)
    X_wkt = pd.concat([competition_encodings, nationality_encodings, age_poly, experience, order_poly, oppo_wkt], axis=1)

    return bat_data, X_run, X_wkt, transformers


def build_ratings_features_bat(ratings, transformers):
    # Competition encodings
    is_global_comp = ratings['competition'].isin(['WT20I', 'tier_2'])
    competition = np.where(is_global_comp, ratings['competition'], ratings['competition'] + ' ' + ratings['H/A_competition']).reshape(-1, 1)
    competition_encodings = make_ohe(competition, transformers['competition_cats'], 'competition')

    # Nationality encodings
    nationality = np.array(ratings['battingteam']).reshape(-1, 1)
    nationality_encodings = make_ohe(nationality, transformers['nationality_cats'], 'nationality')

    # Age poly
    age = pd.DataFrame(ratings.loc[:, ['age']])
    age_poly = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(age), columns=['age_x', 'age_x^2']).reset_index(drop=True)

    # Order poly
    ord_col = 'ord_r' if 'ord_r' in ratings.columns else ('ord_w' if 'ord_w' in ratings.columns else 'ord')
    order = pd.DataFrame(ratings.loc[:, [ord_col]])
    order_poly = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(order), columns=['order_x', 'order_x^2']).reset_index(drop=True)

    # Experience transform
    experience = pd.DataFrame(ratings.loc[:, ['balls_faced_career']])
    experience = pd.DataFrame(transformers['experience_transformer'].transform(experience), columns=['experience']).reset_index(drop=True)

    # Oppo feature RUN
    oppo_run = ratings['wkt_rating_2'].copy()
    mask = ratings[ord_col] > 7
    oppo_run[mask] = (((1 - oppo_run[mask]) / 2) * np.minimum(2, abs(ratings.loc[mask, ord_col] - 7))) + oppo_run[mask]
    oppo_run = pd.DataFrame(oppo_run).reset_index(drop=True)
    oppo_run.columns = ['oppo']

    # Oppo feature WKT
    oppo_wkt = ratings['run_rating_2'].copy()
    mask = ratings[ord_col] > 7
    oppo_wkt[mask] = (((1 - oppo_wkt[mask]) / 2) * np.minimum(2, abs(ratings.loc[mask, ord_col] - 7))) + oppo_wkt[mask]
    oppo_wkt = pd.DataFrame(oppo_wkt).reset_index(drop=True)
    oppo_wkt.columns = ['oppo']

    X_run = pd.concat([competition_encodings, nationality_encodings, age_poly, experience, order_poly, oppo_run], axis=1)
    X_wkt = pd.concat([competition_encodings, nationality_encodings, age_poly, experience, order_poly, oppo_wkt], axis=1)

    return ratings, X_run, X_wkt


for x in np.arange(0, 2, 1):

    # -------------------------
    # 1) Imports
    # -------------------------
    bat_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/batDataCombinedClean_w.csv', parse_dates=['date', 'dob'])
    n2h_factors = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/batN2HFactors_w.csv')

    if x == 0:
        ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle_w.csv', parse_dates=['date'], dtype={'battingteam': str})
    else:
        ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsRasoi_w.csv', parse_dates=['date'], dtype={'battingteam': str})

    # -------------------------
    # 2) Filters
    # -------------------------
    bat_data = bat_data.loc[bat_data['format'] == 't20', :].copy()

    # -------------------------
    # 3) Base cleaning / merges
    # -------------------------
    bat_data['competition'] = bat_data['competition'].str.replace(r"\\'", "'", regex=True)
    ratings['competition'] = ratings['competition'].str.replace(r"\\'", "'", regex=True)
    n2h_factors['host'] = n2h_factors['host'].str.replace(r"\\'", "'", regex=True)

    allaway_runs = n2h_factors['all_away_runs_factor'].mean()
    allaway_wkts = n2h_factors['all_away_wkts_factor'].mean()

    bat_data = bat_data.merge(n2h_factors, on=('nationality', 'host'), how='left')
    bat_data['run_factor'] = bat_data['run_factor'].fillna(allaway_runs)
    bat_data['wkt_factor'] = bat_data['wkt_factor'].fillna(allaway_wkts)

    bat_data = bat_data[bat_data['balls_faced'] > 0].copy()

    bat_data['balls_faced_career'] = bat_data['balls_faced_career'] + 5
    ratings['balls_faced_career'] = ratings['balls_faced_career'] + 5

    bat_data = bat_data.merge(ratings[ratings['i_balls_faced'] > 0].loc[:, ['playerid', 'date', 'run_rating', 'wkt_rating']], how='left', on=['playerid', 'date'])

    # -------------------------
    # 4) Basic reversion for oppo ratings
    # -------------------------
    ovr_run_ratio = bat_data['runs'].sum() / bat_data['realexprbat'].sum()
    ovr_wkt_ratio = bat_data['wkt'].sum() / bat_data['realexpwbat'].sum()

    bat_data.insert(bat_data.columns.get_loc('run_rating') + 1, 'run_rating_2', rep_weight(bat_data['balls_faced_career'], bat_data['run_rating'], ovr_run_ratio, 'run')[1])
    bat_data.insert(bat_data.columns.get_loc('wkt_rating') + 1, 'wkt_rating_2', rep_weight(bat_data['balls_faced_career'], bat_data['wkt_rating'], ovr_wkt_ratio, 'wkt')[1])

    ratings.insert(ratings.columns.get_loc('run_rating') + 1, 'run_rating_2', rep_weight(ratings['balls_faced_career'], ratings['run_rating'], ovr_run_ratio, 'run')[1])
    ratings.insert(ratings.columns.get_loc('wkt_rating') + 1, 'wkt_rating_2', rep_weight(ratings['balls_faced_career'], ratings['wkt_rating'], ovr_wkt_ratio, 'wkt')[1])

    # -------------------------
    # 5) Fit RUN + WKT models
    # -------------------------
    transformers = {}
    bat_data, X_run, X_wkt, transformers = build_training_features_bat(bat_data, transformers)

    y = pd.DataFrame(bat_data['run_ratio'])
    rep_run_ratio_model = sm.OLS(y, X_run, missing='drop').fit()
    run_params = rep_run_ratio_model.params.copy()

    y = pd.DataFrame(bat_data['wkt_ratio'])
    rep_wkt_ratio_model = sm.OLS(y, X_wkt, missing='drop').fit()
    wkt_params = rep_wkt_ratio_model.params.copy()

    # -------------------------
    # 6) Predict training data
    # -------------------------
    bat_data['rep_run_ratio'] = X_run.to_numpy() @ run_params.to_numpy()
    bat_data['rep_run_ratio'] = np.where((bat_data['competition'].isin(['WT20I'])) & (bat_data['H/A_competition'] == 'Away'), bat_data['rep_run_ratio'] * np.minimum(1, bat_data['run_factor'] / allaway_runs), bat_data['rep_run_ratio'])
    bat_data['rep_runs'] = bat_data['rep_run_ratio'] * bat_data['realexprbat']

    bat_data['rep_wkt_ratio'] = X_wkt.to_numpy() @ wkt_params.to_numpy()
    bat_data['rep_wkt_ratio'] = np.where((bat_data['competition'].isin(['WT20I'])) & (bat_data['H/A_competition'] == 'Away'), bat_data['rep_wkt_ratio'] * np.maximum(1, bat_data['wkt_factor'] / allaway_wkts), bat_data['rep_wkt_ratio'])
    bat_data['rep_wkt'] = bat_data['rep_wkt_ratio'] * bat_data['realexpwbat']

    # -------------------------
    # 7) Predict ratings outputs
    # -------------------------
    ratings['age'] = ratings['age'].fillna(bat_data['age'].median())

    ratings = ratings.merge(n2h_factors, on=('nationality', 'host'), how='left')
    ratings['run_factor'] = ratings['run_factor'].fillna(allaway_runs)
    ratings['wkt_factor'] = ratings['wkt_factor'].fillna(allaway_wkts)

    ratings, X_run_r, X_wkt_r = build_ratings_features_bat(ratings, transformers)

    ratings.insert(ratings.columns.get_loc('run_rating_2') + 1, 'rep_run_ratio', X_run_r.to_numpy() @ run_params.to_numpy())
    ratings['rep_run_ratio'] = np.where((ratings['competition'].isin(['WT20I'])) & (ratings['H/A_competition'] == 'Away'), ratings['rep_run_ratio'] * np.minimum(1, ratings['run_factor'] / allaway_runs), ratings['rep_run_ratio'])
    ratings['i_rep_runs'] = ratings['rep_run_ratio'] * ratings['i_realexprbat']

    ratings.insert(ratings.columns.get_loc('wkt_rating_2') + 1, 'rep_wkt_ratio', X_wkt_r.to_numpy() @ wkt_params.to_numpy())
    ratings['rep_wkt_ratio'] = np.where((ratings['competition'].isin(['WT20I'])) & (ratings['H/A_competition'] == 'Away'), ratings['rep_wkt_ratio'] * np.maximum(1, ratings['wkt_factor'] / allaway_wkts), ratings['rep_wkt_ratio'])
    ratings['i_rep_wkt'] = ratings['rep_wkt_ratio'] * ratings['i_realexpwbat']

    # -------------------------
    # 8) One-player breakdown
    # -------------------------
    if x == 1:
        pass
    else:
        batsman = 'Izzy Gaze'
        competition = 'Women\'s Big Bash League'
        host = 'Australia'
        debug_mask = (
            (ratings['batsman'] == batsman) &
            (ratings['competition'] == competition) &
            (ratings['host'] == host)
        )

        debug_rows = ratings.loc[debug_mask, :].reset_index(drop=True)
        debug_X_run = X_run_r.loc[debug_mask, :].reset_index(drop=True)
        debug_X_wkt = X_wkt_r.loc[debug_mask, :].reset_index(drop=True)

        if len(debug_rows) > 0:
            run_contribs = pd.DataFrame(debug_X_run.to_numpy() * run_params.to_numpy(), columns=debug_X_run.columns)
            wkt_contribs = pd.DataFrame(debug_X_wkt.to_numpy() * wkt_params.to_numpy(), columns=debug_X_wkt.columns)

            print('\nRUN RATING BREAKDOWN')
            print('=' * 80)

            for col, val in run_contribs.iloc[0].sort_values(key=lambda z: z.abs(), ascending=False).items():
                print(f'{str(col):<65} {val:>12.6f}')

            print('-' * 80)
            print(f'TOTAL RUN REPLACEMENT RATIO: {debug_rows.loc[0, "rep_run_ratio"]:.6f}')

            print('\nWKT RATING BREAKDOWN')
            print('=' * 80)

            for col, val in wkt_contribs.iloc[0].sort_values(key=lambda z: z.abs(), ascending=False).items():
                print(f'{str(col):<65} {val:>12.6f}')

            print('-' * 80)
            print(f'TOTAL WKT REPLACEMENT RATIO: {debug_rows.loc[0, "rep_wkt_ratio"]:.6f}')

            print(debug_rows[['batsman', 'competition', 'H/A_competition', 'host', 'wkt_factor', 'rep_wkt_ratio']])
            print('raw model:', (debug_X_wkt.to_numpy() @ wkt_params.to_numpy())[0])
            print('adjustment:', np.maximum(1, debug_rows.loc[0, 'wkt_factor'] / allaway_wkts))


    # -------------------------
    # 9) Checks + pivots
    # -------------------------
    test = ratings.copy()

    test['sum_rep_r'] = test['rep_run_ratio'] * test['i_balls_faced']
    test['sum_rep_w'] = test['rep_wkt_ratio'] * test['i_balls_faced']

    sum_rep_r = test['sum_rep_r'].sum()
    sum_rep_w = test['sum_rep_w'].sum()
    sum_balls = test['i_balls_faced'].sum()

    rep_r_o = sum_rep_r / sum_balls
    rep_w_o = sum_rep_w / sum_balls

    bat_data['run_sqe'] = (bat_data['run_ratio'] - bat_data['rep_run_ratio']) ** 2
    bat_data['wkt_sqe'] = (bat_data['wkt_ratio'] - bat_data['rep_wkt_ratio']) ** 2
    bat_data['run_err'] = bat_data['rep_runs'] - bat_data['runs']
    bat_data['wkt_err'] = bat_data['rep_wkt'] - bat_data['wkt']

    bat_data['balls_faced_career_round'] = (bat_data['balls_faced_career'] / 500).round().astype(int) * 500

    bat_data['age_round'] = (bat_data['age'] / 2).round().astype(int) * 2
    bat_data['age_round'] = np.clip(bat_data['age_round'], 18, 42)

    bat_data['run_rating_round'] = (bat_data['run_rating_2'] / 0.05).round() * 0.05
    bat_data['wkt_rating_round'] = (bat_data['wkt_rating_2'] / 0.05).round() * 0.05

    bat_data['count'] = 1

    actuals = pd.pivot_table(bat_data,
                             values=['balls_faced_innings', 'realexprbat', 'rep_runs', 'runs', 'realexpwbat', 'rep_wkt', 'wkt', 'rep_wkt_ratio', 'rep_run_ratio', 'age', 'balls_faced_career',
                                     'run_sqe', 'wkt_sqe', 'run_err', 'wkt_err'],
                             index=['H/A_competition', 'competition'],
                             aggfunc={'balls_faced_innings': 'count', 'balls_faced_career': 'mean', 'age': 'mean', 'realexprbat': 'sum', 'rep_runs': 'sum', 'runs': 'sum', 'realexpwbat': 'sum',
                                      'rep_wkt': 'sum', 'wkt': 'sum', 'rep_run_ratio': 'mean', 'rep_wkt_ratio': 'mean', 'run_sqe': 'mean', 'wkt_sqe': 'mean', 'run_err': 'sum', 'wkt_err': 'sum'}).reset_index()

    actuals['run_ratio'] = actuals['runs'] / actuals['realexprbat']
    actuals['wkt_ratio'] = actuals['wkt'] / actuals['realexpwbat']

    actuals_ratings = ratings.copy()
    actuals_ratings = actuals_ratings[actuals_ratings.matchid > 0].copy()

    actuals_ratings = pd.pivot_table(actuals_ratings, values=['i_balls_faced', 'i_realexprbat', 'i_rep_runs', 'i_runs', 'i_realexpwbat', 'i_rep_wkt', 'i_wkt', 'rep_wkt_ratio', 'rep_run_ratio'], index=['competition'], aggfunc={'i_balls_faced': 'sum', 'i_realexprbat': 'sum', 'i_rep_runs': 'sum', 'i_runs': 'sum', 'i_realexpwbat': 'sum', 'i_rep_wkt': 'sum', 'i_wkt': 'sum', 'rep_run_ratio': 'mean', 'rep_wkt_ratio': 'mean'}).reset_index()
    actuals_ratings['run_ratio'] = actuals_ratings['i_runs'] / actuals_ratings['i_realexprbat']
    actuals_ratings['wkt_ratio'] = actuals_ratings['i_wkt'] / actuals_ratings['i_realexpwbat']

    # -------------------------
    # 10) Export
    # -------------------------
    if x == 0:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle2_w.csv', index=False)
    else:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsRasoi2_w.csv', index=False)


ratings = ratings[ratings['batsman'] == 'Izzy Gaze']
ratings = ratings.loc[:, ['batsman', 'competition', 'H/A_competition', 'host', 'wkt_factor', 'rep_wkt_ratio']]

