import pandas as pd
from sklearn import preprocessing
import numpy as np
import statsmodels.api as sm
from paths import PROJECT_ROOT



def rep_weight(faced, rating, rep_ratio, mode='run'):
    # Constants optimised for the basic reversion used for oppo param
    if mode == 'run':
        k, a, x, y = 0.001296, 0.611901, 0.000757, 0.02
    else:
        k, a, x, y = 0.000942, 0.8874, 0.000957, 0.02

    weight = np.maximum(y, np.maximum((1 - k) ** faced, a - (x * faced)))
    rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)
    return weight, rating_2


def build_comp_index(df):
    is_global_comp = (df['competition'] == 'T20I') | (df['competition'] == 'tier_2')
    comp_str = np.where(is_global_comp,
                        df['competition'],
                        df['competition'] + " " + df['H/A_competition'])
    return comp_str


def build_features(df, transformers, is_training=True, target_type='run', comp_cats=None):
    """
    Constructs the X matrix for the model.
    """

    # 1. Competition Encodings
    comp_str = build_comp_index(df).reshape(-1, 1)

    if is_training:
        if comp_cats is None:
            comp_cats = list(np.unique(comp_str).reshape(1, -1))

        transformers['comp_cats'] = comp_cats
        transformers['comp_enc'] = preprocessing.OneHotEncoder(sparse_output=False, categories=comp_cats, handle_unknown='ignore')
        comp_encoded = transformers['comp_enc'].fit_transform(comp_str)
    else:
        comp_encoded = transformers['comp_enc'].transform(comp_str)

    df_comp = pd.DataFrame(comp_encoded, columns=transformers['comp_cats'])

    # 2. Nationality Encodings
    nat_str = np.array(df['battingteam']).reshape(-1, 1)

    nat_cats = ['England', 'India', 'Afghanistan', 'Australia', 'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'South Africa', 'Pakistan']

    if is_training:
        transformers['nat_cats'] = nat_cats
        transformers['nat_enc'] = preprocessing.OneHotEncoder(sparse_output=False, categories=[nat_cats], handle_unknown='ignore')
        nat_encoded = transformers['nat_enc'].fit_transform(nat_str)
    else:
        nat_encoded = transformers['nat_enc'].transform(nat_str)

    df_nat = pd.DataFrame(nat_encoded, columns=transformers['nat_cats'])

    # 3. Polynomial Features (Age, Order, Overseas)
    poly = preprocessing.PolynomialFeatures(degree=2)

    df_age = pd.DataFrame(poly.fit_transform(df[['age']]), columns=['age_constant', 'age_x', 'age_x^2'])

    ord_col = 'ord_r' if 'ord_r' in df.columns else ('ord_w' if 'ord_w' in df.columns else 'ord')
    df_order = pd.DataFrame(poly.fit_transform(df[[ord_col]]), columns=['order_constant', 'order_x', 'order_x^2'])

    df_overseas = pd.DataFrame(poly.fit_transform(df[['overseas_pct']]), columns=['overseas_pct_constant', 'overseas_pct_x', 'overseas_pct_x^2'])

    # 4. Experience (PowerTransformer)
    exp_data = pd.DataFrame(df[['balls_faced_career']])

    if is_training:
        transformers['exp_trans'] = preprocessing.PowerTransformer(method='box-cox', standardize=False)
        exp_transformed = transformers['exp_trans'].fit_transform(exp_data)
    else:
        exp_transformed = transformers['exp_trans'].transform(exp_data)

    df_exp = pd.DataFrame(exp_transformed, columns=['experience'])

    # 5. Oppo (basic reverted rating), plus tail-ender extra reversion (original logic)
    rating_col = 'wkt_rating_2' if target_type == 'run' else 'run_rating_2'
    adjusted_rating = df[rating_col].copy()

    mask = df[ord_col] > 7
    adjusted_rating.loc[mask] = (((1 - adjusted_rating.loc[mask]) / 2) * np.minimum(2, abs(df.loc[mask, ord_col] - 7))) + adjusted_rating.loc[mask]

    df_oppo = pd.DataFrame(adjusted_rating)
    df_oppo.columns = ['oppo']

    # Combine
    X = pd.concat([df_comp, df_nat, df_age, df_exp, df_order, df_overseas, df_oppo], axis=1)
    return X, transformers


for x in np.arange(0, 2, 1):
    # 1. Import Data
    bat_data = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/data/batDataCombinedClean.csv', parse_dates=['date', 'dob'])
    n2h_factors = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batN2HFactors.csv')
    tier_data = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batTierData.csv')



    n2h_factors['host_2'] = np.where((n2h_factors['host_2'] == 'United Arab Emirates') & (n2h_factors['nationality'] == 'Afghanistan'),
                                    'Afghanistan',
                                    n2h_factors['host_2'])

    bat_data = bat_data[bat_data['format'] == 't20']

    allaway_runs = n2h_factors['all_away_runs_factor'].mean()
    allaway_wkts = n2h_factors['all_away_wkts_factor'].mean()

    if x == 0:
        ratings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsJungle.csv', parse_dates=['date'])
    else:
        ratings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsRasoi.csv', parse_dates=['date'])

    # 2. Data Prep - Filters & Adjustments
    bat_data = bat_data.loc[bat_data['competition'].isin(['International League T20', 'SA20', 'Big Bash League', 'Caribbean Premier League', 'Indian Premier League', 'Pakistan Super League',
                                                         'The Hundred (Men\'s Comp)', 'Vitality Blast', 'T20I', 'Major League Cricket', 'tier_2', 'Lanka Premier League']), :]

    ratings = ratings.loc[ratings['competition'].isin(['International League T20', 'SA20', 'Big Bash League', 'Caribbean Premier League', 'Indian Premier League', 'Pakistan Super League',
                                                      'The Hundred (Men\'s Comp)', 'Vitality Blast', 'T20I', 'Major League Cricket', 'tier_2', 'Lanka Premier League']), :].reset_index(drop=True)

    bat_data = bat_data[bat_data['balls_faced'] > 0]

    bat_data['balls_faced_career'] = bat_data['balls_faced_career'] + 5
    ratings['balls_faced_career'] = ratings['balls_faced_career'] + 5

    bat_data = bat_data.merge(ratings[ratings['i_balls_faced'] > 0].loc[:, ['playerid', 'date', 'run_rating', 'wkt_rating']], how='left', on=['playerid', 'date'])

    bat_data['overseas_pct'] = np.where((bat_data['competition'] == 'T20I') | (bat_data['competition'] == 'Indian Premier League'), 1, bat_data['overseas_pct'])
    ratings['overseas_pct'] = np.where((ratings['competition'] == 'T20I') | (ratings['competition'] == 'Indian Premier League'), 1, ratings['overseas_pct'])

    # 3. Basic reversion of each players current rating to the overall mean, this will be used for 'oppo' param
    ovr_run_ratio = bat_data['runs'].sum() / bat_data['realexprbat'].sum()
    ovr_wkt_ratio = bat_data['wkt'].sum() / bat_data['realexpwbat'].sum()

    bat_data.insert(bat_data.columns.get_loc("run_rating") + 1, 'run_rating_2', rep_weight(bat_data['balls_faced_career'], bat_data['run_rating'], ovr_run_ratio, 'run')[1])
    bat_data.insert(bat_data.columns.get_loc("wkt_rating") + 1, 'wkt_rating_2', rep_weight(bat_data['balls_faced_career'], bat_data['wkt_rating'], ovr_wkt_ratio, 'wkt')[1])

    ratings.insert(ratings.columns.get_loc("run_rating") + 1, 'run_rating_2', rep_weight(ratings['balls_faced_career'], ratings['run_rating'], ovr_run_ratio, 'run')[1])
    ratings.insert(ratings.columns.get_loc("wkt_rating") + 1, 'wkt_rating_2', rep_weight(ratings['balls_faced_career'], ratings['wkt_rating'], ovr_wkt_ratio, 'wkt')[1])

    # 4. fit the model using batdata as our training data
    transformers = {}

    comp_cats = list(np.unique(build_comp_index(ratings).reshape(-1, 1)).reshape(1, -1))

    X1, transformers = build_features(bat_data, transformers, is_training=True, target_type='run', comp_cats=comp_cats)
    y_run = pd.DataFrame(bat_data['run_ratio'])
    rep_run_ratio_model = sm.OLS(y_run, X1, missing='drop').fit()

    X2, transformers = build_features(bat_data, transformers, is_training=False, target_type='wkt', comp_cats=comp_cats)
    y_wkt = pd.DataFrame(bat_data['wkt_ratio'])
    rep_wkt_ratio_model = sm.OLS(y_wkt, X2, missing='drop').fit()

    # 5. predict bat data using the fitted model above
    bat_data['rep_run_ratio'] = rep_run_ratio_model.predict(X1)

    bat_data = bat_data.merge(n2h_factors, on=('nationality', 'host'), how='left')
    bat_data['run_factor'] = bat_data['run_factor'].fillna(allaway_runs)
    bat_data['wkt_factor'] = bat_data['wkt_factor'].fillna(allaway_wkts)

    bat_data['rep_run_ratio'] = np.where((bat_data['competition'] == 'T20I') & (bat_data['H/A_competition'] == 'Away'),
                                         bat_data['rep_run_ratio'] * np.minimum(1, bat_data['run_factor']),
                                         bat_data['rep_run_ratio'])

    bat_data['rep_runs'] = bat_data['rep_run_ratio'] * bat_data['realexprbat']

    bat_data['rep_wkt_ratio'] = rep_wkt_ratio_model.predict(X2)

    bat_data['rep_wkt_ratio'] = np.where((bat_data['competition'] == 'T20I') & (bat_data['H/A_competition'] == 'Away'),
                                         bat_data['rep_wkt_ratio'] * np.maximum(1, bat_data['wkt_factor']),
                                         bat_data['rep_wkt_ratio'])

    bat_data['rep_wkt'] = bat_data['rep_wkt_ratio'] * bat_data['realexpwbat']

    # 6. Apply Predictions to Ratings (Validation/Usage)
    X3, _ = build_features(ratings, transformers, is_training=False, target_type='run', comp_cats=comp_cats)
    ratings.insert(ratings.columns.get_loc("run_rating_2") + 1, 'rep_run_ratio', rep_run_ratio_model.predict(X3))

    ratings = ratings.merge(n2h_factors, on=('nationality', 'host'), how='left')
    ratings['run_factor'] = ratings['run_factor'].fillna(allaway_runs)
    ratings['wkt_factor'] = ratings['wkt_factor'].fillna(allaway_wkts)

    ratings['rep_run_ratio'] = np.where((ratings['competition'] == 'T20I') & (ratings['H/A_competition'] == 'Away'),
                                        ratings['rep_run_ratio'] * np.minimum(1, ratings['run_factor']),
                                        ratings['rep_run_ratio'])

    ratings['i_rep_runs'] = ratings['rep_run_ratio'] * ratings['i_realexprbat']

    X4, _ = build_features(ratings, transformers, is_training=False, target_type='wkt', comp_cats=comp_cats)
    ratings.insert(ratings.columns.get_loc("wkt_rating_2") + 1, 'rep_wkt_ratio', rep_wkt_ratio_model.predict(X4))

    ratings['rep_wkt_ratio'] = np.where((ratings['competition'] == 'T20I') & (ratings['H/A_competition'] == 'Away'),
                                        ratings['rep_wkt_ratio'] * np.maximum(1, ratings['wkt_factor']),
                                        ratings['rep_wkt_ratio'])

    ratings['i_rep_wkt'] = ratings['rep_wkt_ratio'] * ratings['i_realexpwbat']

    # 7. Metrics & Rounding
    bat_data['run_sqe'] = (bat_data['run_ratio'] - bat_data['rep_run_ratio']) ** 2
    bat_data['wkt_sqe'] = (bat_data['wkt_ratio'] - bat_data['rep_wkt_ratio']) ** 2
    bat_data['run_err'] = bat_data['rep_runs'] - bat_data['runs']
    bat_data['wkt_err'] = bat_data['rep_wkt'] - bat_data['wkt']

    bat_data['balls_faced_career_round'] = (bat_data['balls_faced_career'] / 500).round().astype(int) * 500

    bat_data['age_round'] = (bat_data['age'] / 2).round().astype(int) * 2
    bat_data['age_round'] = np.where(bat_data['age_round'] == 16, 18, bat_data['age_round'])
    bat_data['age_round'] = np.where(bat_data['age_round'] > 40, 42, bat_data['age_round'])

    bat_data['run_rating_round'] = (bat_data['run_rating_2'] / 0.05).round() * 0.05
    bat_data['wkt_rating_round'] = (bat_data['wkt_rating_2'] / 0.05).round() * 0.05

    bat_data['overseas_pct_round'] = (bat_data['overseas_pct'] / 0.4).round() * 0.4

    # 8. Analysis Pivot
    bat_data['count'] = 1

    agg_dict = {'balls_faced_innings': 'count', 'balls_faced_career': 'mean', 'age': 'mean',
                'realexprbat': 'sum', 'rep_runs': 'sum', 'runs': 'sum', 'realexpwbat': 'sum',
                'rep_wkt': 'sum', 'wkt': 'sum', 'rep_run_ratio': 'mean', 'rep_wkt_ratio': 'mean',
                'run_sqe': 'mean', 'wkt_sqe': 'mean', 'run_err': 'sum', 'wkt_err': 'sum'}

    actuals = pd.pivot_table(bat_data, values=list(agg_dict.keys()), index=['H/A_competition', 'competition'], aggfunc=agg_dict).reset_index()
    actuals['run_ratio'] = actuals['runs'] / actuals['realexprbat']
    actuals['wkt_ratio'] = actuals['wkt'] / actuals['realexpwbat']

    actuals_ratings = ratings.copy()
    actuals_ratings = actuals_ratings[actuals_ratings.matchid > 0]

    actuals_ratings = pd.pivot_table(actuals_ratings,
                                     values=['i_balls_faced', 'i_realexprbat', 'i_rep_runs', 'i_runs', 'i_realexpwbat', 'i_rep_wkt', 'i_wkt', 'rep_wkt_ratio', 'rep_run_ratio', 'age'],
                                     index=['H/A_competition', 'competition'],
                                     aggfunc={'i_balls_faced': 'sum', 'age': 'mean', 'i_realexprbat': 'sum', 'i_rep_runs': 'sum', 'i_runs': 'sum', 'i_realexpwbat': 'sum', 'i_rep_wkt': 'sum',
                                              'i_wkt': 'sum', 'rep_run_ratio': 'mean', 'rep_wkt_ratio': 'mean'}).reset_index()

    actuals_ratings['run_ratio'] = actuals_ratings['i_runs'] / actuals_ratings['i_realexprbat']
    actuals_ratings['wkt_ratio'] = actuals_ratings['i_wkt'] / actuals_ratings['i_realexpwbat']

    # 9. Export
    if x == 0:
        ratings.to_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsJungle2.csv', index=False)

    else:
        ratings.to_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsRasoi2.csv', index=False)


actualsPDF = actuals.loc[:, ['age',
'balls_faced_career',
'balls_faced_innings',
'realexprbat',
'runs',
'rep_runs',
'run_err',
'run_sqe',
'run_ratio', 'rep_run_ratio']]


