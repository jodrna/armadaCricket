import pandas as pd
import numpy as np
from sklearn import preprocessing
import statsmodels.api as sm
from bowlFunctions_w import build_replacement_debug_tables
from paths import PROJECT_ROOT
DEBUG_CONFIG = globals().get('DEBUG_CONFIG', None)
BOWL_REPLACEMENT_DEBUG_TABLES = None

def make_ohe(values, cats, prefix, drop_first=True):
    drop = 'first' if drop_first else None
    encoder = preprocessing.OneHotEncoder(sparse_output=False, categories=[cats], drop=drop, handle_unknown='ignore')
    encoded = encoder.fit_transform(values)

    if drop_first:
        columns = [f'{prefix}__{cat}' for cat in cats[1:]]
    else:
        columns = [f'{prefix}__{cat}' for cat in cats]

    return pd.DataFrame(encoded, columns=columns)


def get_competition_cats(ratings):
    return sorted(np.unique(ratings['competition'] + ' ' + ratings['H/A_competition']).tolist())


def build_training_features_bowl(bowl_data, ratings, transformers):
    # Competition encodings, explicit WT20I Home baseline
    competition_cats = get_competition_cats(ratings)
    transformers['competition_cats'] = competition_cats
    competition = np.array(bowl_data['competition'] + ' ' + bowl_data['H/A_competition']).reshape(-1, 1)
    competition_encodings = make_ohe(competition, competition_cats, 'competition', drop_first=False)
    competition_encodings = competition_encodings.drop(columns=['competition__WT20I Home'], errors='ignore')

    # Bowler arm encodings
    bowl_data['bowler_arm'] = np.where((bowl_data['bowler_arm'] == 'left_seam') | (bowl_data['bowler_arm'] == 'right_seam') | (bowl_data['bowler_arm'] == 'left_f_spin') | (bowl_data['bowler_arm'] == 'right_f_spin'), bowl_data['bowler_arm'], 'other')
    bowler_arm = np.array(bowl_data['bowler_arm']).reshape(-1, 1)
    bowler_arm_cats = ['other', 'left_seam', 'right_seam', 'left_f_spin', 'right_f_spin']
    bowler_arm_encodings = make_ohe(bowler_arm, bowler_arm_cats, 'bowler_arm')

    # Bowler pace encodings
    bowl_data['bowler_pace'] = np.where(bowl_data['bowler_pace'] == 'fast', bowl_data['bowler_pace'], 'other')
    bowler_pace = np.array(bowl_data['bowler_pace']).reshape(-1, 1)
    bowler_pace_cats = ['other', 'fast']
    bowler_pace_encodings = make_ohe(bowler_pace, bowler_pace_cats, 'bowler_pace')

    # WT20I nationality encodings
    wt20i_nat = np.array(np.where(bowl_data['competition'] == 'WT20I', bowl_data['nationality'], 'nil')).reshape(-1, 1)
    wt20i_nat_cats = ['nil', 'England', 'India', 'Afghanistan', 'Australia', 'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'South Africa', 'Pakistan']
    wt20i_nat_encodings = make_ohe(wt20i_nat, wt20i_nat_cats, 'wt20i_nat')

    # Average balls bowled per match
    ballspermatch = pd.DataFrame(bowl_data.loc[:, ['ballspermatch']]).reset_index(drop=True)

    # Experience transform
    experience = pd.DataFrame(bowl_data.loc[:, ['balls_bowled_career']])
    transformer = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    transformer.fit(experience)
    transformers['experience_transformer'] = transformer
    experience = pd.DataFrame(transformer.transform(experience), columns=['experience']).reset_index(drop=True)

    # Overseas pct poly RUN
    overseas_pct = pd.DataFrame(bowl_data.loc[:, ['overseas_pct']])
    overseas_pct_run = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(overseas_pct), columns=['overseas_pct_x', 'overseas_pct_x^2']).reset_index(drop=True)

    # Overseas pct poly WKT
    overseas_pct = pd.DataFrame(bowl_data.loc[:, ['overseas_pct']])
    overseas_pct_wkt = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(overseas_pct), columns=['overseas_pct_x', 'overseas_pct_x^2']).reset_index(drop=True)

    X_run = pd.concat([competition_encodings, bowler_arm_encodings, bowler_pace_encodings, wt20i_nat_encodings, ballspermatch, overseas_pct_run, experience], axis=1)
    X_wkt = pd.concat([competition_encodings, bowler_arm_encodings, bowler_pace_encodings, wt20i_nat_encodings, ballspermatch, overseas_pct_wkt, experience], axis=1)

    X_run = sm.add_constant(X_run, has_constant='add')
    X_wkt = sm.add_constant(X_wkt, has_constant='add')

    return bowl_data, X_run, X_wkt, transformers


def build_ratings_features_bowl(ratings, transformers):
    # Competition encodings, explicit WT20I Home baseline
    competition = np.array(ratings['competition'] + ' ' + ratings['H/A_competition']).reshape(-1, 1)
    competition_encodings = make_ohe(competition, transformers['competition_cats'], 'competition', drop_first=False)
    competition_encodings = competition_encodings.drop(columns=['competition__WT20I Home'], errors='ignore')

    # Bowler arm encodings
    ratings['bowler_arm'] = np.where((ratings['bowler_arm'] == 'left_seam') | (ratings['bowler_arm'] == 'right_seam') | (ratings['bowler_arm'] == 'left_f_spin') | (ratings['bowler_arm'] == 'right_f_spin'), ratings['bowler_arm'], 'other')
    bowler_arm = np.array(ratings['bowler_arm']).reshape(-1, 1)
    bowler_arm_cats = ['other', 'left_seam', 'right_seam', 'left_f_spin', 'right_f_spin']
    bowler_arm_encodings = make_ohe(bowler_arm, bowler_arm_cats, 'bowler_arm')

    # Bowler pace encodings
    ratings['bowler_pace'] = np.where(ratings['bowler_pace'] == 'fast', ratings['bowler_pace'], 'other')
    bowler_pace = np.array(ratings['bowler_pace']).reshape(-1, 1)
    bowler_pace_cats = ['other', 'fast']
    bowler_pace_encodings = make_ohe(bowler_pace, bowler_pace_cats, 'bowler_pace')

    # WT20I nationality encodings
    wt20i_nat = np.array(np.where(ratings['competition'] == 'WT20I', ratings['nationality'], 'nil')).reshape(-1, 1)
    wt20i_nat_cats = ['nil', 'England', 'India', 'Afghanistan', 'Australia', 'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'South Africa', 'Pakistan']
    wt20i_nat_encodings = make_ohe(wt20i_nat, wt20i_nat_cats, 'wt20i_nat')

    # Balls per match bowled on average
    ballspermatch = pd.DataFrame(ratings.loc[:, ['ballspermatch']]).reset_index(drop=True)

    # Experience transform
    experience = pd.DataFrame(ratings.loc[:, ['balls_bowled_career']])
    experience = pd.DataFrame(transformers['experience_transformer'].transform(experience), columns=['experience']).reset_index(drop=True)

    # Overseas pct poly RUN
    overseas_pct = pd.DataFrame(ratings.loc[:, ['overseas_pct']])
    overseas_pct_run = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(overseas_pct), columns=['overseas_pct_x', 'overseas_pct_x^2']).reset_index(drop=True)

    # Overseas pct poly WKT
    overseas_pct = pd.DataFrame(ratings.loc[:, ['overseas_pct']])
    overseas_pct_wkt = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(overseas_pct), columns=['overseas_pct_x', 'overseas_pct_x^2']).reset_index(drop=True)

    X_run = pd.concat([competition_encodings, bowler_arm_encodings, bowler_pace_encodings, wt20i_nat_encodings, ballspermatch, overseas_pct_run, experience], axis=1)
    X_wkt = pd.concat([competition_encodings, bowler_arm_encodings, bowler_pace_encodings, wt20i_nat_encodings, ballspermatch, overseas_pct_wkt, experience], axis=1)

    X_run = sm.add_constant(X_run, has_constant='add')
    X_wkt = sm.add_constant(X_wkt, has_constant='add')

    return ratings, X_run, X_wkt


for x in np.arange(0, 2, 1):

    # -------------------------
    # 1) Imports
    # -------------------------
    bowl_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/bowlDataCombinedClean_w.csv', parse_dates=['date', 'dob'])

    if x == 0:
        model_name = 'jungle'
        ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle_w.csv', parse_dates=['date'])
    else:
        model_name = 'rasoi'
        ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsRasoi_w.csv', parse_dates=['date'])

    # -------------------------
    # 2) Filters
    # -------------------------
    bowl_data = bowl_data.loc[bowl_data['format'] == 't20', :].copy()

    # -------------------------
    # 3) Base cleaning / merges
    # -------------------------
    ratings['run_rating'], ratings['wkt_rating'] = ratings['run_rating'].fillna(1), ratings['wkt_rating'].fillna(1)
    ratings['balls_bowled_r'], ratings['balls_bowled_w'] = ratings['balls_bowled_r'].fillna(1), ratings['balls_bowled_w'].fillna(1)

    bowl_data['index'] = bowl_data['competition'] + ' ' + bowl_data['H/A_competition']
    league_balls = pd.pivot_table(bowl_data, values=['balls_bowled'], index=['index', 'competition'], aggfunc='sum').reset_index()
    bowl_data = bowl_data.drop(labels=['index'], axis=1)

    bowl_data = bowl_data[(bowl_data['balls_bowled'] > 0)].copy()
    bowl_data = bowl_data.dropna(subset=['bowlertype_2']).copy()
    ratings = ratings.dropna(subset=['bowlertype_2']).reset_index(drop=True).copy()

    bowl_data['balls_bowled_career'] = bowl_data['balls_bowled_career'] + 6
    ratings['balls_bowled_career'] = ratings['balls_bowled_career'] + 6

    bowl_data = bowl_data.merge(ratings[ratings['i_balls_bowled'] > 0].loc[:, ['playerid', 'date', 'run_rating', 'wkt_rating']], how='left', on=['playerid', 'date'])

    # -------------------------
    # 4) Fit RUN + WKT models
    # -------------------------
    transformers = {}
    bowl_data, X_run, X_wkt, transformers = build_training_features_bowl(bowl_data, ratings, transformers)

    y = pd.DataFrame(bowl_data['run_ratio'])
    rep_run_ratio_model = sm.OLS(y, X_run, missing='drop').fit()
    run_params = rep_run_ratio_model.params.copy()

    y = pd.DataFrame(bowl_data['wkt_ratio'])
    rep_wkt_ratio_model = sm.OLS(y, X_wkt,  missing='drop').fit()
    wkt_params = rep_wkt_ratio_model.params.copy()


    # -------------------------
    # 5) Predict training data
    # -------------------------
    bowl_data['rep_run_ratio'] = rep_run_ratio_model.predict(X_run)
    bowl_data['rep_runs'] = bowl_data['rep_run_ratio'] * bowl_data['realexprbowl']

    bowl_data['rep_wkt_ratio'] = rep_wkt_ratio_model.predict(X_wkt)
    bowl_data['rep_wkt'] = bowl_data['rep_wkt_ratio'] * bowl_data['realexpwbowl']

    # -------------------------
    # 6) League coefficient reversion
    # -------------------------

    league_balls['feature_name'] = 'competition__' + league_balls['index']
    league_balls['weight'] = np.where(league_balls['balls_bowled'] > 20000, 1, league_balls['balls_bowled'] / 20000)

    for _, row in league_balls.iterrows():
        if row['feature_name'] in run_params.index:
            run_params.loc[row['feature_name']] = row['weight'] * run_params.loc[row['feature_name']]

        if row['feature_name'] in wkt_params.index:
            wkt_params.loc[row['feature_name']] = row['weight'] * wkt_params.loc[row['feature_name']]

    params = pd.merge(pd.DataFrame(run_params), pd.DataFrame(wkt_params), how='left', left_index=True, right_index=True).reset_index()
    aux = pd.DataFrame([['λ', str(transformers['experience_transformer'].lambdas_[0]), str(transformers['experience_transformer'].lambdas_[0])]], columns=params.columns)
    params = pd.concat([params, aux], axis=0)

    # -------------------------
    # 6) Predict ratings outputs
    # -------------------------
    ratings, X_run_r, X_wkt_r = build_ratings_features_bowl(ratings, transformers)

    ratings.insert(ratings.columns.get_loc('run_rating') + 1, 'rep_run_ratio', X_run_r.to_numpy() @ run_params.to_numpy())
    ratings['i_rep_runs'] = ratings['rep_run_ratio'] * ratings['i_realexprbowl']

    ratings.insert(ratings.columns.get_loc('wkt_rating') + 1, 'rep_wkt_ratio', X_wkt_r.to_numpy() @ wkt_params.to_numpy())
    ratings['i_rep_wkt'] = ratings['rep_wkt_ratio'] * ratings['i_realexpwbowl']

    # -------------------------
    # 7) Debug replacement breakdown
    # -------------------------
    if DEBUG_CONFIG is not None and DEBUG_CONFIG['model'] == model_name:
        BOWL_REPLACEMENT_DEBUG_TABLES = build_replacement_debug_tables(
            DEBUG_CONFIG,
            ratings,
            X_run_r,
            X_wkt_r,
            run_params,
            wkt_params
        )

    # -------------------------
    # 8) Checks + pivots
    # -------------------------
    test = ratings.copy()

    test['sum_rep_r'] = test['rep_run_ratio'] * test['i_balls_bowled']
    test['sum_rep_w'] = test['rep_wkt_ratio'] * test['i_balls_bowled']

    sum_rep_r = test['sum_rep_r'].sum()
    sum_rep_w = test['sum_rep_w'].sum()
    sum_balls = test['i_balls_bowled'].sum()

    rep_r_o = sum_rep_r / sum_balls
    rep_w_o = sum_rep_w / sum_balls

    bowl_data['run_sqe'] = (bowl_data['run_ratio'] - bowl_data['rep_run_ratio']) ** 2
    bowl_data['wkt_sqe'] = (bowl_data['wkt_ratio'] - bowl_data['rep_wkt_ratio']) ** 2
    bowl_data['run_err'] = bowl_data['rep_runs'] - bowl_data['runs']
    bowl_data['wkt_err'] = bowl_data['rep_wkt'] - bowl_data['wkt']

    actuals = pd.pivot_table(bowl_data,
                             values=['balls_bowled', 'realexprbowl', 'rep_runs', 'runs', 'realexpwbowl', 'rep_wkt', 'wkt', 'rep_wkt_ratio', 'rep_run_ratio', 'age', 'balls_bowled_career',
                                     'run_sqe', 'wkt_sqe', 'run_err', 'wkt_err'],
                             index=['competition'],
                             aggfunc={'balls_bowled': 'count', 'balls_bowled_career': 'mean', 'age': 'mean', 'realexprbowl': 'sum', 'rep_runs': 'sum', 'runs': 'sum', 'realexpwbowl': 'sum',
                                      'rep_wkt': 'sum', 'wkt': 'sum', 'rep_run_ratio': 'mean', 'rep_wkt_ratio': 'mean', 'run_sqe': 'mean', 'wkt_sqe': 'mean', 'run_err': 'sum', 'wkt_err': 'sum'}).reset_index()

    actuals['run_ratio'] = actuals['runs'] / actuals['realexprbowl']
    actuals['wkt_ratio'] = actuals['wkt'] / actuals['realexpwbowl']

    actuals_ratings = ratings.copy()
    actuals_ratings = actuals_ratings[actuals_ratings.matchid > 0].copy()

    actuals_ratings = pd.pivot_table(actuals_ratings, values=['i_balls_bowled', 'i_realexprbowl', 'i_rep_runs', 'i_runs', 'i_realexpwbowl', 'i_rep_wkt', 'i_wkt', 'rep_wkt_ratio', 'rep_run_ratio'], index=['competition'], aggfunc={'i_balls_bowled': 'sum', 'i_realexprbowl': 'sum', 'i_rep_runs': 'sum', 'i_runs': 'sum', 'i_realexpwbowl': 'sum', 'i_rep_wkt': 'sum', 'i_wkt': 'sum', 'rep_run_ratio': 'mean', 'rep_wkt_ratio': 'mean'}).reset_index()
    actuals_ratings['run_ratio'] = actuals_ratings['i_runs'] / actuals_ratings['i_realexprbowl']
    actuals_ratings['wkt_ratio'] = actuals_ratings['i_wkt'] / actuals_ratings['i_realexpwbowl']

    # -------------------------
    # 9) Export
    # -------------------------
    if x == 0:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle2_w.csv', index=False)
    else:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsRasoi2_w.csv', index=False)





