import pandas as pd
import numpy as np
from sklearn import preprocessing
import statsmodels.api as sm
from paths import PROJECT_ROOT


def build_training_features_bowl(bowl_data, ratings):
    # Competition encodings (categories from outputs, fit on bowl_data like original)
    competition = np.array(ratings['competition'] + " " + ratings['H/A_competition']).reshape(-1, 1)
    competition_cats = list(np.unique(competition).reshape(1, -1))
    competition = np.array(bowl_data['competition'] + " " + bowl_data['H/A_competition']).reshape(-1, 1)
    competition_encodings = pd.DataFrame(preprocessing.OneHotEncoder(sparse_output=False, categories=competition_cats).fit_transform(competition), columns=competition_cats)

    # Bowler type encodings (fit on bowl_data like original)
    bowler_type = np.array(bowl_data['bowlertype_2']).reshape(-1, 1)
    bowler_type_cats = list(np.unique(bowler_type).reshape(1, -1))
    bowler_type_encodings = pd.DataFrame(preprocessing.OneHotEncoder(sparse_output=False, categories=bowler_type_cats).fit_transform(bowler_type), columns=bowler_type_cats)

    # Bowler arm encodings
    bowl_data['bowler_arm'] = np.where((bowl_data['bowler_arm'] == 'left_seam') | (bowl_data['bowler_arm'] == 'right_seam') | (bowl_data['bowler_arm'] == 'left_f_spin') | (bowl_data['bowler_arm'] == 'right_f_spin'), bowl_data['bowler_arm'], 'other')
    bowler_arm = np.array(bowl_data['bowler_arm']).reshape(-1, 1)
    bowler_arm_cats = list(np.unique(np.array(['left_seam', 'right_seam', 'left_f_spin', 'right_f_spin']).reshape(-1, 1)).reshape(1, -1))
    bowler_arm_encodings = pd.DataFrame(preprocessing.OneHotEncoder(sparse_output=False, categories=bowler_arm_cats, handle_unknown='ignore').fit_transform(bowler_arm), columns=bowler_arm_cats)

    # Bowler pace encodings
    bowl_data['bowler_pace'] = np.where(bowl_data['bowler_pace'] == 'fast', bowl_data['bowler_pace'], 'other')
    bowler_pace = np.array(bowl_data['bowler_pace']).reshape(-1, 1)
    bowler_pace_cats = list(np.unique(np.array(['fast']).reshape(-1, 1)).reshape(1, -1))
    bowler_pace_encodings = pd.DataFrame(preprocessing.OneHotEncoder(sparse_output=False, categories=bowler_pace_cats, handle_unknown='ignore').fit_transform(bowler_pace), columns=bowler_pace_cats)

    # T20I nationality encodings (else nil)
    t20i_nat = np.array(np.where(bowl_data['competition'] == 'T20I', bowl_data['nationality'], 'nil')).reshape(-1, 1)
    t20i_nat_cats = list(np.unique(np.array(['England', 'India', 'Afghanistan', 'Australia', 'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'South Africa', 'Pakistan']).reshape(-1, 1)).reshape(1, -1))
    t20i_nat_encodings = pd.DataFrame(preprocessing.OneHotEncoder(sparse_output=False, categories=t20i_nat_cats, handle_unknown='ignore').fit_transform(t20i_nat), columns=t20i_nat_cats)

    ballspermatch = pd.DataFrame(bowl_data.loc[:, ['ballspermatch']])

    # Experience (Box-Cox) for RUN model
    experience = pd.DataFrame(bowl_data.loc[:, ['balls_bowled_career']])
    run_transformer = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    run_transformer.fit(experience)
    experience_run = pd.DataFrame(run_transformer.transform(experience), columns=['experience'])

    # Overseas pct poly (RUN)
    overseas_pct = pd.DataFrame(bowl_data.loc[:, ['overseas_pct']])
    overseas_pct_run = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2).fit_transform(overseas_pct), columns=['overseas_pct_constant', 'overseas_pct_x', 'overseas_pct_x^2'])

    # Overseas pct poly (WKT)
    overseas_pct = pd.DataFrame(bowl_data.loc[:, ['overseas_pct']])
    overseas_pct_wkt = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2).fit_transform(overseas_pct), columns=['overseas_pct_constant', 'overseas_pct_x', 'overseas_pct_x^2'])

    # Experience (Box-Cox) for WKT lambda capture (like original) — not used in X for wkt model
    experience = pd.DataFrame(bowl_data.loc[:, ['balls_bowled_career']])
    wkt_transformer = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    wkt_transformer.fit(experience)

    X_run = pd.concat([competition_encodings, bowler_type_encodings, bowler_arm_encodings, bowler_pace_encodings, t20i_nat_encodings, ballspermatch, overseas_pct_run, experience_run], axis=1)
    X_wkt = pd.concat([competition_encodings, bowler_type_encodings, bowler_arm_encodings, bowler_pace_encodings, t20i_nat_encodings, ballspermatch, overseas_pct_wkt], axis=1)

    return bowl_data, X_run, X_wkt, run_transformer, wkt_transformer


def build_ratings_features_bowl(ratings, run_transformer):
    # IMPORTANT: preserve original behaviour (refit encoders on outputs, not reuse training encoders)

    competition = np.array(ratings['competition'] + " " + ratings['H/A_competition']).reshape(-1, 1)
    competition_cats = list(np.unique(competition).reshape(1, -1))
    competition_encodings = pd.DataFrame(preprocessing.OneHotEncoder(sparse_output=False, categories=competition_cats).fit_transform(competition), columns=competition_cats)

    bowler_type = np.array(ratings['bowlertype_2']).reshape(-1, 1)
    bowler_type_cats = list(np.unique(bowler_type).reshape(1, -1))
    bowler_type_encodings = pd.DataFrame(preprocessing.OneHotEncoder(sparse_output=False, categories=bowler_type_cats).fit_transform(bowler_type), columns=bowler_type_cats)

    ratings['bowler_arm'] = np.where((ratings['bowler_arm'] == 'left_seam') | (ratings['bowler_arm'] == 'right_seam') | (ratings['bowler_arm'] == 'left_f_spin') | (ratings['bowler_arm'] == 'right_f_spin'), ratings['bowler_arm'], 'other')
    bowler_arm = np.array(ratings['bowler_arm']).reshape(-1, 1)
    bowler_arm_cats = list(np.unique(np.array(['left_seam', 'right_seam', 'left_f_spin', 'right_f_spin']).reshape(-1, 1)).reshape(1, -1))
    bowler_arm_encodings = pd.DataFrame(preprocessing.OneHotEncoder(sparse_output=False, categories=bowler_arm_cats, handle_unknown='ignore').fit_transform(bowler_arm), columns=bowler_arm_cats)

    ratings['bowler_pace'] = np.where(ratings['bowler_pace'] == 'fast', ratings['bowler_pace'], 'other')
    bowler_pace = np.array(ratings['bowler_pace']).reshape(-1, 1)
    bowler_pace_cats = list(np.unique(np.array(['fast']).reshape(-1, 1)).reshape(1, -1))
    bowler_pace_encodings = pd.DataFrame(preprocessing.OneHotEncoder(sparse_output=False, categories=bowler_pace_cats, handle_unknown='ignore').fit_transform(bowler_pace), columns=bowler_pace_cats)

    t20i_nat = np.array(np.where(ratings['competition'] == 'T20I', ratings['nationality'], 'nil')).reshape(-1, 1)
    t20i_nat_cats = list(np.unique(np.array(['England', 'India', 'Afghanistan', 'Australia', 'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'South Africa', 'Pakistan']).reshape(-1, 1)).reshape(1, -1))
    t20i_nat_encodings = pd.DataFrame(preprocessing.OneHotEncoder(sparse_output=False, categories=t20i_nat_cats, handle_unknown='ignore').fit_transform(t20i_nat), columns=t20i_nat_cats)

    ballspermatch = pd.DataFrame(ratings.loc[:, ['ballspermatch']])

    experience = pd.DataFrame(ratings.loc[:, ['balls_bowled_career']])
    experience = pd.DataFrame(run_transformer.transform(experience), columns=['experience'])

    overseas_pct = pd.DataFrame(ratings.loc[:, ['overseas_pct']])
    overseas_pct = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2).fit_transform(overseas_pct), columns=['overseas_pct_constant', 'overseas_pct_x', 'overseas_pct_x^2'])

    X_run = pd.concat([competition_encodings, bowler_type_encodings, bowler_arm_encodings, bowler_pace_encodings, t20i_nat_encodings, ballspermatch, overseas_pct, experience], axis=1)

    overseas_pct = pd.DataFrame(ratings.loc[:, ['overseas_pct']])
    overseas_pct = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2).fit_transform(overseas_pct), columns=['overseas_pct_constant', 'overseas_pct_x', 'overseas_pct_x^2'])

    X_wkt = pd.concat([competition_encodings, bowler_type_encodings, bowler_arm_encodings, bowler_pace_encodings, t20i_nat_encodings, ballspermatch, overseas_pct], axis=1)

    return ratings, X_run, X_wkt


for x in np.arange(0, 2, 1):

    # -------------------------
    # 1) Imports
    # -------------------------
    bowl_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/bowlDataCombinedClean_w.csv', parse_dates=['date', 'dob'])
    tier_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/auxiliaries/bowlTierData_w.csv')

    if x == 0:
        ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle_w.csv', parse_dates=['date'])
    else:
        ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsRasoi_w.csv', parse_dates=['date'])

    # -------------------------
    # 2) Filters (main comps only)
    # -------------------------
    bowl_data = bowl_data.loc[
        bowl_data['competition'].isin([
                    "Abu Dhabi Women's T20 Counties Super Cup",
        'FairBreak Invitational Tournament',
        'New Zealand',
        'South Africa',
        "The Hundred (Women's Comp)",
        "Vitality Women's T20 County Cup",
        'WT20I',
        "Women's Big Bash League",
        "Women's Premier League",
        "Women's Vitality Blast",
        'tier_2']), :]

    ratings = ratings.loc[ratings['competition'].isin([
                    "Abu Dhabi Women's T20 Counties Super Cup",
        'FairBreak Invitational Tournament',
        'New Zealand',
        'South Africa',
        "The Hundred (Women's Comp)",
        "Vitality Women's T20 County Cup",
        'WT20I',
        "Women's Big Bash League",
        "Women's Premier League",
        "Women's Vitality Blast",
        'tier_2']), :].reset_index(drop=True)

    # -------------------------
    # 3) Base cleaning / merges
    # -------------------------
    ratings['run_rating'], ratings['wkt_rating'] = ratings['run_rating'].fillna(1), ratings['wkt_rating'].fillna(1)
    ratings['balls_bowled_r'], ratings['balls_bowled_w'] = ratings['balls_bowled_r'].fillna(1), ratings['balls_bowled_w'].fillna(1)

    bowl_data['index'] = bowl_data['competition'] + " " + bowl_data['H/A_competition']
    league_balls = pd.pivot_table(bowl_data, values=['balls_bowled'], index=['index', 'competition'], aggfunc='sum').reset_index()
    bowl_data = bowl_data.drop(labels=['index'], axis=1)

    bowl_data = bowl_data[(bowl_data['balls_bowled'] > 0)].copy()
    bowl_data = bowl_data.dropna(subset=['bowlertype_2']).copy()
    ratings = ratings.dropna(subset=['bowlertype_2']).reset_index(drop=True).copy()

    bowl_data['balls_bowled_career'] = bowl_data['balls_bowled_career'] + 6
    ratings['balls_bowled_career'] = ratings['balls_bowled_career'] + 6

    bowl_data = bowl_data.merge(ratings[ratings['i_balls_bowled'] > 0].loc[:, ['playerid', 'date', 'run_rating', 'wkt_rating']], how='left', on=['playerid', 'date'])

    # -------------------------
    # 4) Fit RUN + WKT models (train on bowl_data)
    # -------------------------
    bowl_data, X_run, X_wkt, run_transformer, wkt_transformer = build_training_features_bowl(bowl_data, ratings)

    y = pd.DataFrame(bowl_data['run_ratio'])
    rep_run_ratio_model = sm.OLS(y, X_run).fit()
    bowl_data['rep_run_ratio'] = rep_run_ratio_model.predict(X_run)
    bowl_data['rep_runs'] = bowl_data['rep_run_ratio'] * bowl_data['realexprbowl']

    y = pd.DataFrame(bowl_data['wkt_ratio'])
    rep_wkt_ratio_model = sm.OLS(y, X_wkt).fit()
    bowl_data['rep_wkt_ratio'] = rep_wkt_ratio_model.predict(X_wkt)
    bowl_data['rep_wkt'] = bowl_data['rep_wkt_ratio'] * bowl_data['realexpwbowl']

    # -------------------------
    # 5) League coefficient reversion (tier-based)
    # -------------------------
    params = pd.merge(pd.DataFrame(rep_run_ratio_model.params), pd.DataFrame(rep_wkt_ratio_model.params), how='left', left_index=True, right_index=True).reset_index()
    params['index'] = np.where(params['index'].apply(len) == 1, params['index'].str[0], params['index'])
    league_balls = league_balls.merge(params, how='left', on=['index'])

    league_balls = league_balls.merge(tier_data, on='competition')

    league_balls_no_intl = league_balls[league_balls.competition != 'T20I'].copy()
    avg_0x = league_balls_no_intl['0_x'].mean()
    avg_runs = league_balls_no_intl['avg_runs'].mean()
    avg_0y = league_balls_no_intl['0_y'].mean()
    avg_wkts = league_balls_no_intl['avg_wkts'].mean()

    league_balls['runs_diff'] = league_balls['avg_runs'] - avg_runs
    league_balls['wkts_diff'] = league_balls['avg_wkts'] - avg_wkts
    league_balls['new_0_x'] = avg_0x + league_balls['runs_diff']
    league_balls['new_0_y'] = avg_0y + league_balls['wkts_diff']

    league_balls['avg_runs'] = np.where(league_balls['competition'] == 'T20I', league_balls['0_x'], league_balls['new_0_x'])
    league_balls['avg_wkts'] = np.where(league_balls['competition'] == 'T20I', league_balls['0_y'], league_balls['new_0_y'])

    league_balls['weight'] = np.where(league_balls['balls_bowled'] > 20000, 1, league_balls['balls_bowled'] / 20000)
    league_balls['weight_2'] = np.where(league_balls['balls_bowled'] > 20000, 1, league_balls['balls_bowled'] / 20000)
    league_balls['weight'] = np.where(league_balls['competition'] == 'tier_2', league_balls['weight_2'], league_balls['weight'])

    league_balls['runs'] = (league_balls['weight'] * league_balls['0_x']) + ((1 - league_balls['weight']) * league_balls['avg_runs'])
    league_balls['wkts'] = (league_balls['weight'] * league_balls['0_y']) + ((1 - league_balls['weight']) * league_balls['avg_wkts'])

    runs_series = pd.Series(league_balls['runs'].values, index=league_balls['index'])
    rep_run_ratio_model.params[:len(runs_series)] = runs_series.astype(float)

    wkt_series = pd.Series(league_balls['wkts'].values, index=league_balls['index'])
    rep_wkt_ratio_model.params[:len(wkt_series)] = wkt_series.astype(float)

    params = pd.merge(pd.DataFrame(rep_run_ratio_model.params), pd.DataFrame(rep_wkt_ratio_model.params), how='left', left_index=True, right_index=True).reset_index()
    params['index'] = np.where(params['index'].apply(len) == 1, params['index'].str[0], params['index'])
    aux = pd.DataFrame([['λ', str(run_transformer.lambdas_[0]), str(wkt_transformer.lambdas_[0])]], columns=params.columns)
    params = pd.concat([params, aux], axis=0)

    # -------------------------
    # 6) Predict outputs (using updated params)
    # -------------------------
    ratings, X_run_r, X_wkt_r = build_ratings_features_bowl(ratings, run_transformer)

    ratings.insert(ratings.columns.get_loc("run_rating") + 1, 'rep_run_ratio', rep_run_ratio_model.predict(X_run_r))
    # outputs['rep_run_ratio'] = outputs['rep_run_ratio'] * outputs['t2h_factor_r']
    ratings['i_rep_runs'] = ratings['rep_run_ratio'] * ratings['i_realexprbowl']

    ratings.insert(ratings.columns.get_loc("wkt_rating") + 1, 'rep_wkt_ratio', rep_wkt_ratio_model.predict(X_wkt_r))
    # outputs['rep_wkt_ratio'] = outputs['rep_wkt_ratio'] * outputs['t2h_factor_w']
    ratings['i_rep_wkt'] = ratings['rep_wkt_ratio'] * ratings['i_realexpwbowl']

    # -------------------------
    # 7) Checks + pivots
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

    bowl_data['balls_bowled_career_round'] = bowl_data['balls_bowled_career'].apply(lambda z: int(100 * round(float(z) / 100)))
    bowl_data['age_round'] = bowl_data['age'].apply(lambda z: int(2 * round(float(z) / 2)))
    bowl_data['age_round'] = np.where(bowl_data['age_round'] == 16, 18, np.where(bowl_data['age_round'] > 40, 42, bowl_data['age_round']))
    bowl_data['overseas_pct_round'] = bowl_data['overseas_pct'].apply(lambda z: 0.4 * round(float(z) / 0.4))

    actuals = pd.pivot_table(bowl_data, values=['balls_bowled', 'realexprbowl', 'rep_runs', 'runs', 'realexpwbowl', 'rep_wkt', 'wkt', 'rep_wkt_ratio', 'rep_run_ratio', 'age', 'balls_bowled_career', 'run_sqe', 'wkt_sqe', 'run_err', 'wkt_err'], index=['innings'], aggfunc={'balls_bowled': 'count', 'balls_bowled_career': 'mean', 'age': 'mean', 'realexprbowl': 'sum', 'rep_runs': 'sum', 'runs': 'sum', 'realexpwbowl': 'sum', 'rep_wkt': 'sum', 'wkt': 'sum', 'rep_run_ratio': 'mean', 'rep_wkt_ratio': 'mean', 'run_sqe': 'mean', 'wkt_sqe': 'mean', 'run_err': 'sum', 'wkt_err': 'sum'}).reset_index()
    actuals['run_ratio'] = actuals['runs'] / actuals['realexprbowl']
    actuals['wkt_ratio'] = actuals['wkt'] / actuals['realexpwbowl']

    actuals_ratings = ratings.copy()
    actuals_ratings = actuals_ratings[actuals_ratings.matchid > 0].copy()

    actuals_ratings = pd.pivot_table(actuals_ratings, values=['i_balls_bowled', 'i_realexprbowl', 'i_rep_runs', 'i_runs', 'i_realexpwbowl', 'i_rep_wkt', 'i_wkt', 'rep_wkt_ratio', 'rep_run_ratio'], index=['competition'], aggfunc={'i_balls_bowled': 'sum', 'i_realexprbowl': 'sum', 'i_rep_runs': 'sum', 'i_runs': 'sum', 'i_realexpwbowl': 'sum', 'i_rep_wkt': 'sum', 'i_wkt': 'sum', 'rep_run_ratio': 'mean', 'rep_wkt_ratio': 'mean'}).reset_index()
    actuals_ratings['run_ratio'] = actuals_ratings['i_runs'] / actuals_ratings['i_realexprbowl']
    actuals_ratings['wkt_ratio'] = actuals_ratings['i_wkt'] / actuals_ratings['i_realexpwbowl']

    # -------------------------
    # 8) Export
    # -------------------------
    if x == 0:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle2_w.csv', index=False)
    else:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsRasoi2_w.csv', index=False)


