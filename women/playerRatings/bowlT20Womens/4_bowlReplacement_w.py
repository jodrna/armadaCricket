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


def build_training_features_bowl(bowl_data, ratings):
    # Competition encodings
    competition_cats = sorted(np.unique(ratings['competition'] + ' ' + ratings['H/A_competition']).tolist())
    competition = np.array(bowl_data['competition'] + ' ' + bowl_data['H/A_competition']).reshape(-1, 1)
    competition_encodings = make_ohe(competition, competition_cats, 'competition')

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

    ballspermatch = pd.DataFrame(bowl_data.loc[:, ['ballspermatch']]).reset_index(drop=True)

    # Experience for RUN model
    experience = pd.DataFrame(bowl_data.loc[:, ['balls_bowled_career']])
    run_transformer = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    run_transformer.fit(experience)
    experience_run = pd.DataFrame(run_transformer.transform(experience), columns=['experience']).reset_index(drop=True)

    # Experience for WKT model
    experience = pd.DataFrame(bowl_data.loc[:, ['balls_bowled_career']])
    wkt_transformer = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    wkt_transformer.fit(experience)
    experience_wkt = pd.DataFrame(wkt_transformer.transform(experience), columns=['experience']).reset_index(drop=True)

    # Overseas pct poly RUN
    overseas_pct = pd.DataFrame(bowl_data.loc[:, ['overseas_pct']])
    overseas_pct_run = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(overseas_pct), columns=['overseas_pct_x', 'overseas_pct_x^2']).reset_index(drop=True)

    # Overseas pct poly WKT
    overseas_pct = pd.DataFrame(bowl_data.loc[:, ['overseas_pct']])
    overseas_pct_wkt = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(overseas_pct), columns=['overseas_pct_x', 'overseas_pct_x^2']).reset_index(drop=True)

    X_run = pd.concat([competition_encodings, bowler_arm_encodings, bowler_pace_encodings, wt20i_nat_encodings, ballspermatch, overseas_pct_run, experience_run], axis=1)
    X_wkt = pd.concat([competition_encodings, bowler_arm_encodings, bowler_pace_encodings, wt20i_nat_encodings, ballspermatch, overseas_pct_wkt, experience_wkt], axis=1)

    return bowl_data, X_run, X_wkt, run_transformer, wkt_transformer


def build_ratings_features_bowl(ratings, run_transformer, wkt_transformer):
    # Competition encodings
    competition_cats = sorted(np.unique(ratings['competition'] + ' ' + ratings['H/A_competition']).tolist())
    competition = np.array(ratings['competition'] + ' ' + ratings['H/A_competition']).reshape(-1, 1)
    competition_encodings = make_ohe(competition, competition_cats, 'competition')

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

    ballspermatch = pd.DataFrame(ratings.loc[:, ['ballspermatch']]).reset_index(drop=True)

    # Experience RUN
    experience = pd.DataFrame(ratings.loc[:, ['balls_bowled_career']])
    experience_run = pd.DataFrame(run_transformer.transform(experience), columns=['experience']).reset_index(drop=True)

    # Experience WKT
    experience = pd.DataFrame(ratings.loc[:, ['balls_bowled_career']])
    experience_wkt = pd.DataFrame(wkt_transformer.transform(experience), columns=['experience']).reset_index(drop=True)

    # Overseas pct poly RUN
    overseas_pct = pd.DataFrame(ratings.loc[:, ['overseas_pct']])
    overseas_pct_run = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(overseas_pct), columns=['overseas_pct_x', 'overseas_pct_x^2']).reset_index(drop=True)

    # Overseas pct poly WKT
    overseas_pct = pd.DataFrame(ratings.loc[:, ['overseas_pct']])
    overseas_pct_wkt = pd.DataFrame(preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(overseas_pct), columns=['overseas_pct_x', 'overseas_pct_x^2']).reset_index(drop=True)

    X_run = pd.concat([competition_encodings, bowler_arm_encodings, bowler_pace_encodings, wt20i_nat_encodings, ballspermatch, overseas_pct_run, experience_run], axis=1)
    X_wkt = pd.concat([competition_encodings, bowler_arm_encodings, bowler_pace_encodings, wt20i_nat_encodings, ballspermatch, overseas_pct_wkt, experience_wkt], axis=1)

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
    # 2) Filters
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
            'tier_2'
        ]), :]

    ratings = ratings.loc[
        ratings['competition'].isin([
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
            'tier_2'
        ]), :].reset_index(drop=True)

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
    # 5) League coefficient reversion
    # -------------------------
    params = pd.merge(pd.DataFrame(rep_run_ratio_model.params), pd.DataFrame(rep_wkt_ratio_model.params), how='left', left_index=True, right_index=True).reset_index()
    league_balls['feature_name'] = 'competition__' + league_balls['index']
    league_balls = league_balls.merge(params, how='left', left_on='feature_name', right_on='index')

    league_balls = league_balls.merge(tier_data, on='competition')

    league_balls_no_intl = league_balls[league_balls.competition != 'WT20I'].copy()
    avg_0x = league_balls_no_intl['0_x'].mean()
    avg_runs = league_balls_no_intl['avg_runs'].mean()
    avg_0y = league_balls_no_intl['0_y'].mean()
    avg_wkts = league_balls_no_intl['avg_wkts'].mean()

    league_balls['runs_diff'] = league_balls['avg_runs'] - avg_runs
    league_balls['wkts_diff'] = league_balls['avg_wkts'] - avg_wkts
    league_balls['new_0_x'] = avg_0x + league_balls['runs_diff']
    league_balls['new_0_y'] = avg_0y + league_balls['wkts_diff']

    league_balls['avg_runs'] = np.where(league_balls['competition'] == 'WT20I', league_balls['0_x'], league_balls['new_0_x'])
    league_balls['avg_wkts'] = np.where(league_balls['competition'] == 'WT20I', league_balls['0_y'], league_balls['new_0_y'])

    league_balls['weight'] = np.where(league_balls['balls_bowled'] > 20000, 1, league_balls['balls_bowled'] / 20000)
    league_balls['weight_2'] = np.where(league_balls['balls_bowled'] > 20000, 1, league_balls['balls_bowled'] / 20000)
    league_balls['weight'] = np.where(league_balls['competition'] == 'tier_2', league_balls['weight_2'], league_balls['weight'])

    league_balls['runs'] = (league_balls['weight'] * league_balls['0_x']) + ((1 - league_balls['weight']) * league_balls['avg_runs'])
    league_balls['wkts'] = (league_balls['weight'] * league_balls['0_y']) + ((1 - league_balls['weight']) * league_balls['avg_wkts'])

    run_params = rep_run_ratio_model.params.copy()
    wkt_params = rep_wkt_ratio_model.params.copy()

    for _, row in league_balls.iterrows():
        if row['feature_name'] in run_params.index and pd.notna(row['runs']):
            run_params.loc[row['feature_name']] = row['runs']

        if row['feature_name'] in wkt_params.index and pd.notna(row['wkts']):
            wkt_params.loc[row['feature_name']] = row['wkts']

    params = pd.merge(pd.DataFrame(run_params), pd.DataFrame(wkt_params), how='left', left_index=True, right_index=True).reset_index()
    aux = pd.DataFrame([['λ', str(run_transformer.lambdas_[0]), str(wkt_transformer.lambdas_[0])]], columns=params.columns)
    params = pd.concat([params, aux], axis=0)

    # -------------------------
    # 6) Predict ratings outputs
    # -------------------------
    ratings, X_run_r, X_wkt_r = build_ratings_features_bowl(ratings, run_transformer, wkt_transformer)

    ratings.insert(ratings.columns.get_loc('run_rating') + 1, 'rep_run_ratio', X_run_r.to_numpy() @ run_params.to_numpy())
    ratings['i_rep_runs'] = ratings['rep_run_ratio'] * ratings['i_realexprbowl']

    ratings.insert(ratings.columns.get_loc('wkt_rating') + 1, 'rep_wkt_ratio', X_wkt_r.to_numpy() @ wkt_params.to_numpy())
    ratings['i_rep_wkt'] = ratings['rep_wkt_ratio'] * ratings['i_realexpwbowl']

    # -------------------------
    # 7) One-player breakdown
    # -------------------------
    if x == 1:
        pass
    else:
        debug_mask = (
            (ratings['bowler'] == 'Alexa Stonehouse') &
            (ratings['competition'] == 'WT20I') &
            (ratings['host'] == 'England')
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


