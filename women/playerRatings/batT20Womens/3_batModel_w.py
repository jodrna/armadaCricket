import pandas as pd
import numpy as np
from batFunctions_w import buildRunRatingsMapPriority, buildWktRatingsMapPriority


# t20 only issues
# players who play their first 2 games on the same day, e.g., final's day, 2 ratingsT20 will be missing instead of just 1 for their 'first game'
# players who only ever play ODI, will appear in the ratingsT20 for because it comes from the dummy ratingsT20 for values, but won't be any t20 data to rate them

# with ODI data issue's
# players who only ever played ODI's will have a rating for T20's today because it can use the ODI data
# players who play ODI before a t20 will have a rating for their first t20 game
# players who play ODI first but only play in the opening overs, means they will have a wicket rating but not a run rating

# outputs are longer than ratings_player_r because it also has a players first innings which we just assign a 1 rating

for x in np.arange(0, 2, 1):
    # read data from documents
    bat_data = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/data/combinedBatDataClean.csv', parse_dates=['date', 'dob'])
    n2h_factors = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/auxiliaries/batN2HFactors.csv')
    n2h_factors = n2h_factors.loc[:, ['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]
    # coeff_adjust = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/auxiliaries/batN2HFactorsCoeffs.csv')
    # coeff_values = coeff_adjust.mean().to_dict()  # this makes a dictionary of values, the mean of each column (there is only one value for each column) named the name of each column. I can call on these values later

    # filter a specific player
    # bat_data = bat_data[(bat_data['batsman'] == 'Imran Tahir')]
    # bat_data = bat_data[(bat_data['playerid'] == 489889)]

    # split odi innings
    bat_data['competition'] = np.where(bat_data['competition'] == 'ODI', np.where(bat_data['ballsremaining'] < 84, 'ODI2', 'ODI1'), bat_data['competition'])

    # make format in the dummy games 'T20'
    bat_data['format'] = bat_data['format'].fillna('t20')

    # we create 2 dataframes and merge so we have all innings, we need 2 because creating a pivot excludes the dummy innings, so to get them do a remove duplicates on bat data
    innings_info = bat_data.loc[:, ['date', 'matchid', 'playerid', 'batsman', 'nationality', 'competition', 'host', 'host_region', 'balls_faced_career',
                                    'balls_faced_host']].drop_duplicates(['matchid', 'playerid', 'date', 'host', 'competition'])
    innings_perf = pd.pivot_table(bat_data, values=['balls_faced', 'runs', 'realexprbat', 'wkt', 'realexpwbat'], index=['date', 'matchid', 'playerid', 'competition', 'host', 'ord'], aggfunc='sum').reset_index()
    innings = innings_info.merge(innings_perf, how='left', left_on=['date', 'matchid', 'playerid', 'competition', 'host'], right_on=['date', 'matchid', 'playerid', 'competition', 'host'])

    # now we have df of all innings, we merge them to themselves, so for every inning we have all other innings also, and then drop the ones after the date of the predicting inning
    # lookbacks is every innings a player has played duplicated *x, x being the number of innings he has played total
    lookbacks_player = innings.set_index('playerid').merge(innings.set_index('playerid'), how='left', left_index=True, right_index=True, suffixes=('', '_2')).reset_index()
    lookbacks_player = lookbacks_player[lookbacks_player['date'] > lookbacks_player['date_2']]  # date is the date of the innings we are predicting, drop innings after it
    lookbacks_player = lookbacks_player[~lookbacks_player['competition'].isin(['ODI1', 'ODI2'])]

    # Calculate the difference in days and create a new column
    lookbacks_player['date'] = pd.to_datetime(lookbacks_player['date'])
    lookbacks_player['date_2'] = pd.to_datetime(lookbacks_player['date_2'])
    lookbacks_player['days_ago'] = (lookbacks_player['date'] - lookbacks_player['date_2']).dt.days
    lookbacks_player['balls_ago'] = lookbacks_player['balls_faced_career'] - lookbacks_player['balls_faced_career_2']

    # don't actully use these but the function needs them for the outputs as its just the same as the optimisation function
    bat_weightings = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/auxiliaries/batWeightings.csv')
    bat_data = bat_data.merge(bat_weightings, on='balls_faced_career', how='left')
    bat_data['runs_weight_curve'] = bat_data['runs_weight_curve'].fillna(1)
    bat_data['wkts_weight_curve'] = bat_data['wkts_weight_curve'].fillna(1)

    # avg ord will effect the rating
    avg_ord = bat_data.groupby(['playerid', 'batsman'])['ord'].mean().reset_index()
    avg_ord.rename(columns={'ord': 'avg_ord'}, inplace=True)
    lookbacks_player = lookbacks_player.merge(avg_ord, on=('playerid', 'batsman'), how='left')


    # run ratingsT20
    if x == 0:
        param = [15.17348002,	6.89753,	12.88380525,	6.307514689,	1.692501798,	1.000755457,	0.621652589,	1.241297181,	1.381979717,	0.000496309]
    else:
        param = [20,	12.59457633,	17.46079646,	7.338761994,	2.72804768,	1,	0.469010748,	1,	1.444896715,	0.000802191]
    ratings_player_r, lookbacks_player_r = buildRunRatingsMapPriority(param, lookbacks_player)


    # wkt ratingsT20, this is all the exact same as above but for wkts this time
    if x == 0:
        param = [1.293158205,	3.979704835,	1.80296017,	2.070824639,	1.518191934,	1.071059076,	0.97542588,	1.477165773,	1,	0.000499531]
    else:
        param = [10.5339096,	20,	5.233032976,	7.782822175,	1,	4.423330281,	0.977275005,	1.589376541,	1.080660285,	0.000850588]
    ratings_player_w, lookbacks_player_w = buildWktRatingsMapPriority(param, lookbacks_player)



    # drop the odi data now that outputs are done
    bat_data = bat_data[(bat_data['format'] == 't20')]
    # merge the wkt and run ratingsT20 tables
    ratings_player = pd.merge(ratings_player_r.drop(labels=['realexprbat_2', 'runs_2', 'weight_exprbat', 'weight_runs'], axis=1),
                              ratings_player_w.drop(labels=['realexpwbat_2', 'wkt_2', 'weight_expwbat', 'weight_wkt'], axis=1),
                              how='left', on=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'], suffixes=('_r', '_w'))

    # merge some info to the ratingsT20 which isn't there, I don't include this at the beginning to speed up the process above, less columns means faster merging
    ratings_player = ratings_player.merge(bat_data.loc[:, ['date', 'matchid', 'battingteam', 'playerid', 'batsman', 'age', 'nationality', 'home_region', 'host', 'host_region', 'H/A_competition',
                                                           'H/A_country', 'H/A_region', 'competition', 'overseas_pct', 'careerT20MatchNumber']].drop_duplicates(subset=['date', 'matchid', 'playerid', 'host', 'competition']),
                                          how='outer', on=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'])


    # merge how the batter actually performed in the past innings we rated
    innings_perf = pd.pivot_table(bat_data, values=['balls_faced', 'balls_faced_career', 'balls_faced_host', 'runs', 'wkt', 'realexprbat', 'realexpwbat', 'ord'],
                                  index=['date', 'playerid', 'matchid', 'batsman', 'host', 'competition'],
                                  aggfunc={'balls_faced': 'sum', 'balls_faced_career': 'min', 'balls_faced_host': 'min', 'runs': 'sum', 'wkt': 'sum', 'realexprbat': 'sum', 'realexpwbat': 'sum', 'ord': 'mean'}).reset_index()
    innings_perf['i_run_ratio'] = innings_perf['runs'] / innings_perf['realexprbat']
    innings_perf['i_wkt_ratio'] = innings_perf['wkt'] / innings_perf['realexpwbat']

    # when we do the merge to innings performance we lose the innings which haven't happened which are the ratingsT20 we actually want, so do a merge to ratingsT20 player, outer merge includes everything in both tables
    ratings = innings_perf.merge(ratings_player, how='outer', on=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'])
    ratings = ratings[~ratings['competition'].isin(['ODI1', 'ODI2'])]

    # order and rename columns
    ratings = ratings.loc[:, ['date', 'matchid', 'battingteam', 'playerid', 'batsman', 'age', 'nationality', 'home_region', 'host', 'host_region', 'H/A_competition', 'H/A_country', 'H/A_region', 'competition',
                              'careerT20MatchNumber', 'balls_faced_career', 'balls_faced_host', 'overseas_pct', 'balls_faced_2_r', 'ord_2_r', 'z_run_ratio', 'run_rating', 'weight_balls_r',
                              'balls_faced_2_w', 'ord_2_w', 'z_wkt_ratio', 'wkt_rating', 'weight_balls_w', 'balls_faced', 'ord', 'realexprbat', 'runs',
                              'i_run_ratio', 'realexpwbat', 'wkt', 'i_wkt_ratio']]

    ratings = ratings.rename(columns={'balls_faced_2_r': 'balls_faced_r', 'ord_2_r': 'ord_r', 'balls_faced_2_w': 'balls_faced_w', 'ord_2_w': 'ord_w', 'balls_faced': 'i_balls_faced', 'ord': 'i_ord',
                                      'realexprbat': 'i_realexprbat', 'runs': 'i_runs', 'realexpwbat': 'i_realexpwbat', 'wkt': 'i_wkt'})
    ratings['i_ord'] = ratings['i_ord'].round(0) # sometimes the ord value changes throughout someone's innings for some reason

    # there is no rating for the first innings in a players career for obvious reasons, make it 1 for now
    ratings['run_rating'], ratings['wkt_rating'] = ratings['run_rating'].fillna(1), ratings['wkt_rating'].fillna(1)
    ratings['balls_faced_r'], ratings['balls_faced_w'] = ratings['balls_faced_r'].fillna(1), ratings['balls_faced_w'].fillna(1)
    ratings['ord_r'], ratings['ord_w'] = ratings['ord_r'].fillna(ratings['i_ord']), ratings['ord_w'].fillna(ratings['i_ord'])



    # this is just a check of the number of ratingsT20 being produced
    ratecount = pd.pivot_table(ratings, values=['matchid'], aggfunc={'matchid': 'count'}, index=['playerid', 'batsman']).reset_index()
    ratingswcount = pd.pivot_table(ratings_player_w, values=['matchid'], aggfunc={'matchid': 'count'}, index=['playerid', 'batsman']).reset_index()
    ratingsrcount = pd.pivot_table(ratings_player_r, values=['matchid'], aggfunc={'matchid': 'count'}, index=['playerid', 'batsman']).reset_index()
    ratecount = ratecount.merge(ratingswcount, how='outer', on=['playerid', 'batsman'], suffixes=('', '_w'))
    ratecount = ratecount.merge(ratingsrcount, how='outer', on=['playerid', 'batsman'], suffixes=('', '_r'))
    ratecount = ratecount.fillna(0)
    ratecount['wDiff'] = ratecount['matchid'] - ratecount['matchid_w']
    ratecount['rDiff'] = ratecount['matchid'] - ratecount['matchid_r']


    if x == 0:
        # recency weightings for each ball, this is uploaded directly to sql as bat recency weightings
        # recency will be the same regardless of comp and host, so just look at a random comp and host, t20i and west indies I chose, for no reason, this will isolate the games that go into the WI T20I rating
        recencies_r = lookbacks_player_r[(lookbacks_player_r['competition'] == 'WT20I') & (lookbacks_player_r['host'] == 'West Indies') &
                                         (lookbacks_player_r['date'] == (lookbacks_player_r['date'].max()))].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_faced_2']]
        # now we have all the previous games for a player that are used in their rating, for each match multiply balls faced by recency weight for that particular date
        recencies_r['recency_weight_match_sum'] = recencies_r['recency_weight'] * recencies_r['balls_faced_2']
        # sum up the total recency weight for a player because it doesn't always equal 1, we can't assume it is 1
        recencies_t = pd.pivot_table(recencies_r, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
        # merge in the total weight beside each innings
        recencies_r = recencies_r.merge(recencies_t, how='left', on=['playerid'])
        # then divide the total weight by the match weight by the balls to get the weight for each ball
        recencies_r['recency_weight_bbb_runs'] = recencies_r['recency_weight_match_sum_x'] / recencies_r['recency_weight_match_sum_y'] / recencies_r['balls_faced_2']

        # now for wickets, same as above
        recencies_w = lookbacks_player_w[(lookbacks_player_w['competition'] == 'WT20I') & (lookbacks_player_w['host'] == 'West Indies') &
                                            (lookbacks_player_r['date'] == (lookbacks_player_r['date'].max()))].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_faced_2']]
        recencies_w['recency_weight_match_sum'] = recencies_w['recency_weight'] * recencies_w['balls_faced_2']
        recencies_t = pd.pivot_table(recencies_w, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
        recencies_w = recencies_w.merge(recencies_t, how='left', on=['playerid'])
        recencies_w['recency_weight_bbb_wkt'] = recencies_w['recency_weight_match_sum_x'] / recencies_w['recency_weight_match_sum_y'] / recencies_w['balls_faced_2']
        # merge the runs and wickets
        recencies = pd.merge(recencies_r.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_runs']], recencies_w.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_wkt']], how='outer')
        recencies.to_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/recenciesJungle.csv', index=False)

        # export ratingsT20
        ratings.to_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/outputs/batRatingsPlayer.csv', index=False)
    else:
        ratings.to_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/outputs/batRatingsInnings.csv', index=False)




